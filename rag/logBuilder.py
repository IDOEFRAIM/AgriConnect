from __future__ import annotations
import logging
import logging.handlers
import threading
import time
import json
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from contextvars import ContextVar
import asyncio
import uuid
import sys

# ---------- Context propagation ----------
_request_ctx: ContextVar[Dict[str, Any]] = ContextVar("_request_ctx", default={})

def set_request_context(ctx: Dict[str, Any]) -> None:
    """Sets the request context for the current execution context (e.g., thread or async task)."""
    _request_ctx.set(ctx)

def get_request_context() -> Dict[str, Any]:
    """Retrieves a copy of the current request context."""
    return _request_ctx.get().copy()

# ---------- Config dataclasses ----------
@dataclass
class LoggerSinkConfig:
    """Configuration for various logging destinations (sinks)."""
    console: bool = True
    file_path: Optional[str] = None
    file_max_bytes: int = 10 * 1024 * 1024
    file_backup_count: int = 5
    json_format: bool = True
    level: str = "INFO"
    remote_http: Optional[str] = None
    sample_rate: float = 1.0  # 0..1 for DEBUG sampling

@dataclass
class LoggerConfig:
    """Overall configuration for the StructuredLogger."""
    service_name: str = "rag_service"
    env: str = "prod"
    host: str = socket.gethostname()
    sinks: LoggerSinkConfig = field(default_factory=LoggerSinkConfig)
    enrichers: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    error_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    redact_keys: List[str] = field(default_factory=lambda: ["password", "token", "secret", "ssn"])
    redact_patterns: List[str] = field(default_factory=list)
    persist_path: Optional[str] = None  # optional JSONL consign

# ---------- Scrubber ----------
class Scrubber:
    """Handles scrubbing of sensitive data based on key names and regex patterns."""
    def __init__(self, keys: List[str], patterns: List[str]):
        self.keys = set(k.lower() for k in keys)
        self.patterns = [re.compile(p) for p in patterns]

    def scrub(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for scrubbing a dictionary payload."""
        return self._scrub_dict(payload)

    def _scrub_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            lk = k.lower()
            if lk in self.keys:
                out[k] = "[REDACTED]"
            else:
                out[k] = self._scrub_value(v)
        return out

    def _scrub_value(self, v: Any) -> Any:
        if isinstance(v, dict):
            return self._scrub_dict(v)
        if isinstance(v, list):
            return [self._scrub_value(x) for x in v]
        s = str(v)
        for pat in self.patterns:
            if pat.search(s):
                # Only redact the matched pattern, not the whole string
                s = pat.sub("[REDACTED]", s)
        return s

# ---------- JSON formatter ----------
class JsonFormatter(logging.Formatter):
    """Formats log records into a standardized JSON structure."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        # Get extra attributes set via `extra={"extra": ...}` in log calls
        # This is where StructuredLogger passes its enriched/scrubbed context
        extra = getattr(record, "extra", {}).get("extra", {}) or {}
        base.update(extra)
        
        # Handle exception information
        if record.exc_info:
            try:
                base["exc"] = self.formatException(record.exc_info)
            except Exception:
                base["exc"] = "exception formatting failed"
        
        return json.dumps(base, ensure_ascii=False)

# ---------- HTTP handler (best-effort non-blocking) ----------
class _HTTPHandler(logging.Handler):
    """A logging handler for best-effort, non-blocking HTTP remote logging."""
    def __init__(self, url: str, timeout: float = 1.0):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self._session = None
        self._thread_pool = None

    def emit(self, record: logging.LogRecord) -> None:
        """Emits the log record by asynchronously posting it to the HTTP endpoint."""
        try:
            payload = self.format(record)
            
            # Lazy import requests and initialize session/threadpool
            if self._session is None:
                try:
                    import requests
                    from concurrent.futures import ThreadPoolExecutor
                    self._session = requests.Session()
                    self._thread_pool = ThreadPoolExecutor(max_workers=1) # Dedicated worker for non-blocking
                except ImportError:
                    # Fallback to sys.stderr if requests is missing
                    print("requests library not found, cannot use HTTP sink.", file=sys.stderr)
                    return
            
            # Use thread pool to send the request without blocking the main log flow
            self._thread_pool.submit(self._send_request, payload)
            
        except Exception:
            # Swallow any formatting/setup errors
            pass

    def _send_request(self, payload: str) -> None:
        """The actual blocking HTTP POST operation executed in a background thread."""
        try:
            self._session.post(
                self.url, 
                data=payload, 
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
        except Exception:
            # Swallow connection/timeout errors to ensure robustness
            pass

    def close(self) -> None:
        """Closes the session and shuts down the thread pool."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        if self._session:
            self._session.close()
        super().close()


# ---------- Structured logger ----------
class StructuredLogger:
    """
    A robust, structured logger that supports context propagation, redaction, 
    multiple sinks (console, file, HTTP), and metric/error hooks.
    """
    _instances: Dict[str, "StructuredLogger"] = {}
    _lock = threading.Lock()

    def __init__(self, name: str, cfg: LoggerConfig):
        self.name = name
        self.cfg = cfg
        # Python's logging module for handling formatters and sinks
        self._logger = logging.getLogger(name)
        level = getattr(logging, cfg.sinks.level.upper(), logging.INFO)
        self._logger.setLevel(level)
        self._scrubber = Scrubber(cfg.redact_keys, cfg.redact_patterns)
        self._setup_handlers()
        self._sample_rate = max(0.0, min(1.0, cfg.sinks.sample_rate))
        self._persist_lock = threading.Lock()
        # Thread for background flush (optional)
        self._bg_task: Optional[threading.Thread] = None

    @classmethod
    def get(cls, name: str, cfg: Optional[LoggerConfig] = None) -> "StructuredLogger":
        """Singleton-like accessor to retrieve or create a logger instance."""
        with cls._lock:
            if name in cls._instances:
                # Reconfigure if a new config is provided and is different
                current_cfg = cls._instances[name].cfg
                if cfg is not None and current_cfg != cfg:
                    cls._instances[name]._reconfigure(cfg)
                return cls._instances[name]
            
            # Create new instance
            if cfg is None:
                cfg = LoggerConfig()
            inst = StructuredLogger(name, cfg)
            cls._instances[name] = inst
            return inst
    
    def _reconfigure(self, cfg: LoggerConfig) -> None:
        """Applies a new configuration to an existing logger instance."""
        # Stop background flush if active, as handlers will be reset
        self.stop_background_flush() 
        
        self.cfg = cfg
        level = getattr(logging, cfg.sinks.level.upper(), logging.INFO)
        self._logger.setLevel(level)
        self._scrubber = Scrubber(cfg.redact_keys, cfg.redact_patterns)
        self._setup_handlers()
        self._sample_rate = max(0.0, min(1.0, cfg.sinks.sample_rate))

    def _setup_handlers(self) -> None:
        """Initializes and registers all configured handlers (sinks)."""
        # Remove existing handlers to ensure idempotency and prevent duplicate logs
        for h in list(self._logger.handlers):
            h.close() # Important for file and HTTP handlers
            self._logger.removeHandler(h)

        # Basic formatter for console, otherwise use JsonFormatter
        if self.cfg.sinks.json_format:
            fmt = JsonFormatter() 
        else:
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

        if self.cfg.sinks.console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)

        if self.cfg.sinks.file_path:
            fh = logging.handlers.RotatingFileHandler(
                filename=self.cfg.sinks.file_path,
                maxBytes=self.cfg.sinks.file_max_bytes,
                backupCount=self.cfg.sinks.file_backup_count,
                encoding="utf-8"
            )
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        if self.cfg.sinks.remote_http:
            rh = _HTTPHandler(self.cfg.sinks.remote_http)
            rh.setFormatter(fmt)
            self._logger.addHandler(rh)

    # ---------- public logging methods ----------
    def log_event(self, level: str, message: str, **kwargs) -> None:
        """The core logging method, handling enrichment, scrubbing, and hooks."""
        lvl = getattr(logging, level.upper(), logging.INFO)
        
        # Sampling for DEBUG level
        if lvl <= logging.DEBUG and self._sample_rate < 1.0:
            import random
            if random.random() > self._sample_rate:
                return
        
        # 1. Build and enrich the log context
        extra = self._build_extra(kwargs)
        
        # 2. Scrub sensitive fields
        try:
            extra_scrubbed = self._scrubber.scrub(extra)
        except Exception:
            extra_scrubbed = extra # fallback
        
        # 3. Emit to standard sinks (console, file, http)
        # We pass the scrubbed context as 'extra={"extra": ...}' which JsonFormatter extracts.
        self._logger.log(lvl, message, extra={"extra": extra_scrubbed})
        
        # 4. Metrics hook (triggered after successful processing)
        if self.cfg.metrics_hook:
            try:
                self.cfg.metrics_hook({
                    "service": self.cfg.service_name, 
                    "level": level, 
                    "message": message,
                    **extra_scrubbed # Include all extra fields for hooks
                })
            except Exception:
                pass
        
        # 5. Optional persist to JSONL
        if self.cfg.persist_path:
            # We persist the scrubbed, enriched payload
            payload_to_persist = {"ts": int(time.time()*1000), "level": level, "message": message, **extra_scrubbed}
            self._persist_jsonl(payload_to_persist)

    def info(self, message: str, **kwargs) -> None:
        """Logs an INFO level event."""
        self.log_event("INFO", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Logs a DEBUG level event."""
        self.log_event("DEBUG", message, **kwargs)

    def warn(self, message: str, **kwargs) -> None:
        """Logs a WARNING level event."""
        self.log_event("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Logs an ERROR level event."""
        self.log_event("ERROR", message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Logs an EXCEPTION level event, automatically including traceback."""
        extra = self._build_extra(kwargs)
        try:
            extra_scrubbed = self._scrubber.scrub(extra)
        except Exception:
            extra_scrubbed = extra
            
        # Logging the exception will automatically add 'exc_info' to the record
        self._logger.exception(message, extra={"extra": extra_scrubbed})
        
        # Metrics hook for exception/error
        if self.cfg.metrics_hook:
            try:
                self.cfg.metrics_hook({
                    "service": self.cfg.service_name, 
                    "level": "EXCEPTION", 
                    "message": message,
                    **extra_scrubbed
                })
            except Exception:
                pass
                
        # Persist logic (requires capturing the traceback if needed)
        if self.cfg.persist_path:
            import traceback
            # Capture the full traceback text
            exc_text = traceback.format_exc()
            payload_to_persist = {
                "ts": int(time.time()*1000), 
                "level": "EXCEPTION", 
                "message": message, 
                "exc": exc_text.strip(),
                **extra_scrubbed
            }
            self._persist_jsonl(payload_to_persist)

    def _build_extra(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Collects context variables, default metadata, and applies enrichers."""
        ctx = get_request_context()
        base: Dict[str, Any] = {
            "service": self.cfg.service_name,
            "env": self.cfg.env,
            "host": self.cfg.host,
            # Ensure a correlation_id is always present
            "correlation_id": ctx.get("correlation_id") or str(uuid.uuid4()), 
            "request_id": ctx.get("request_id"),
            "user_id": ctx.get("user_id"),
            "agent": ctx.get("agent"),
            "pipeline_stage": ctx.get("stage"),
        }
        
        # Copy user-provided fields, overwriting base fields if necessary
        base.update(fields or {}) 
        
        # Apply enrichers
        for enr in self.cfg.enrichers:
            try:
                # Pass a copy to the enricher to prevent accidental mutation of the base dictionary
                extra = enr(base.copy()) 
                if extra:
                    base.update(extra)
            except Exception:
                # ignore enricher errors
                pass
        
        # Remove fields that were None in context
        return {k: v for k, v in base.items() if v is not None}

    def _persist_jsonl(self, payload: Dict[str, Any]) -> None:
        """Synchronously appends a log entry to the JSONL persistence file."""
        try:
            # Synchronous write protected by a lock to ensure file integrity in multithreaded environments.
            with self._persist_lock:
                with open(self.cfg.persist_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            if self.cfg.error_hook:
                try:
                    self.cfg.error_hook({"event": "persist_failed", "error": str(e), "path": self.cfg.persist_path})
                except Exception:
                    pass

    def flush(self) -> None:
        """Flushes all buffered handlers."""
        for h in self._logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    # ---------- background flush helpers ----------
    def start_background_flush(self, interval_s: float = 2.0) -> None:
        """Starts a daemon thread to periodically flush log buffers."""
        # Check if already running (using a simple flag on the thread object)
        if self._bg_task and self._bg_task.is_alive():
            return
        
        # Target function for the background thread
        def _target():
            # Use an event object to control the thread's lifecycle
            while not self._bg_task._stopped.is_set():
                try:
                    self._bg_task._stopped.wait(interval_s) # Wait with a timeout
                    self.flush()
                except Exception:
                    # Report thread error via error hook, but keep the thread alive
                    if self.cfg.error_hook:
                        self.cfg.error_hook({"event": "flush_thread_error"})
        
        # Create and start the thread
        self._bg_task = threading.Thread(target=_target, daemon=True)
        # Add a custom stop event to the thread instance
        self._bg_task._stopped = threading.Event()
        self._bg_task.start()

    def stop_background_flush(self) -> None:
        """Stops the periodic flush thread gracefully."""
        if self._bg_task and self._bg_task.is_alive():
            try:
                self._bg_task._stopped.set() # Signal the thread to stop
                self._bg_task.join(timeout=0.2) # Give it a moment to exit
            except Exception:
                pass
            finally:
                self._bg_task = None # Clear reference

    def __del__(self):
        """Clean up on destruction (stops background thread)."""
        self.stop_background_flush()


# ---------- Example enricher ----------
def example_enricher(base: Dict[str, Any]) -> Dict[str, Any]:
    """Adds a static application version to all log entries."""
    return {"app_version": "1.0.0"}

# ---------- Usage example (only runs if script is executed directly) ----------
if __name__ == "__main__":
    # Configure logging at the root level to suppress internal warnings/errors from the logging module itself
    logging.basicConfig(level=logging.WARNING) 

    cfg = LoggerConfig(
        service_name="rag_orchestrator",
        env="dev",
        sinks=LoggerSinkConfig(
            console=True, 
            file_path="orchestrator.log", 
            json_format=True, 
            level="DEBUG", # Set console level to DEBUG
            sample_rate=1.0
        ), 
        enrichers=[example_enricher],
        # Example: redact credit card numbers
        redact_patterns=[r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"], 
        persist_path="logs.jsonl"
    )
    
    # Clean up previous log files for fresh run
    import os
    if os.path.exists(cfg.sinks.file_path):
        os.remove(cfg.sinks.file_path)
    if os.path.exists(cfg.persist_path):
        os.remove(cfg.persist_path)

    logger = StructuredLogger.get("orchestrator", cfg)
    # Start the background thread for periodic flushing
    logger.start_background_flush(interval_s=1.0)

    # Set context for the thread/request
    set_request_context({"request_id": "r1", "correlation_id": "c1", "user_id": "u42", "stage": "retrieve"})
    logger.info("Started retrieval", query="irrigation best practices", user_payment_token="secret_token_123")
    
    # Check redaction (user_payment_token is in redact_keys and CC number is in redact_patterns)
    logger.debug("Debug details, trying CC", debug_info={"step": "vector_search", "card": "4111222233334444", "user_token": "token_to_redact"})

    # Change context for a different request
    set_request_context({"request_id": "r2", "correlation_id": "c2", "user_id": "u43", "stage": "generate"})
    logger.warn("Generation taking longer than 500ms", latency_ms=650)
    
    # Log exception
    try:
        raise ValueError("sample error in component X")
    except Exception:
        # api_key is sensitive and should be redacted by default key list
        logger.exception("Something failed during response generation", extra_info={"phase": "LLM_call", "api_key": "my-secret-key-123", "password": "hunter2"})
    
    # Give background thread time to flush and ensure synchronous flush before exit
    time.sleep(1.5)
    logger.flush()
    logger.stop_background_flush()
    
    print("\n--- Example logs written to orchestrator.log and logs.jsonl ---")


    'ig'