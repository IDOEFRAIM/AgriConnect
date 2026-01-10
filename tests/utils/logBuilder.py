# rag/logger.py
from __future__ import annotations
import asyncio
import logging
import sys
import time
import json
from typing import Any, Dict, Optional
from contextvars import ContextVar

# Context var to carry request-specific fields (e.g., request_id, agent)
_log_context: ContextVar[Dict[str, Any]] = ContextVar("_log_context", default={})

# Default logger name
DEFAULT_LOGGER_NAME = "rag"

# Simple JSON formatter for structured logs
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Base payload
        payload: Dict[str, Any] = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge record.extra if present
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        # Merge context
        ctx = _log_context.get()
        if ctx:
            payload.setdefault("ctx", {}).update(ctx)
        # Add module/func/line for debugging
        payload["module"] = f"{record.module}:{record.funcName}:{record.lineno}"
        return json.dumps(payload, ensure_ascii=False)

# Human readable formatter (fallback)
class HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        ctx = _log_context.get()
        ctx_str = f" {json.dumps(ctx, ensure_ascii=False)}" if ctx else ""
        base = f"{ts} [{record.levelname}] {record.name}: {record.getMessage()}{ctx_str}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base

# Configure root logger once
def configure_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    stream: Optional[Any] = None,
) -> None:
    """
    Configure structured logging for the application.
    - level: logging level (logging.INFO, logging.DEBUG, ...)
    - json_output: if True, logs are emitted as JSON; else human readable.
    - stream: output stream (defaults to sys.stdout)
    """
    stream = stream or sys.stdout
    root = logging.getLogger()
    # Avoid duplicate handlers on repeated configure calls
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter())
    root.setLevel(level)
    root.addHandler(handler)
    # Silence noisy libraries by default
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# Get a module logger
def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or DEFAULT_LOGGER_NAME)

# Context helpers
def set_log_context(**kwargs: Any) -> None:
    """
    Replace the current logging context with provided key-values.
    Use for per-request or per-job context (request_id, agent, etc).
    """
    _log_context.set(dict(kwargs))

def update_log_context(**kwargs: Any) -> None:
    """
    Update existing context with new keys (merge).
    """
    ctx = dict(_log_context.get() or {})
    ctx.update(kwargs)
    _log_context.set(ctx)

def clear_log_context() -> None:
    _log_context.set({})

# Convenience structured logging helpers
def _log_with_extra(logger: logging.Logger, level: int, msg: str, **extra: Any) -> None:
    # Attach extra as attribute so formatter can pick it up
    logger.log(level, msg, extra={"extra": extra})

def info(msg: str, **extra: Any) -> None:
    _log_with_extra(get_logger(), logging.INFO, msg, **extra)

def debug(msg: str, **extra: Any) -> None:
    _log_with_extra(get_logger(), logging.DEBUG, msg, **extra)

def warning(msg: str, **extra: Any) -> None:
    _log_with_extra(get_logger(), logging.WARNING, msg, **extra)

def error(msg: str, **extra: Any) -> None:
    _log_with_extra(get_logger(), logging.ERROR, msg, **extra)

def exception(msg: str, **extra: Any) -> None:
    _log_with_extra(get_logger(), logging.ERROR, msg, **extra)

# Context manager to scope request context easily
class log_context:
    """
    Usage:
        with log_context(request_id="r1", agent="ingestor"):
            info("started")
            ...
    The previous context is restored on exit.
    """
    def __init__(self, **ctx: Any):
        self._new = ctx
        self._token = None
        self._prev = None

    def __enter__(self):
        self._prev = _log_context.get()
        merged = dict(self._prev or {})
        merged.update(self._new)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _log_context.reset(self._token)

# Example of adding timing diagnostics to logs
def log_timed(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Decorator to log function execution time and result summary.
    Usage:
        @log_timed()
        async def my_task(...):
            ...
    Works with sync and async functions.
    """
    def _decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def _wrapped(*args, **kwargs):
                lg = logger or get_logger(func.__module__)
                start = time.time()
                try:
                    res = await func(*args, **kwargs)
                    elapsed = time.time() - start
                    _log_with_extra(lg, level, f"{func.__name__} completed", elapsed_s=elapsed)
                    return res
                except Exception as e:
                    elapsed = time.time() - start
                    _log_with_extra(lg, logging.ERROR, f"{func.__name__} failed", elapsed_s=elapsed, error=str(e))
                    raise
            return _wrapped
        else:
            def _wrapped(*args, **kwargs):
                lg = logger or get_logger(func.__module__)
                start = time.time()
                try:
                    res = func(*args, **kwargs)
                    elapsed = time.time() - start
                    _log_with_extra(lg, level, f"{func.__name__} completed", elapsed_s=elapsed)
                    return res
                except Exception as e:
                    elapsed = time.time() - start
                    _log_with_extra(lg, logging.ERROR, f"{func.__name__} failed", elapsed_s=elapsed, error=str(e))
                    raise
            return _wrapped
    return _decorator