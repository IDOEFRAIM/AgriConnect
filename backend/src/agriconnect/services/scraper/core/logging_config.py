"""
Configuration de logging structuré pour AWS CloudWatch.

Permet une observabilité complète du scraper en production.
"""
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    Formatter pour logs structurés (JSON) compatibles CloudWatch.
    
    Facilite le parsing et les recherches dans CloudWatch Logs Insights.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add context from record.__dict__ (custom fields)
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'extra_fields']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter pour ajouter du contexte automatiquement.
    
    Usage:
        logger = ContextAdapter(logging.getLogger(__name__), {"scraper": "GoogleDrive"})
        logger.info("Download started", extra={"url": url, "file_size": 1024})
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add context to log records."""
        # Merge context with extra fields
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = {"extra_fields": extra}
        
        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path = None,
    structured: bool = True,
    console_output: bool = True
):
    """
    Configure le système de logging.
    
    Args:
        log_level: Niveau minimum (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Répertoire pour les logs fichier (None = pas de fichier)
        structured: True = JSON structuré, False = format humain
        console_output: True = logs sur console/stdout
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logs généraux
        file_handler = logging.FileHandler(
            log_dir / f"scraper_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
        
        root_logger.addHandler(file_handler)
        
        # Error logs séparés
        error_handler = logging.FileHandler(
            log_dir / f"scraper_errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter() if structured else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        root_logger.addHandler(error_handler)
    
    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)
    
    logging.info("Logging configured", extra={
        "log_level": log_level,
        "structured": structured,
        "file_logging": log_dir is not None
    })


def get_logger(name: str, context: Dict[str, Any] = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Default context to add to all logs
    
    Returns:
        Logger or ContextAdapter with context
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger


# CloudWatch Logs Insights query examples for documentation
CLOUDWATCH_QUERIES = {
    "errors_by_scraper": """
        fields @timestamp, logger, message, scraper, url
        | filter level = "ERROR"
        | stats count() by scraper
        | sort count desc
    """,
    
    "slow_requests": """
        fields @timestamp, scraper, url, duration_ms
        | filter duration_ms > 5000
        | sort duration_ms desc
        | limit 20
    """,
    
    "success_rate_by_category": """
        fields @timestamp, category, status
        | stats count() by category, status
        | filter status in ["success", "failed"]
    """,
    
    "lambda_execution_summary": """
        fields @timestamp, execution_id, total_urls, processed_urls, successful_urls, failed_urls
        | filter message like /Execution completed/
        | sort @timestamp desc
        | limit 10
    """
}
