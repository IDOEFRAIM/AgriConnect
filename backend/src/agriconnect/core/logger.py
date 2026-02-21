"""
Logger centralisé pour AgriConnect.

Usage:
    from backend.core.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure le logging une seule fois, appelé au startup."""
    global _configured
    if _configured:
        return

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", encoding="utf-8"),
        ],
    )

    # Réduire le bruit des librairies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger standard Python, nommé par module."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
