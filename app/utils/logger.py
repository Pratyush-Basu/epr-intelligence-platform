"""
app/utils/logger.py
-------------------
Centralized structured logging configuration.

A single call to get_logger() returns a named logger that inherits the
root configuration set at application startup. This prevents ad-hoc
basicConfig() calls scattered across modules.
"""

import logging
import sys
from app.core.config import settings


def configure_logging() -> None:
    """
    Configure the root logger once at application startup.
    Called from app/main.py lifespan event.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — writes to stdout so Docker/systemd captures it
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on hot-reload
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Usage:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
