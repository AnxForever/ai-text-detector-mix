"""Shared logging configuration for all scripts.

Usage:
    from scripts.utils.log import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console output.

    Args:
        name: Logger name (typically ``__name__``).
        level: Minimum log level.

    Returns:
        A :class:`logging.Logger` instance with a stream handler attached
        (if one has not already been added).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
