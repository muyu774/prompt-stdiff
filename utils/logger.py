"""Logger helpers."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "prompt_stdiff") -> logging.Logger:
    """Create a stream logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
