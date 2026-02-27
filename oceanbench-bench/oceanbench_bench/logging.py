"""Simple logging helpers for benchmark runs."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "oceanbench_bench",
    level: int = logging.INFO,
    stream: Optional[sys.__class__] = None,
) -> logging.Logger:
    """Configure and return a logger for benchmark output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(stream or sys.stdout)
        h.setLevel(level)
        logger.addHandler(h)
    return logger
