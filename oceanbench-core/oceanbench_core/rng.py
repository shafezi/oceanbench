"""Seed and RNG helpers for reproducible experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a numpy default_rng(seed) for reproducible randomness."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """Set numpy global seed. Prefer passing a Generator to components instead."""
    np.random.seed(seed)
