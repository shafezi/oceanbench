"""
Core shared types and utilities for the OceanBench benchmark.

This package intentionally stays lightweight and dependency-minimal so that
other OceanBench components can depend on it without creating heavy cycles.
"""

from .types import Scenario, Observation, ObservationBatch, QueryPoints

__all__ = [
    "Scenario",
    "Observation",
    "ObservationBatch",
    "QueryPoints",
]

