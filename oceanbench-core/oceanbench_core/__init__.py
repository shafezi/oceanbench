"""
Core shared types and utilities for the OceanBench benchmark.

This package intentionally stays lightweight and dependency-minimal so that
other OceanBench components can depend on it without creating heavy cycles.
"""

from .types import Observation, ObservationBatch, QueryPoints, Scenario
from .graph import WaypointGraph
from .sampling import MeasurementItem, arrival_time, features_from_items, h, snap_times_to_available
from .eval_grid import EvalGrid, build_eval_grid, subsample_eval_grid

__all__ = [
    "Scenario",
    "Observation",
    "ObservationBatch",
    "QueryPoints",
    "WaypointGraph",
    "MeasurementItem",
    "arrival_time",
    "h",
    "features_from_items",
    "snap_times_to_available",
    "EvalGrid",
    "build_eval_grid",
    "subsample_eval_grid",
]

