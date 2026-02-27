"""Common metrics for field and task evaluation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. Flattens arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same size")
    diff = y_true - y_pred
    valid = np.isfinite(diff)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(diff[valid] ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. Flattens arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same size")
    diff = np.abs(y_true - y_pred)
    valid = np.isfinite(diff)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(diff[valid]))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    include_mae: bool = True,
    include_rmse: bool = True,
) -> Dict[str, float]:
    """Return a dict of metric name -> value."""
    out: Dict[str, Any] = {}
    if include_rmse:
        out["rmse"] = rmse(y_true, y_pred)
    if include_mae:
        out["mae"] = mae(y_true, y_pred)
    return out
