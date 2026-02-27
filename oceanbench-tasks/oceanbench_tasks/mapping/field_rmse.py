"""
Minimal evaluation task: score field model predictions against held-out truth.

Used by the comparison pipeline and notebooks to compute RMSE/MAE.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from oceanbench_core.types import QueryPoints

from ..metrics.common import compute_metrics


def field_rmse_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    query_points: QueryPoints | None = None,
    include_mae: bool = True,
) -> Dict[str, float]:
    """
    Compare predicted field values to ground truth and return metrics.

    Parameters
    ----------
    y_true : (n,) array of ground-truth values at evaluation points.
    y_pred : (n,) array of predicted values at the same points.
    query_points : optional, for logging; not used in computation.
    include_mae : if True, include MAE in the returned dict.

    Returns
    -------
    Dict with at least "rmse" and optionally "mae".
    """
    return compute_metrics(
        y_true,
        y_pred,
        include_rmse=True,
        include_mae=include_mae,
    )
