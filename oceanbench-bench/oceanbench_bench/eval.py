"""
Evaluation harness for comparing field models on held-out truth.

Runs a list of field models on the same train/eval split, computes metrics,
and returns a structured result for logging and visualization.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_tasks.mapping.field_rmse import field_rmse_score

from .results import EvaluationResult, ModelRunSummary


def run_field_model_comparison(
    train_observations: ObservationBatch,
    query_points: QueryPoints,
    y_true: np.ndarray,
    models: List[Tuple[str, Any]],  # (name, FieldBeliefModel)
    *,
    seed: Optional[int] = None,
    scenario_name: Optional[str] = None,
    variable: Optional[str] = None,
) -> EvaluationResult:
    """
    Fit each model on train_observations, predict at query_points, and compare to y_true.

    Parameters
    ----------
    train_observations : ObservationBatch used for fitting.
    query_points : QueryPoints at which to evaluate (same order as y_true).
    y_true : Ground-truth values at query_points, shape (query_points.size,).
    models : List of (model_name, model_instance). Each model is reset then fitted.
    seed : Optional seed for reproducibility (models may use it).
    scenario_name : Optional label for the scenario.
    variable : Optional variable name for the field.

    Returns
    -------
    EvaluationResult with one ModelRunSummary per model (metrics, fit time, predict time).
    """
    n_eval = query_points.size
    if np.asarray(y_true).ravel().shape[0] != n_eval:
        raise ValueError("y_true length must equal query_points.size")

    y_true_flat = np.asarray(y_true, dtype=float).ravel()
    runs: List[ModelRunSummary] = []
    var = variable or (train_observations.variable if hasattr(train_observations, "variable") else "")

    for name, model in models:
        model.reset()
        if seed is not None and hasattr(model, "seed"):
            model.seed(seed)

        t0 = time.perf_counter()
        model.fit(train_observations)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = model.predict(query_points)
        predict_time = time.perf_counter() - t0

        y_pred = np.asarray(pred.mean, dtype=float).ravel()
        if y_pred.shape[0] != n_eval:
            raise RuntimeError(f"Model {name} returned prediction length {y_pred.shape[0]}, expected {n_eval}")

        metrics = field_rmse_score(y_true_flat, y_pred, include_mae=True)

        runs.append(ModelRunSummary(
            model_name=name,
            metrics=metrics,
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
            n_train=train_observations.size,
            n_eval=n_eval,
            config=dict(model.config) if hasattr(model, "config") else {},
        ))

    return EvaluationResult(
        scenario_name=scenario_name,
        variable=var,
        seed=seed,
        runs=runs,
        metadata={"n_train": train_observations.size, "n_eval": n_eval},
    )


def evaluate_binney_dense_grid(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    query_points: QueryPoints,
) -> Dict[str, float]:
    """
    Evaluate Binney-style runs on a dense grid using RMSE/MAE.

    This is a thin wrapper around ``field_rmse_score`` for consistency with
    the existing field-model evaluation API.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for evaluation.")
    return field_rmse_score(y_true, y_pred, query_points=query_points, include_mae=True)
