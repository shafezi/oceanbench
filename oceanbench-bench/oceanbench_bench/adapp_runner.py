"""Evaluation runner for AdaPP experiments.

Provides ``run_adapp_episode`` (single run) and ``run_adapp_time_budget_sweep``
(multiple time budgets, AdaPP vs lawnmower) with RMSE/MAE evaluation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_policies.ipp.adapp import (
    AdaPPConfig,
    AdaPPPlanner,
    LawnmowerBaseline,
    build_field_model,
)


def run_adapp_episode(
    graph: WaypointGraph,
    region: Mapping[str, float],
    truth_field: Any,
    config: AdaPPConfig,
    time_budget: float,
    eval_query_points: QueryPoints,
    truth_values: np.ndarray,
    *,
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    variable: str = "temp",
    prior_obs: Any = None,
) -> dict[str, Any]:
    """Run one AdaPP episode and evaluate against truth.

    Returns dict with: trajectory, n_steps, time_used, rmse, mae, runtime_s.
    """
    if start_lat is None:
        start_lat = 0.5 * (float(region["lat_min"]) + float(region["lat_max"]))
    if start_lon is None:
        start_lon = 0.5 * (float(region["lon_min"]) + float(region["lon_max"]))

    field_model = build_field_model(config, seed=config.seed)
    planner = AdaPPPlanner(graph, region, field_model, config, truth_field=truth_field)

    t0 = time.perf_counter()
    result = planner.run_episode(start_lat, start_lon, time_budget, variable=variable, prior_obs=prior_obs)
    runtime = time.perf_counter() - t0

    # Evaluate: fit final model on all collected observations, predict on eval grid.
    rmse, mae = np.nan, np.nan
    pred_mean, pred_std = None, None
    if result["observations"] is not None and result["observations"].size >= 3:
        try:
            eval_model = build_field_model(config, seed=config.seed)
            eval_model.fit(result["observations"])
            pred = eval_model.predict(eval_query_points)
            pred_mean = np.asarray(pred.mean, dtype=float).ravel()
            pred_std = np.asarray(pred.std, dtype=float).ravel() if pred.std is not None else None
            error = truth_values - pred_mean
            finite_mask = np.isfinite(error)
            if finite_mask.any():
                rmse = float(np.sqrt(np.mean(error[finite_mask] ** 2)))
                mae = float(np.mean(np.abs(error[finite_mask])))
        except Exception as e:
            logger.debug("[adapp_runner] Eval failed: %s", e)

    result["rmse"] = rmse
    result["mae"] = mae
    result["pred_mean"] = pred_mean
    result["pred_std"] = pred_std
    result["runtime_s"] = runtime
    result["time_budget"] = time_budget
    return result


def run_lawnmower_episode(
    graph: WaypointGraph,
    region: Mapping[str, float],
    truth_field: Any,
    config: AdaPPConfig,
    time_budget: float,
    eval_query_points: QueryPoints,
    truth_values: np.ndarray,
    *,
    start_lat: Optional[float] = None,
    start_lon: Optional[float] = None,
    variable: str = "temp",
) -> dict[str, Any]:
    """Run one lawnmower episode and evaluate against truth."""
    if start_lat is None:
        start_lat = 0.5 * (float(region["lat_min"]) + float(region["lat_max"]))
    if start_lon is None:
        start_lon = 0.5 * (float(region["lon_min"]) + float(region["lon_max"]))

    lm = LawnmowerBaseline(graph, speed_mps=config.speed_mps)
    rng = np.random.default_rng(config.seed)

    t0 = time.perf_counter()
    result = lm.run_episode(
        start_lat, start_lon, time_budget,
        truth_field=truth_field, variable=variable,
        noise_variance=config.noise_variance, rng=rng,
    )
    runtime = time.perf_counter() - t0

    rmse, mae = np.nan, np.nan
    if result["observations"] is not None and result["observations"].size >= 3:
        try:
            eval_model = build_field_model(config, seed=config.seed)
            eval_model.fit(result["observations"])
            pred = eval_model.predict(eval_query_points)
            error = truth_values - pred.mean
            finite_mask = np.isfinite(error)
            if finite_mask.any():
                rmse = float(np.sqrt(np.mean(error[finite_mask] ** 2)))
                mae = float(np.mean(np.abs(error[finite_mask])))
        except Exception as e:
            logger.debug("[lawnmower_runner] Eval failed: %s", e)

    result["rmse"] = rmse
    result["mae"] = mae
    result["runtime_s"] = runtime
    result["time_budget"] = time_budget
    return result


def run_adapp_time_budget_sweep(
    graph: WaypointGraph,
    region: Mapping[str, float],
    truth_field: Any,
    config: AdaPPConfig,
    time_budgets: Sequence[float],
    eval_query_points: QueryPoints,
    truth_values: np.ndarray,
    *,
    variable: str = "temp",
) -> dict[str, list[dict[str, Any]]]:
    """Run AdaPP and lawnmower for each time budget, return comparison.

    Returns dict with keys "adapp" and "lawnmower", each a list of
    per-budget result dicts.
    """
    adapp_results = []
    lm_results = []

    for T in time_budgets:
        print(f"  T={T:.0f}s — AdaPP...", end="", flush=True)
        ar = run_adapp_episode(
            graph, region, truth_field, config, T,
            eval_query_points, truth_values, variable=variable,
        )
        adapp_results.append(ar)
        print(f" RMSE={ar['rmse']:.4f}", end="")

        print(f" | Lawnmower...", end="", flush=True)
        lr = run_lawnmower_episode(
            graph, region, truth_field, config, T,
            eval_query_points, truth_values, variable=variable,
        )
        lm_results.append(lr)
        print(f" RMSE={lr['rmse']:.4f}")

    return {"adapp": adapp_results, "lawnmower": lm_results}
