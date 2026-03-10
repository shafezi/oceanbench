"""Runners for Binney-style waypoint-planning experiments."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import QueryPoints, Scenario
from oceanbench_data_provider import DataProvider
from oceanbench_env import OceanTruthField
from oceanbench_tasks.mapping.binney_task import BinneyMappingTask
from oceanbench_policies.ipp.grg import GRGConfig, GRGPlanner
from oceanbench_policies.baselines.greedy_myopic import (
    GreedyMyopicBinneyPlanner,
    GreedyMyopicConfig,
)
from oceanbench_policies.baselines.random_walk import (
    RandomWalkBinneyPlanner,
    RandomWalkConfig,
)
from oceanbench_policies.ipp.bruteforce_finite_horizon import (
    BruteforceConfig,
    BruteforceFiniteHorizonPlanner,
)
from oceanbench_policies.baselines.lawnmower import (
    LawnmowerBinneyPlanner,
    LawnmowerConfig,
)
from .logging import append_log, save_config, save_json, setup_logger


def _load_scenario_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return cfg.get("scenario", cfg)


def _flatten_truth_per_point(arr: np.ndarray, n_points: int) -> np.ndarray:
    """Reduce truth array to one value per point when dataset has depth dimension."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1 and arr.shape[0] == n_points:
        return arr[:, 0].ravel()  # use surface (first depth)
    out = arr.ravel()
    if out.size != n_points:
        raise ValueError(f"Truth array size {out.size} does not match {n_points} points.")
    return out


def _select_planner(
    policy_cfg: Mapping[str, Any],
    task: BinneyMappingTask,
) -> Any:
    policy_type = str(policy_cfg.get("type", "grg")).lower()
    if policy_type == "grg":
        grg_cfg = policy_cfg.get("planning", {})
        cfg = GRGConfig(
            depth=int(grg_cfg.get("depth", 2)),
            n_splits=int(grg_cfg.get("n_splits", 5)),
            split_strategy=str(grg_cfg.get("split_strategy", "uniform")),
        )
        return GRGPlanner(
            graph=task.graph,
            objective=task.objective,
            sampling_config=task.sampling_config,
            config=cfg,
        )
    if policy_type == "greedy_myopic":
        cfg = GreedyMyopicConfig(
            max_steps=int(policy_cfg.get("max_steps", 1_000)),
        )
        return GreedyMyopicBinneyPlanner(
            graph=task.graph,
            objective=task.objective,
            sampling_config=task.sampling_config,
            config=cfg,
        )
    if policy_type == "random_walk":
        cfg = RandomWalkConfig(
            max_steps=int(policy_cfg.get("max_steps", 1_000)),
        )
        return RandomWalkBinneyPlanner(
            graph=task.graph,
            objective=task.objective,
            sampling_config=task.sampling_config,
            config=cfg,
        )
    if policy_type == "bruteforce_finite_horizon":
        bcfg = policy_cfg.get("bruteforce", {})
        cfg = BruteforceConfig(
            max_depth=int(bcfg.get("max_depth", 4)),
            max_nodes=int(bcfg.get("max_nodes", 50)),
            max_paths=int(bcfg.get("max_paths", 10_000)),
        )
        return BruteforceFiniteHorizonPlanner(
            graph=task.graph,
            objective=task.objective,
            sampling_config=task.sampling_config,
            config=cfg,
        )
    if policy_type == "lawnmower":
        cfg = LawnmowerConfig(
            max_nodes=int(policy_cfg.get("max_nodes", 200)),
        )
        return LawnmowerBinneyPlanner(
            graph=task.graph,
            objective=task.objective,
            sampling_config=task.sampling_config,
            config=cfg,
        )
    raise ValueError(f"Unknown policy.type: {policy_type!r}")


def _truncate_path_by_budget(
    path: Sequence[int],
    graph: Any,
    budget_seconds: float,
) -> Tuple[Sequence[int], float]:
    """
    Truncate a planned path to the portion executable within a time budget.
    """
    if not path:
        return path, 0.0
    executed: list[int] = [int(path[0])]
    used = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge = graph.edge_attributes(int(u), int(v))
        dt = float(edge["time_s"])
        if used + dt > budget_seconds:
            break
        executed.append(int(v))
        used += dt
    return executed, used


def _posterior_mean_from_samples(
    task: BinneyMappingTask,
    samples: Sequence[Any],
    y_samples: np.ndarray,
) -> np.ndarray:
    """
    Compute posterior mean on the evaluation grid given sample locations/values.
    """
    if not samples:
        # No information; fall back to zeros.
        return np.zeros(task.eval_grid.size, dtype=float)

    A = task.features_from_measurements(samples)
    cov = task.covariance

    # Prepare training targets from sampled truth values.
    y_arr = np.asarray(y_samples, dtype=float)
    # Handle multi-depth: truth.query_array can return (n_points, n_depths) when dataset has depth
    if y_arr.ndim > 1 and y_arr.shape[0] == A.shape[0]:
        y_vec = y_arr[:, 0].ravel()  # use surface (first depth)
    else:
        y_vec = y_arr.reshape(-1)
    if y_vec.size != A.shape[0]:
        raise ValueError(
            f"y_samples size {y_vec.size} does not match {A.shape[0]} samples."
        )

    # Drop NaNs (or infs) from training targets and features.
    finite_mask = np.isfinite(y_vec)
    if A.size > 0:
        finite_mask &= np.all(np.isfinite(A), axis=1)
    if not np.all(finite_mask):
        A = A[finite_mask]
        y_vec = y_vec[finite_mask]

    if y_vec.size == 0:
        # All samples invalid; fall back to zeros on the eval grid.
        return np.zeros(task.eval_grid.size, dtype=float)

    # Optional hyperparameter fitting for kernel-based covariance backends.
    # Backends that do not override `fit` simply no-op.
    cov.fit(A, y_vec)

    Y = np.column_stack(
        [
            np.asarray(task.eval_grid.query_points.lats, dtype=float),
            np.asarray(task.eval_grid.query_points.lons, dtype=float),
        ]
    )
    if task.eval_grid.query_points.times is not None:
        t = (
            np.asarray(task.eval_grid.query_points.times, dtype="datetime64[ns]")
            .astype("int64")
            / 1e9
        )
        Y = np.column_stack([Y, t.astype(float)])

    Sigma_AA = cov.cov_block(A, A)
    Sigma_YA = cov.cov_block(Y, A)

    nA = Sigma_AA.shape[0]
    noise_var = float(
        task.sampling_config.get("measurement_noise_var", 1e-2)
    )
    Sigma_AA_noise = Sigma_AA + noise_var * np.eye(nA, dtype=float)
    try:
        alpha = np.linalg.solve(Sigma_AA_noise, y_vec)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(Sigma_AA_noise) @ y_vec

    mu_Y = Sigma_YA @ alpha
    return np.asarray(mu_Y, dtype=float).reshape(-1)


def run_binney_offline(
    config: Mapping[str, Any],
    *,
    run_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    """
    Offline mode: plan once with full budget and evaluate the resulting path.
    """
    logger = setup_logger()
    log_cfg = config.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", run_dir is not None))
    log_cfg = config.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", run_dir is not None))

    base_seed = int(config.get("seed", 42))
    rng = np.random.default_rng(base_seed)

    scenario_cfg = config.get("scenario", {})
    task_cfg = config.get("task", {})
    policy_cfg = config.get("policy", {})

    provider = DataProvider()
    task = BinneyMappingTask.from_configs(
        scenario_cfg=scenario_cfg,
        task_cfg=task_cfg,
        provider=provider,
        rng=rng,
    )

    planner = _select_planner(policy_cfg, task)

    planning_cfg = policy_cfg.get("planning", {})
    budget_seconds = float(planning_cfg.get("budget_seconds", 3600.0))
    tau0 = np.datetime64(task.time_config.get("tau0", "2014-01-01T00:00:00"), "ns")

    start_node = int(task_cfg.get("graph", {}).get("start", {}).get("node_id", 0))
    goal_node = int(task_cfg.get("graph", {}).get("goal", {}).get("node_id", task.graph.n_nodes - 1))

    if log_enabled:
        logger.info(
            "Running Binney offline: start=%s goal=%s budget=%s",
            start_node,
            goal_node,
            budget_seconds,
        )

    path, samples, gain = planner.plan(
        s=start_node,
        t=goal_node,
        B=budget_seconds,
        tau=tau0,
    )

    result: dict[str, Any] = {
        "path": path,
        "samples": samples,
        "n_samples": len(samples),
        "used_budget_seconds": task.graph.path_travel_time(path) if path else 0.0,
        "objective_value": float(gain),
    }

    # Evaluate against truth on the dense eval grid using OceanTruthField.
    scenario = task.scenario
    region = {
        "lon": [
            float(scenario.region["lon_min"]),
            float(scenario.region["lon_max"]),
        ],
        "lat": [
            float(scenario.region["lat_min"]),
            float(scenario.region["lat_max"]),
        ],
    }
    if scenario.time_range is None:
        raise ValueError("Scenario.time_range must be set for offline evaluation.")
    time_range = (
        str(scenario.time_range[0].astype("datetime64[s]")),
        str(scenario.time_range[1].astype("datetime64[s]")),
    )
    ds = provider.subset(
        product_id=scenario.metadata.get("product_id", ""),
        region=region,
        time=time_range,
        variables=[scenario.variable],
        depth_opts=None,
        target_grid=None,
        overwrite=False,
    )
    truth = OceanTruthField(dataset=ds, variable=scenario.variable, scenario=scenario)
    y_true_raw = truth.query_array(task.eval_grid.query_points, method="linear", bounds_mode="nan")
    n_eval = task.eval_grid.size
    y_true = _flatten_truth_per_point(y_true_raw, n_eval)

    # Query truth at sample locations to form a simple GP-style posterior mean.
    # To avoid pathological all-NaN samples when trajectories step slightly
    # outside the provider's time window, we clamp sample times to the
    # scenario's configured time_range (if available).
    if samples:
        times = np.array([s.time for s in samples], dtype="datetime64[ns]")
        if scenario.time_range is not None:
            t0 = np.datetime64(scenario.time_range[0], "ns")
            t1 = np.datetime64(scenario.time_range[1], "ns")
            times = np.clip(times, t0, t1)

        qp_samples = QueryPoints(
            lats=[s.lat for s in samples],
            lons=[s.lon for s in samples],
            times=times,
        )
        y_samples = truth.query_array(qp_samples, method="linear", bounds_mode="nan")
        y_pred = _posterior_mean_from_samples(task, samples, y_samples)
    else:
        y_samples = np.zeros(0, dtype=float)
        y_pred = np.zeros_like(y_true)

    from oceanbench_bench.eval import evaluate_binney_dense_grid

    metrics = evaluate_binney_dense_grid(
        y_true=y_true,
        y_pred=y_pred,
        query_points=task.eval_grid.query_points,
    )
    result["metrics"] = metrics

    if run_dir is not None:
        from .results import EvaluationResult, ModelRunSummary
        from .results import save_results

        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(run_dir, config)
        save_json(run_dir, "graph.json", task.graph.serialize_graph())
        save_json(run_dir, "path.json", {"path": path})
        save_json(run_dir, "metrics.json", metrics)

        model_summary = ModelRunSummary(
            model_name=policy_cfg.get("type", "grg"),
            metrics=metrics,
            fit_time_seconds=0.0,
            predict_time_seconds=0.0,
            n_train=0,
            n_eval=task.eval_grid.size,
            config=dict(config),
        )
        eval_result = EvaluationResult(
            scenario_name=scenario.name or "",
            variable=scenario.variable,
            seed=base_seed,
            runs=[model_summary],
            metadata={"n_eval": task.eval_grid.size},
        )
        save_results(eval_result, run_dir / "results.json")
        append_log(
            run_dir,
            {
                "event": "binney_offline_completed",
                "objective_value": float(gain),
                "n_samples": len(samples),
            },
        )

    return result


def run_binney_receding_horizon(
    config: Mapping[str, Any],
    *,
    run_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    """
    Receding-horizon mode: repeatedly plan for a finite horizon, execute a
    prefix of the plan, and replan until budget is exhausted.
    """
    logger = setup_logger()
    log_cfg = config.get("logging", {})
    log_enabled = bool(log_cfg.get("enabled", run_dir is not None))

    base_seed = int(config.get("seed", 42))
    rng = np.random.default_rng(base_seed)

    scenario_cfg = config.get("scenario", {})
    task_cfg = config.get("task", {})
    policy_cfg = config.get("policy", {})

    provider = DataProvider()
    task = BinneyMappingTask.from_configs(
        scenario_cfg=scenario_cfg,
        task_cfg=task_cfg,
        provider=provider,
        rng=rng,
    )

    planner = _select_planner(policy_cfg, task)
    if not isinstance(planner, GRGPlanner):
        raise ValueError("Receding-horizon mode currently requires policy.type='grg'.")

    planning_cfg = policy_cfg.get("planning", {})
    budget_seconds = float(planning_cfg.get("budget_seconds", 3600.0))
    horizon_seconds = float(planning_cfg.get("horizon_seconds", budget_seconds))
    execute_seconds = float(planning_cfg.get("execute_seconds", horizon_seconds))
    tau = np.datetime64(task.time_config.get("tau0", "2014-01-01T00:00:00"), "ns")

    start_node = int(task_cfg.get("graph", {}).get("start", {}).get("node_id", 0))
    goal_node = int(task_cfg.get("graph", {}).get("goal", {}).get("node_id", task.graph.n_nodes - 1))

    if log_enabled:
        logger.info(
            "Running Binney receding-horizon: start=%s goal=%s budget=%s horizon=%s execute=%s",
            start_node,
            goal_node,
            budget_seconds,
            horizon_seconds,
            execute_seconds,
        )

    remaining = budget_seconds
    current = start_node
    global_path: list[int] = [current]
    X_items: list[Any] = []
    visit_counts: dict[int, int] = {current: 1}

    while remaining > 0.0 and current != goal_node:
        horizon_B = min(remaining, horizon_seconds)
        path_plan, samples_plan, _ = planner.plan(
            s=current,
            t=goal_node,
            B=horizon_B,
            tau=tau,
            X_items=X_items,
        )
        if not path_plan:
            break

        exec_path, used = _truncate_path_by_budget(path_plan, task.graph, execute_seconds)
        if len(exec_path) <= 1 or used <= 0.0:
            break

        # Sample executed segment and update selected set.
        samples_exec = task.sample_path(exec_path, tau)
        X_items.extend(samples_exec)

        # Advance time and state.
        from oceanbench_core import arrival_time as _arrival_time

        tau = _arrival_time(exec_path, tau, task.graph)
        remaining -= used
        current = int(exec_path[-1])
        visit_counts[current] = visit_counts.get(current, 0) + 1
        if global_path:
            global_path.extend(exec_path[1:])
        else:
            global_path.extend(exec_path)

        if run_dir is not None:
            append_log(
                run_dir,
                {
                    "event": "receding_step",
                    "current_node": current,
                    "remaining_budget": remaining,
                    "tau": str(tau),
                    "step_path": exec_path,
                },
            )
        if log_enabled:
            logger.info(
                "Receding step: exec_path=%s used=%.3f current=%s remaining=%.3f",
                exec_path,
                used,
                current,
                remaining,
            )

        # Safeguard against oscillation / getting stuck at the same node.
        if visit_counts[current] > 3 and current != goal_node:
            if log_enabled:
                logger.info(
                    "Stopping receding-horizon: repeated visits to node %s (%s times).",
                    current,
                    visit_counts[current],
                )
            break

    # Evaluate final trajectory using the offline machinery.
    if run_dir is not None:
        save_config(run_dir, config)
        save_json(run_dir, "path.json", {"path": global_path})
        save_json(run_dir, "graph.json", task.graph.serialize_graph())

    # Build synthetic result compatible with offline format.
    # We reuse the offline posterior-mean computation for metrics.
    scenario = task.scenario
    region = {
        "lon": [
            float(scenario.region["lon_min"]),
            float(scenario.region["lon_max"]),
        ],
        "lat": [
            float(scenario.region["lat_min"]),
            float(scenario.region["lat_max"]),
        ],
    }
    if scenario.time_range is None:
        raise ValueError("Scenario.time_range must be set for receding-horizon evaluation.")
    time_range = (
        str(scenario.time_range[0].astype("datetime64[s]")),
        str(scenario.time_range[1].astype("datetime64[s]")),
    )
    ds = provider.subset(
        product_id=scenario.metadata.get("product_id", ""),
        region=region,
        time=time_range,
        variables=[scenario.variable],
        depth_opts=None,
        target_grid=None,
        overwrite=False,
    )
    truth = OceanTruthField(dataset=ds, variable=scenario.variable, scenario=scenario)

    y_true_raw = truth.query_array(task.eval_grid.query_points, method="linear", bounds_mode="nan")
    n_eval = task.eval_grid.size
    y_true = _flatten_truth_per_point(y_true_raw, n_eval)

    if X_items:
        qp_samples = QueryPoints(
            lats=[s.lat for s in X_items],
            lons=[s.lon for s in X_items],
            times=[s.time for s in X_items],
        )
        y_samples = truth.query_array(qp_samples, method="linear", bounds_mode="nan")
        y_pred = _posterior_mean_from_samples(task, X_items, y_samples)
    else:
        y_pred = np.zeros_like(y_true)

    from oceanbench_bench.eval import evaluate_binney_dense_grid

    metrics = evaluate_binney_dense_grid(
        y_true=y_true,
        y_pred=y_pred,
        query_points=task.eval_grid.query_points,
    )

    result: dict[str, Any] = {
        "path": global_path,
        "n_samples": len(X_items),
        "objective_value": float(task.objective.value(task.features_from_measurements(X_items))),
        "metrics": metrics,
    }

    if run_dir is not None:
        from .results import EvaluationResult, ModelRunSummary
        from .results import save_results

        model_summary = ModelRunSummary(
            model_name=policy_cfg.get("type", "grg"),
            metrics=metrics,
            fit_time_seconds=0.0,
            predict_time_seconds=0.0,
            n_train=0,
            n_eval=task.eval_grid.size,
            config=dict(config),
        )
        eval_result = EvaluationResult(
            scenario_name=scenario.name or "",
            variable=scenario.variable,
            seed=base_seed,
            runs=[model_summary],
            metadata={"n_eval": task.eval_grid.size},
        )
        save_results(eval_result, run_dir / "results.json")
        append_log(
            run_dir,
            {
                "event": "binney_receding_completed",
                "objective_value": result["objective_value"],
                "n_samples": len(X_items),
            },
        )

    return result

