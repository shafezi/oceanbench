"""Runners for Binney-style waypoint-planning experiments."""

from __future__ import annotations

import logging
import time
from pathlib import Path
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


def _haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    depth1: Optional[float] = None,
    depth2: Optional[float] = None,
) -> float:
    """Distance in meters, optionally including depth via Pythagorean combination."""
    r_earth = 6_371_000.0
    lat1_r, lon1_r = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2_r, lon2_r = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    horiz = float(r_earth * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))
    if depth1 is not None and depth2 is not None:
        dz = float(depth2) - float(depth1)
        return float(np.sqrt(horiz**2 + dz**2))
    return horiz


def _load_scenario_config(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return cfg.get("scenario", cfg)


def _flatten_truth_per_point(
    arr: np.ndarray,
    n_points: int,
    depth_index: int = 0,
) -> np.ndarray:
    """Reduce truth array to one value per point when dataset has depth dimension.

    Parameters
    ----------
    depth_index:
        Which depth level to select when the array has a depth axis.
        Defaults to 0 (surface).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1 and arr.shape[0] == n_points:
        idx = min(depth_index, arr.shape[1] - 1)
        return arr[:, idx].ravel()
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
            seed=policy_cfg.get("seed"),
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


def _posterior_mean_std_from_samples(
    task: BinneyMappingTask,
    samples: Sequence[Any],
    y_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GP posterior mean and std on the evaluation grid.

    Uses a constant prior mean equal to the sample mean of the observed
    values.  Returns (mu_Y, std_Y).
    """
    n_eval = task.eval_grid.size
    if not samples:
        return np.zeros(n_eval, dtype=float), np.full(n_eval, np.nan, dtype=float)

    A = task.features_from_measurements(samples)
    cov = task.covariance

    # Prepare training targets from sampled truth values.
    y_arr = np.asarray(y_samples, dtype=float)
    if y_arr.ndim > 1 and y_arr.shape[0] == A.shape[0]:
        y_vec = y_arr[:, 0].ravel()
    else:
        y_vec = y_arr.reshape(-1)
    if y_vec.size != A.shape[0]:
        raise ValueError(
            f"y_samples size {y_vec.size} does not match {A.shape[0]} samples."
        )

    finite_mask = np.isfinite(y_vec)
    if A.size > 0:
        finite_mask &= np.all(np.isfinite(A), axis=1)
    if not np.all(finite_mask):
        A = A[finite_mask]
        y_vec = y_vec[finite_mask]

    if y_vec.size == 0:
        return np.zeros(n_eval, dtype=float), np.full(n_eval, np.nan, dtype=float)

    cov.fit(A, y_vec)

    # Use only spatial features (lat, lon) for covariance computation.
    # Including time causes mismatched time references between eval grid
    # and training samples, producing zero cross-covariance in static mode.
    A = A[:, :2]  # keep only lat, lon
    Y = np.column_stack(
        [
            np.asarray(task.eval_grid.query_points.lats, dtype=float),
            np.asarray(task.eval_grid.query_points.lons, dtype=float),
        ]
    )

    Sigma_AA = cov.cov_block(A, A)
    Sigma_YA = cov.cov_block(Y, A)

    nA = Sigma_AA.shape[0]
    noise_var = float(
        task.sampling_config.get("measurement_noise_var", 1e-2)
    )
    Sigma_AA_noise = Sigma_AA + noise_var * np.eye(nA, dtype=float)

    prior_mean = float(np.mean(y_vec))

    try:
        L = np.linalg.cholesky(Sigma_AA_noise)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_vec - prior_mean))
        V = np.linalg.solve(L, Sigma_YA.T)  # (nA, n_eval)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(Sigma_AA_noise) @ (y_vec - prior_mean)
        V = np.linalg.pinv(Sigma_AA_noise) @ Sigma_YA.T

    mu_Y = prior_mean + Sigma_YA @ alpha

    # Posterior variance: diag(Sigma_YY) - diag(Sigma_YA @ inv(Sigma_AA_noise) @ Sigma_AY)
    if hasattr(cov, "diag_cov"):
        prior_var = np.asarray(cov.diag_cov(Y), dtype=float)
    else:
        prior_var = np.diag(cov.cov_block(Y, Y)).astype(float)
    var_reduction = np.sum(V ** 2, axis=0)  # diag(V^T V) = diag(Sigma_YA inv(Sigma_AA) Sigma_AY)
    var_Y = np.clip(prior_var - var_reduction, 0.0, None)
    std_Y = np.sqrt(var_Y)

    return np.asarray(mu_Y, dtype=float).reshape(-1), np.asarray(std_Y, dtype=float).reshape(-1)


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
        y_samples = np.asarray(y_samples, dtype=float)
        noise_var = float(task.sampling_config.get("measurement_noise_var", 1e-2))
        y_samples = y_samples + rng.normal(0.0, np.sqrt(noise_var), size=y_samples.shape)
        y_pred, _ = _posterior_mean_std_from_samples(task, samples, y_samples)
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
    prior_obs: Optional[Any] = None,
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
    if not hasattr(planner, "plan") or "X_items" not in planner.plan.__code__.co_varnames:
        raise ValueError(
            "Receding-horizon mode requires a planner whose plan() accepts X_items "
            "(e.g. GRG or brute-force finite-horizon)."
        )

    planning_cfg = policy_cfg.get("planning", {})
    budget_seconds = float(planning_cfg.get("budget_seconds", 3600.0))
    horizon_seconds = float(planning_cfg.get("horizon_seconds", budget_seconds))
    execute_seconds = float(planning_cfg.get("execute_seconds", horizon_seconds))
    tau = np.datetime64(task.time_config.get("tau0", "2014-01-01T00:00:00"), "ns")

    start_node = int(task_cfg.get("graph", {}).get("start", {}).get("node_id", 0))
    goal_node = int(task_cfg.get("graph", {}).get("goal", {}).get("node_id", task.graph.n_nodes - 1))

    # goal_node == -1 means "explore without a fixed goal".
    # Goal is selected dynamically per horizon window inside the loop.
    dynamic_goal = goal_node < 0

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

    # Pre-populate with shared prior observations if provided.
    # Use tau as the time for prior items so features match the planning frame.
    if prior_obs is not None and prior_obs.size >= 1:
        from oceanbench_core.sampling import MeasurementItem
        for k in range(prior_obs.size):
            X_items.append(MeasurementItem(
                lat=float(prior_obs.lats[k]),
                lon=float(prior_obs.lons[k]),
                time=tau,
            ))

    while remaining > 0.0:
        horizon_B = min(remaining, horizon_seconds)

        # Pick goal for this horizon window.
        if dynamic_goal:
            n_nodes = task.graph.n_nodes
            candidates = list(range(0, n_nodes, max(1, n_nodes // 40)))
            if current in candidates:
                candidates.remove(current)
            best_goal = candidates[0] if candidates else current
            best_time = 0.0
            for candidate in candidates:
                try:
                    sp = task.graph.shortest_path(current, candidate)
                    tt = task.graph.path_travel_time(sp)
                    if tt <= horizon_B * 0.9 and tt > best_time:
                        best_time = tt
                        best_goal = candidate
                except Exception:
                    continue
            iter_goal = best_goal
        else:
            iter_goal = goal_node
            if current == goal_node:
                break

        path_plan, samples_plan, _ = planner.plan(
            s=current,
            t=iter_goal,
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
        times = np.array([s.time for s in X_items], dtype="datetime64[ns]")
        if scenario.time_range is not None:
            t0 = np.datetime64(scenario.time_range[0], "ns")
            t1 = np.datetime64(scenario.time_range[1], "ns")
            times = np.clip(times, t0, t1)
        qp_samples = QueryPoints(
            lats=[s.lat for s in X_items],
            lons=[s.lon for s in X_items],
            times=times,
        )
        y_samples = truth.query_array(qp_samples, method="linear", bounds_mode="nan")
        y_samples = np.asarray(y_samples, dtype=float)
        noise_var = float(task.sampling_config.get("measurement_noise_var", 1e-2))
        y_samples = y_samples + rng.normal(0.0, np.sqrt(noise_var), size=y_samples.shape)
        y_pred, y_std = _posterior_mean_std_from_samples(task, X_items, y_samples)
    else:
        y_pred = np.zeros_like(y_true)
        y_std = np.full_like(y_true, np.nan)

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
        "pred_mean": y_pred,
        "pred_std": y_std,
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


def run_persistent_sampling(
    config: Mapping[str, Any],
    *,
    run_dir: Optional[Path] = None,
    prior_obs: Optional[Any] = None,
) -> Mapping[str, Any]:
    """
    Online loop for data-driven learning and planning environmental sampling.
    """
    logger = setup_logger()
    cfg = dict(config)
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    scenario_cfg = dict(cfg.get("scenario", {}))
    candidates_cfg = dict(cfg.get("candidates", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    planner_cfg = dict(cfg.get("planner", {}))
    routing_cfg = dict(cfg.get("routing", {}))
    truth_cfg = dict(cfg.get("truth", {}))
    hyper_cfg = dict(cfg.get("hyperparams", {}))
    replan_cfg = dict(cfg.get("replan", {}))
    mission_cfg = dict(cfg.get("mission", {}))
    field_cfg = dict(cfg.get("field_model", {}))

    provider = DataProvider(config=cfg.get("provider_config"))
    scenario = _scenario_from_config(scenario_cfg)
    dataset = _provider_subset_for_scenario(provider, scenario_cfg=scenario_cfg, scenario=scenario)
    truth = OceanTruthField(dataset=dataset, variable=scenario.variable, scenario=scenario)

    from oceanbench_tasks.mapping.persistent_sampling import (
        build_candidate_points,
        build_eval_spatial_grid,
        resolve_eval_times,
        route_distance,
        score_persistent_sampling,
    )
    from oceanbench_policies.routing import solve_tsp_route
    from oceanbench_models.belief.field.hyperparams import HyperparameterScheduler

    candidate_points = build_candidate_points(
        scenario,
        candidates_cfg,
        seed=int(candidates_cfg.get("seed", seed)),
    )
    eval_spatial = build_eval_spatial_grid(
        scenario,
        eval_cfg=eval_cfg,
        seed=seed,
    )

    model = _build_field_model_for_persistent(
        field_model_cfg=field_cfg,
        root_cfg=cfg,
        seed=seed,
    )
    planner = _build_mi_planner(
        planner_cfg=planner_cfg,
        root_cfg=cfg,
        seed=seed,
    )
    hyper_scheduler = HyperparameterScheduler(hyper_cfg)

    provider_times = None
    if "time" in dataset.coords:
        provider_times = np.asarray(dataset.coords["time"].values, dtype="datetime64[ns]")
    time_mode = str(truth_cfg.get("time_mode", "interpolate")).lower()
    truth_mode = str(truth_cfg.get("mode", "static")).lower()
    frame_change_every = max(1, int(truth_cfg.get("frame_change_every_samples", 200)))

    # Mission / loop controls.
    batch_size_n = int(planner_cfg.get("batch_size_n", 10))
    max_samples = int(mission_cfg.get("max_samples", 500))
    eval_every_samples = max(1, int(eval_cfg.get("every_n_samples", batch_size_n)))
    sample_interval_s = float(mission_cfg.get("sample_interval_s", 60.0))
    measurement_noise_std = float(mission_cfg.get("measurement_noise_std", 0.0))
    travel_time_budget_s = float(mission_cfg.get("travel_time_budget_s", float("inf")))
    speed_mps_mission = float(mission_cfg.get("speed_mps", 1.0))
    fixed_time_trigger_s = float(replan_cfg.get("fixed_time_seconds", batch_size_n * sample_interval_s))
    uncertainty_threshold = float(replan_cfg.get("uncertainty_threshold", np.inf))
    rho_threshold = float(replan_cfg.get("rho_threshold", hyper_cfg.get("rho0", 0.6)))
    replan_trigger = str(replan_cfg.get("trigger", "end_of_batch")).lower()
    if replan_trigger not in {
        "end_of_batch",
        "fixed_time",
        "uncertainty_threshold",
        "rho_threshold",
        "combined",
    }:
        raise ValueError(
            "replan.trigger must be one of "
            "{end_of_batch,fixed_time,uncertainty_threshold,rho_threshold,combined}."
        )

    # Initial mission state.
    if scenario.time_range is not None:
        current_time = np.datetime64(scenario.time_range[0], "ns")
    elif provider_times is not None and provider_times.size > 0:
        current_time = np.datetime64(provider_times[0], "ns")
    else:
        current_time = np.datetime64("2000-01-01T00:00:00", "ns")

    current_position = np.array(
        [
            float(candidate_points.lats[0]),
            float(candidate_points.lons[0]),
        ],
        dtype=float,
    )
    if "start" in mission_cfg:
        s = mission_cfg["start"]
        current_position = np.array([float(s["lat"]), float(s["lon"])], dtype=float)

    from oceanbench_core.types import ObservationBatch as _ObservationBatch

    # Warm-start with shared prior observations if provided,
    # otherwise bootstrap with nearby candidates.
    if prior_obs is not None and prior_obs.size >= 3:
        model.fit(prior_obs, scenario=scenario)
        init_n = prior_obs.size
    else:
        start_position = current_position.copy()
        init_n = min(max(1, int(mission_cfg.get("initial_samples", 5))), candidate_points.size)
        cand_lats_all = np.asarray(candidate_points.lats, dtype=float)
        cand_lons_all = np.asarray(candidate_points.lons, dtype=float)
        dist_to_start = (cand_lats_all - start_position[0]) ** 2 + (cand_lons_all - start_position[1]) ** 2
        nearest_idx = np.argsort(dist_to_start)[:init_n]
        init_idx = np.sort(nearest_idx)
        init_lats = cand_lats_all[init_idx]
        init_lons = cand_lons_all[init_idx]
        init_times = []
        init_values = []
        for k, (la, lo) in enumerate(zip(init_lats, init_lons)):
            t_query = _resolve_truth_time(
                current_time=current_time + np.timedelta64(int(round((k * sample_interval_s) * 1e9)), "ns"),
                sample_index=k,
                provider_times=provider_times,
                truth_mode=truth_mode,
                frame_change_every_samples=frame_change_every,
                static_reference_time=np.datetime64(truth_cfg.get("static_time", current_time), "ns"),
                time_mode=time_mode,
            )
            qp = QueryPoints(
                lats=np.array([la], dtype=float),
                lons=np.array([lo], dtype=float),
                times=np.array([t_query], dtype="datetime64[ns]"),
            )
            method = "nearest" if time_mode == "snap_to_provider" else "linear"
            y_true = float(np.asarray(truth.query_array(qp, method=method, bounds_mode="clip"), dtype=float).ravel()[0])
            y_obs = y_true + float(rng.normal(0.0, measurement_noise_std))
            init_times.append(t_query)
            init_values.append(y_obs)
        model.fit(
            _ObservationBatch(
                lats=init_lats,
                lons=init_lons,
                values=np.asarray(init_values, dtype=float),
                variable=scenario.variable,
                times=np.asarray(init_times, dtype="datetime64[ns]"),
            ),
            scenario=scenario,
        )

    sample_count = 0  # Prior obs don't count toward mission samples.
    if prior_obs is None and init_n > 0:
        current_position = np.array([init_lats[-1], init_lons[-1]], dtype=float)
        current_time = np.datetime64(init_times[-1], "ns")

    plan_queue: list[np.ndarray] = []
    route_history: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    trajectory_points: list[np.ndarray] = [current_position.copy()]
    planning_time_total = 0.0
    update_time_total = 0.0
    travel_time_used = 0.0
    n_replans = 0
    last_replan_time = current_time
    last_replan_sample = sample_count

    while sample_count < max_samples and travel_time_used < travel_time_budget_s:
        rho = float(getattr(model, "rho_since_last_hyper_update", lambda: 0.0)())
        mean_queue_std = np.inf
        if plan_queue and hasattr(model, "predict"):
            qp_q = QueryPoints(
                lats=np.array([p[0] for p in plan_queue], dtype=float),
                lons=np.array([p[1] for p in plan_queue], dtype=float),
                times=np.full(len(plan_queue), current_time, dtype="datetime64[ns]"),
            )
            pred_q = model.predict(qp_q)
            if pred_q.std is not None:
                mean_queue_std = float(np.nanmean(np.asarray(pred_q.std, dtype=float)))

        need_replan = _should_trigger_replan(
            trigger=replan_trigger,
            queue_empty=(len(plan_queue) == 0),
            elapsed_s=float((current_time - last_replan_time).astype("timedelta64[ns]").astype(np.int64) / 1e9),
            fixed_time_trigger_s=fixed_time_trigger_s,
            mean_queue_std=mean_queue_std,
            uncertainty_threshold=uncertainty_threshold,
            rho=rho,
            rho_threshold=rho_threshold,
        )
        if need_replan:
            t0 = time.perf_counter()

            # Filter candidates to those reachable within remaining budget.
            if travel_time_budget_s < float("inf"):
                budget_remaining_plan = travel_time_budget_s - travel_time_used
                cand_lats = np.asarray(candidate_points.lats, dtype=float)
                cand_lons = np.asarray(candidate_points.lons, dtype=float)
                dists = np.array([
                    _haversine_m(float(current_position[0]), float(current_position[1]),
                                 float(cand_lats[j]), float(cand_lons[j]))
                    for j in range(cand_lats.size)
                ])
                reachable = dists / max(speed_mps_mission, 1e-12) <= budget_remaining_plan
                if not reachable.any():
                    break  # No candidates reachable within budget.
                reachable_cands = QueryPoints(
                    lats=cand_lats[reachable], lons=cand_lons[reachable],
                )
            else:
                reachable_cands = candidate_points

            plan_res = planner.plan(model, reachable_cands, eval_points=eval_spatial)
            selected_xy = np.column_stack(
                [
                    np.asarray(plan_res.selected_points.lats, dtype=float),
                    np.asarray(plan_res.selected_points.lons, dtype=float),
                ]
            )
            if selected_xy.shape[0] == 0:
                break

            route_mode = str(routing_cfg.get("mode", "open_end_anywhere")).lower()
            route_backend = str(routing_cfg.get("backend", "ortools")).lower()
            route_metric = str(routing_cfg.get("metric", "haversine")).lower()
            speed_mps = float(mission_cfg.get("speed_mps", 1.0))

            # Route starts at current robot position and visits planned points.
            tsp_points = np.vstack([current_position.reshape(1, 2), selected_xy])
            end_idx = None
            if route_mode == "open_fixed_end":
                if "fixed_end" in routing_cfg:
                    fe = routing_cfg["fixed_end"]
                    tsp_points = np.vstack(
                        [
                            tsp_points,
                            np.array([[float(fe["lat"]), float(fe["lon"])]], dtype=float),
                        ]
                    )
                    end_idx = int(tsp_points.shape[0] - 1)
                else:
                    end_idx = int(tsp_points.shape[0] - 1)

            route = solve_tsp_route(
                tsp_points,
                start_index=0,
                end_index=end_idx,
                backend=route_backend,
                mode=route_mode,
                metric=route_metric,
                speed_mps=speed_mps,
            )
            planning_time_total += float(time.perf_counter() - t0)
            n_replans += 1
            last_replan_time = current_time
            last_replan_sample = sample_count

            # Convert route to execution queue; skip first element (current position).
            route_pts = np.asarray(route.points, dtype=float)
            if route_pts.shape[0] > 1:
                queue = [route_pts[i].copy() for i in range(1, route_pts.shape[0])]
            else:
                queue = []
            if route_mode == "closed" and len(queue) > 0:
                # Closed route returns to start; avoid immediately resampling same start.
                if np.allclose(queue[-1], current_position):
                    queue = queue[:-1]
            plan_queue = queue

            # Trim queue to fit within remaining travel-time budget.
            if travel_time_budget_s < float("inf") and plan_queue:
                trimmed: list[np.ndarray] = []
                pos = current_position.copy()
                budget_remaining = travel_time_budget_s - travel_time_used
                for wp in plan_queue:
                    d = _haversine_m(float(pos[0]), float(pos[1]), float(wp[0]), float(wp[1]))
                    dt = d / max(speed_mps_mission, 1e-12)
                    if budget_remaining - dt < 0:
                        break
                    budget_remaining -= dt
                    trimmed.append(wp)
                    pos = wp
                plan_queue = trimmed

            route_history.append(
                {
                    "sample_count": int(sample_count),
                    "selected_indices": plan_res.selected_indices.tolist(),
                    "selected_points": selected_xy.tolist(),
                    "route_indices": route.indices.tolist(),
                    "route_points": route_pts.tolist(),
                    "route_total_cost": float(route.total_cost),
                    "planner_objective": float(plan_res.objective_value),
                    "planner_marginal_gains": np.asarray(plan_res.marginal_gains, dtype=float).tolist(),
                    "planner_debug": dict(plan_res.debug),
                    "routing_backend": route.backend,
                    "routing_mode": route.mode,
                    "routing_metric": route.metric,
                }
            )
            if run_dir is not None:
                append_log(
                    run_dir,
                    {
                        "event": "replan",
                        "sample_count": int(sample_count),
                        "n_replans": int(n_replans),
                        "route_points": route_pts.tolist(),
                    },
                )

        if not plan_queue:
            break

        # Execute one sample waypoint.
        target = np.asarray(plan_queue.pop(0), dtype=float)

        # Compute actual travel time for this leg.
        leg_dist = _haversine_m(
            float(current_position[0]), float(current_position[1]),
            float(target[0]), float(target[1]),
        )
        leg_time = leg_dist / max(speed_mps_mission, 1e-12)

        # Check travel-time budget before moving.
        if travel_time_budget_s < float("inf"):
            if travel_time_used + leg_time > travel_time_budget_s:
                break
            travel_time_used += leg_time

        current_time = np.datetime64(
            current_time + np.timedelta64(int(round(leg_time * 1e9)), "ns"),
            "ns",
        )
        t_query = _resolve_truth_time(
            current_time=current_time,
            sample_index=sample_count,
            provider_times=provider_times,
            truth_mode=truth_mode,
            frame_change_every_samples=frame_change_every,
            static_reference_time=np.datetime64(truth_cfg.get("static_time", current_time), "ns"),
            time_mode=time_mode,
        )
        qp_target = QueryPoints(
            lats=np.array([float(target[0])], dtype=float),
            lons=np.array([float(target[1])], dtype=float),
            times=np.array([t_query], dtype="datetime64[ns]"),
        )
        interp_method = "nearest" if time_mode == "snap_to_provider" else "linear"
        y_true = float(np.asarray(truth.query_array(qp_target, method=interp_method, bounds_mode="clip"), dtype=float).ravel()[0])
        y_obs = y_true + float(rng.normal(0.0, measurement_noise_std))

        obs = _ObservationBatch(
            lats=np.array([target[0]], dtype=float),
            lons=np.array([target[1]], dtype=float),
            values=np.array([y_obs], dtype=float),
            variable=scenario.variable,
            times=np.array([t_query], dtype="datetime64[ns]"),
        )
        t0 = time.perf_counter()
        model.update(obs)
        update_time_total += float(time.perf_counter() - t0)

        sample_count += 1
        current_position = target
        trajectory_points.append(target.copy())

        # Hyperparameter schedule update.
        rho = float(getattr(model, "rho_since_last_hyper_update", lambda: 0.0)())
        if hyper_scheduler.should_update(step=sample_count, rho=rho):
            if hasattr(model, "fit_hyperparameters"):
                _ = model.fit_hyperparameters()
            if hasattr(model, "train_replan"):
                model.train_replan()
            hyper_scheduler.mark_updated(step=sample_count)
            if hyper_scheduler.on_update == "replan_immediately":
                plan_queue = []

        # Evaluate on the fixed dense spatial grid at selected times.
        budget_exhausted = travel_time_used >= travel_time_budget_s
        if (sample_count % eval_every_samples == 0) or (sample_count >= max_samples) or budget_exhausted:
            eval_times = resolve_eval_times(
                eval_cfg,
                scenario=scenario,
                provider_times=provider_times,
                current_time=current_time,
            )
            route_points_arr = np.asarray(trajectory_points, dtype=float)
            metrics = score_persistent_sampling(
                model=model,
                truth_field=truth,
                eval_spatial_points=eval_spatial,
                eval_times=eval_times,
                time_mode=time_mode,
                available_provider_times=provider_times,
                route_points=route_points_arr,
                route_metric=str(routing_cfg.get("metric", "haversine")),
                speed_mps=float(mission_cfg.get("speed_mps", 1.0)),
                planning_time_s=planning_time_total,
                update_time_s=update_time_total,
                n_replans=n_replans,
                bounds_mode="clip",
                return_maps=bool(eval_cfg.get("store_maps", False)),
            )
            metrics["sample_count"] = int(sample_count)
            metric_history.append(metrics)
            if run_dir is not None:
                append_log(
                    run_dir,
                    {
                        "event": "metrics",
                        "sample_count": int(sample_count),
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "replans": metrics.get("replans"),
                    },
                )

        if sample_count >= max_samples or travel_time_used >= travel_time_budget_s:
            break

    final_metrics = metric_history[-1] if metric_history else {}
    route_points_arr = np.asarray(trajectory_points, dtype=float)
    total_distance = 0.0
    if route_points_arr.shape[0] >= 2:
        total_distance = route_distance(route_points_arr, metric=str(routing_cfg.get("metric", "haversine")))

    # Final predictions on the eval grid for plotting.
    pred_mean_arr, pred_std_arr = None, None
    if hasattr(model, "predict") and eval_spatial is not None:
        try:
            final_pred = model.predict(eval_spatial)
            pred_mean_arr = np.asarray(final_pred.mean, dtype=float).ravel()
            if final_pred.std is not None:
                pred_std_arr = np.asarray(final_pred.std, dtype=float).ravel()
        except Exception:
            pass

    result: dict[str, Any] = {
        "scenario": {
            "name": scenario.name,
            "variable": scenario.variable,
            "region": dict(scenario.region),
        },
        "model_backend": str(field_cfg.get("backend", "sogp_paper")).lower(),
        "planner_type": str(planner_cfg.get("type", "mi_dp")).lower(),
        "routing_backend": str(routing_cfg.get("backend", "ortools")).lower(),
        "routing_mode": str(routing_cfg.get("mode", "open_end_anywhere")).lower(),
        "truth_mode": truth_mode,
        "time_mode": time_mode,
        "n_samples": int(sample_count),
        "n_replans": int(n_replans),
        "planning_time": float(planning_time_total),
        "update_time": float(update_time_total),
        "total_runtime": float(planning_time_total + update_time_total),
        "distance": float(total_distance),
        "travel_time_used": float(travel_time_used),
        "route_history": route_history,
        "metrics_history": metric_history,
        "final_metrics": final_metrics,
        "trajectory_points": route_points_arr.tolist(),
        "pred_mean": pred_mean_arr,
        "pred_std": pred_std_arr,
    }

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(run_dir, cfg)
        save_json(
            run_dir,
            "persistent_sampling_result.json",
            {
                "n_samples": result["n_samples"],
                "n_replans": result["n_replans"],
                "final_metrics": result["final_metrics"],
                "distance": result["distance"],
            },
        )
        save_json(run_dir, "route_history.json", {"routes": route_history})
        save_json(run_dir, "metrics_history.json", {"history": metric_history})

    logger.debug(
        "Persistent sampling completed: samples=%s replans=%s rmse=%s",
        result["n_samples"],
        result["n_replans"],
        result["final_metrics"].get("rmse"),
    )
    return result


def _scenario_from_config(scenario_cfg: Mapping[str, Any]) -> Scenario:
    region_cfg = dict(scenario_cfg.get("region", scenario_cfg.get("bbox", {})))
    if "lat_min" not in region_cfg:
        region_cfg = {
            "lat_min": float(region_cfg["lat"][0]),
            "lat_max": float(region_cfg["lat"][1]),
            "lon_min": float(region_cfg["lon"][0]),
            "lon_max": float(region_cfg["lon"][1]),
        }
    time_window = scenario_cfg.get("time_window", scenario_cfg.get("time_range"))
    time_range = None
    if time_window is not None:
        time_range = (
            np.datetime64(time_window[0], "ns"),
            np.datetime64(time_window[1], "ns"),
        )
    return Scenario(
        name=scenario_cfg.get("name", "dlp_region"),
        variable=str(scenario_cfg.get("variable", "temp")),
        region={
            "lat_min": float(region_cfg["lat_min"]),
            "lat_max": float(region_cfg["lat_max"]),
            "lon_min": float(region_cfg["lon_min"]),
            "lon_max": float(region_cfg["lon_max"]),
        },
        time_range=time_range,
        depth_range=None if scenario_cfg.get("depth") is None else (float(scenario_cfg["depth"]), float(scenario_cfg["depth"])),
        metadata={"product_id": scenario_cfg.get("product_id", "")},
    )


def _provider_subset_for_scenario(
    provider: DataProvider,
    *,
    scenario_cfg: Mapping[str, Any],
    scenario: Scenario,
) -> Any:
    product_id = str(scenario_cfg.get("product_id", scenario.metadata.get("product_id", "")))
    if not product_id:
        raise ValueError("scenario.product_id is required.")
    if scenario.time_range is None:
        raise ValueError("scenario.time_window/time_range is required.")
    region = {
        "lat": [float(scenario.region["lat_min"]), float(scenario.region["lat_max"])],
        "lon": [float(scenario.region["lon_min"]), float(scenario.region["lon_max"])],
    }
    time_range = (
        str(np.datetime64(scenario.time_range[0], "s")),
        str(np.datetime64(scenario.time_range[1], "s")),
    )
    depth = scenario_cfg.get("depth")
    depth_opts = None
    if depth is not None:
        depth_opts = {"min": float(depth), "max": float(depth)}
    ds = provider.subset(
        product_id=product_id,
        region=region,
        time=time_range,
        variables=[scenario.variable],
        depth_opts=depth_opts,
        target_grid=None,
        overwrite=bool(scenario_cfg.get("overwrite", False)),
    )
    return ds


def _build_field_model_for_persistent(
    *,
    field_model_cfg: Mapping[str, Any],
    root_cfg: Mapping[str, Any],
    seed: int,
) -> Any:
    backend = str(field_model_cfg.get("backend", "sogp_paper")).lower()
    model_cfg = dict(field_model_cfg.get("params", field_model_cfg))
    if "noise" not in model_cfg and "noise" in root_cfg:
        model_cfg["noise"] = dict(root_cfg.get("noise", {}))
    if "hyperparams" not in model_cfg and "hyperparams" in root_cfg:
        model_cfg["hyperparams"] = dict(root_cfg.get("hyperparams", {}))
    if backend == "sogp_paper":
        from oceanbench_models.belief.field import SOGPPaperFieldModel

        return SOGPPaperFieldModel(model_cfg, seed=seed)
    if backend == "svgp_gpytorch":
        from oceanbench_models.belief.field import SVGPGPyTorchFieldModel

        return SVGPGPyTorchFieldModel(model_cfg, seed=seed)
    raise ValueError("field_model.backend must be one of {'sogp_paper','svgp_gpytorch'}.")


def _build_mi_planner(
    *,
    planner_cfg: Mapping[str, Any],
    root_cfg: Mapping[str, Any],
    seed: int,
) -> Any:
    planner_type = str(planner_cfg.get("type", "mi_dp")).lower()
    merged_cfg = {
        "planner": dict(planner_cfg),
        "mi": dict(root_cfg.get("mi", {})),
    }
    if planner_type == "mi_dp":
        from oceanbench_policies.ipp import MIDPPlanner

        return MIDPPlanner(merged_cfg, seed=seed)
    if planner_type == "mi_greedy":
        from oceanbench_policies.ipp import MIGreedyPlanner

        return MIGreedyPlanner(merged_cfg, seed=seed)
    raise ValueError("planner.type must be one of {'mi_dp','mi_greedy'}.")


def _resolve_truth_time(
    *,
    current_time: np.datetime64,
    sample_index: int,
    provider_times: Optional[np.ndarray],
    truth_mode: str,
    frame_change_every_samples: int,
    static_reference_time: np.datetime64,
    time_mode: str,
) -> np.datetime64:
    t = np.datetime64(current_time, "ns")
    mode = str(truth_mode).lower()
    if mode == "static":
        t = np.datetime64(static_reference_time, "ns")
    elif mode == "dynamic_provider":
        t = np.datetime64(current_time, "ns")
    elif mode == "dynamic_piecewise":
        if provider_times is None or provider_times.size == 0:
            t = np.datetime64(current_time, "ns")
        else:
            frame_idx = int(sample_index // max(1, frame_change_every_samples))
            frame_idx = min(frame_idx, int(provider_times.size - 1))
            t = np.datetime64(provider_times[frame_idx], "ns")
    else:
        raise ValueError("truth.mode must be one of {'static','dynamic_provider','dynamic_piecewise'}.")

    if str(time_mode).lower() == "snap_to_provider" and provider_times is not None and provider_times.size > 0:
        arr = np.asarray(provider_times, dtype="datetime64[ns]")
        t_int = t.astype("int64")
        idx = int(np.argmin(np.abs(arr.astype("int64") - t_int)))
        t = np.datetime64(arr[idx], "ns")
    return t


def _should_trigger_replan(
    *,
    trigger: str,
    queue_empty: bool,
    elapsed_s: float,
    fixed_time_trigger_s: float,
    mean_queue_std: float,
    uncertainty_threshold: float,
    rho: float,
    rho_threshold: float,
) -> bool:
    trigger = str(trigger).lower()
    if trigger == "end_of_batch":
        return bool(queue_empty)
    if trigger == "fixed_time":
        return bool(queue_empty or elapsed_s >= fixed_time_trigger_s)
    if trigger == "uncertainty_threshold":
        return bool(queue_empty or mean_queue_std >= uncertainty_threshold)
    if trigger == "rho_threshold":
        return bool(queue_empty or rho >= rho_threshold)
    if trigger == "combined":
        return bool(
            queue_empty
            or elapsed_s >= fixed_time_trigger_s
            or mean_queue_std >= uncertainty_threshold
            or rho >= rho_threshold
        )
    raise ValueError(
        "replan.trigger must be one of "
        "{end_of_batch,fixed_time,uncertainty_threshold,rho_threshold,combined}."
    )

