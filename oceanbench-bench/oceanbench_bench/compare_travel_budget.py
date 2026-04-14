"""Travel-time budget comparison harness for 4 IPP methods.

Runs Binney (GRG), DLP (persistent sampling), POMCP, and AdaPP on the
same scenario across multiple travel-time budgets and seeds, then
produces RMSE-vs-T plots and trajectory overlays.

This is a first-pass sanity harness -- not a full profiling suite.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oceanbench_core import WaypointGraph
from oceanbench_core.eval_grid import build_eval_grid
from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario
from oceanbench_data_provider import DataProvider
from oceanbench_env import OceanTruthField

logger = logging.getLogger(__name__)

# ── Method name constants ────────────────────────────────────────────────
BINNEY = "binney_grg"
DLP = "dlp_persistent"
POMCP = "pomcp"
ADAPP = "adapp"
ALL_METHODS = [BINNEY, DLP, POMCP, ADAPP]

# ── Default scenario config ─────────────────────────────────────────────


def default_scenario_config() -> dict[str, Any]:
    """Return inline defaults for the Gulf of Mexico comparison scenario."""
    return {
        "product_id": "copernicus_phy_reanalysis_001_030",
        "variable": "temp",
        "region": {"lon": [-90, -88], "lat": [24, 26]},
        "time_window": ("2014-01-01", "2014-01-07"),
        "static_time": "2014-01-03T00:00:00",
        "depth": None,
        # Graph / shared infra
        "graph_n_lat": 20,
        "graph_n_lon": 20,
        "speed_mps": 1.0,
        "connectivity": "4",
        # Eval grid
        "eval_n_lat": 40,
        "eval_n_lon": 40,
        "eval_max_points": 10_000,
        # Shared prior observations (warm-start all methods with same data)
        "n_prior_samples": 30,
        "prior_sampling": "lhs",  # "random", "lhs", or "grid"
        # DLP-specific
        "sample_interval_s": 60.0,
        # Sweep — T values in seconds.
        # 2x2 deg region: graph edge ≈ 11 km ≈ 11,000 s at 1 m/s.
        "T_list": [1_000_000],
        "seeds": [0],
        "T_plot": 1_000_000,
    }


# ── Region dict helpers ─────────────────────────────────────────────────


def _region_minmax(cfg: Mapping[str, Any]) -> dict[str, float]:
    """Convert region {lon: [a,b], lat: [c,d]} to lat_min/max lon_min/max."""
    r = cfg["region"]
    return {
        "lat_min": float(r["lat"][0]),
        "lat_max": float(r["lat"][1]),
        "lon_min": float(r["lon"][0]),
        "lon_max": float(r["lon"][1]),
    }


def _region_provider(cfg: Mapping[str, Any]) -> dict[str, list[float]]:
    """Convert to DataProvider format {lat: [min,max], lon: [min,max]}."""
    r = cfg["region"]
    return {
        "lat": [float(r["lat"][0]), float(r["lat"][1])],
        "lon": [float(r["lon"][0]), float(r["lon"][1])],
    }


# ── Shared infrastructure (built once, reused for POMCP & AdaPP) ────────


def setup_shared_infrastructure(
    scenario_cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Build graph, truth field, eval grid, and start position once.

    Returns a dict with keys: graph, truth_field, eval_qp, truth_values,
    region_minmax, scenario, start_node, start_lat, start_lon.
    """
    rng = np.random.default_rng(seed)
    region_mm = _region_minmax(scenario_cfg)
    variable = str(scenario_cfg.get("variable", "temp"))
    time_window = scenario_cfg["time_window"]
    static_time = scenario_cfg.get("static_time", time_window[0])

    scenario = Scenario(
        name="compare_travel_budget",
        variable=variable,
        region=region_mm,
        time_range=(
            np.datetime64(time_window[0], "ns"),
            np.datetime64(time_window[1], "ns"),
        ),
        metadata={"product_id": scenario_cfg.get("product_id", "")},
    )

    # Data provider subset
    provider = DataProvider()
    ds = provider.subset(
        product_id=scenario_cfg.get("product_id", ""),
        region=_region_provider(scenario_cfg),
        time=(str(time_window[0]), str(time_window[1])),
        variables=[variable],
        depth_opts=None,
        target_grid=None,
        overwrite=False,
    )
    truth_field = OceanTruthField(dataset=ds, variable=variable, scenario=scenario)

    # Graph — inset nodes by half a cell so no node sits on the boundary.
    n_lat = int(scenario_cfg.get("graph_n_lat", 20))
    n_lon = int(scenario_cfg.get("graph_n_lon", 20))
    step_lat = (region_mm["lat_max"] - region_mm["lat_min"]) / n_lat
    step_lon = (region_mm["lon_max"] - region_mm["lon_min"]) / n_lon
    graph_region = {
        "lat_min": region_mm["lat_min"] + step_lat / 2,
        "lat_max": region_mm["lat_max"] - step_lat / 2,
        "lon_min": region_mm["lon_min"] + step_lon / 2,
        "lon_max": region_mm["lon_max"] - step_lon / 2,
    }
    graph = WaypointGraph.grid(
        graph_region,
        n_lat, n_lon,
        speed_mps=float(scenario_cfg.get("speed_mps", 1.0)),
        connectivity=str(scenario_cfg.get("connectivity", "4")),
    )

    # Start from the node closest to the region center.
    center_lat = 0.5 * (region_mm["lat_min"] + region_mm["lat_max"])
    center_lon = 0.5 * (region_mm["lon_min"] + region_mm["lon_max"])
    best_node, best_dist = 0, float("inf")
    for nid in range(graph.n_nodes):
        nlat, nlon = graph.node_coords(nid)
        d = (nlat - center_lat) ** 2 + (nlon - center_lon) ** 2
        if d < best_dist:
            best_dist, best_node = d, nid
    start_node = best_node
    start_lat, start_lon = graph.node_coords(start_node)

    # Eval grid (built once, reused)
    eval_cfg = {
        "grid_n_lat": int(scenario_cfg.get("eval_n_lat", 40)),
        "grid_n_lon": int(scenario_cfg.get("eval_n_lon", 40)),
        "max_points": int(scenario_cfg.get("eval_max_points", 10_000)),
        "eval_time": static_time,
    }
    eval_grid = build_eval_grid(scenario, eval_cfg, rng=rng)
    eval_qp = eval_grid.query_points

    truth_values_raw = truth_field.query_array(eval_qp, method="linear", bounds_mode="nan")
    truth_values_raw = np.asarray(truth_values_raw, dtype=float)

    if truth_values_raw.ndim == 2 and truth_values_raw.shape[0] == eval_qp.size:
        truth_values = truth_values_raw[:, 0]  # surface
    else:
        truth_values = truth_values_raw.reshape(-1)
        if truth_values.size != eval_qp.size:
            raise ValueError(f"Truth size {truth_values.size} != eval_qp.size {eval_qp.size}")

    # Generate shared prior observations for warm-starting all methods.
    n_prior = int(scenario_cfg.get("n_prior_samples", 30))
    prior_strategy = str(scenario_cfg.get("prior_sampling", "lhs")).lower()
    lat_lo, lat_hi = float(region_mm["lat_min"]), float(region_mm["lat_max"])
    lon_lo, lon_hi = float(region_mm["lon_min"]), float(region_mm["lon_max"])

    if prior_strategy == "lhs":
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=2, seed=seed + 1000)
        samples_01 = sampler.random(n=n_prior)
        prior_lats = lat_lo + samples_01[:, 0] * (lat_hi - lat_lo)
        prior_lons = lon_lo + samples_01[:, 1] * (lon_hi - lon_lo)
    elif prior_strategy == "grid":
        n_side = max(2, int(np.ceil(np.sqrt(n_prior))))
        gl = np.linspace(lat_lo, lat_hi, n_side, endpoint=False) + 0.5 * (lat_hi - lat_lo) / n_side
        gn = np.linspace(lon_lo, lon_hi, n_side, endpoint=False) + 0.5 * (lon_hi - lon_lo) / n_side
        lat_m, lon_m = np.meshgrid(gl, gn, indexing="ij")
        prior_lats = lat_m.ravel()[:n_prior]
        prior_lons = lon_m.ravel()[:n_prior]
    else:  # "random"
        prior_rng = np.random.default_rng(seed + 1000)
        prior_lats = prior_rng.uniform(lat_lo, lat_hi, n_prior)
        prior_lons = prior_rng.uniform(lon_lo, lon_hi, n_prior)

    prior_qp = QueryPoints(lats=prior_lats, lons=prior_lons)
    prior_vals_raw = truth_field.query_array(prior_qp, method="linear", bounds_mode="nan")
    prior_vals_flat = np.asarray(prior_vals_raw, dtype=float).ravel()
    finite = np.isfinite(prior_vals_flat[:len(prior_lats)])
    prior_obs = ObservationBatch(
        lats=prior_lats[finite],
        lons=prior_lons[finite],
        values=prior_vals_flat[:len(prior_lats)][finite],
        variable=variable,
    )

    return {
        "graph": graph,
        "truth_field": truth_field,
        "eval_qp": eval_qp,
        "truth_values": truth_values,
        "region_minmax": region_mm,
        "scenario": scenario,
        "start_node": start_node,
        "start_lat": start_lat,
        "start_lon": start_lon,
        "provider": provider,
        "prior_obs": prior_obs,
    }


# ── Config builders for existing runners ────────────────────────────────


def build_binney_config(
    scenario_cfg: Mapping[str, Any],
    T: float,
    seed: int,
) -> dict[str, Any]:
    """Build the nested config dict for run_binney_receding_horizon."""
    region_mm = _region_minmax(scenario_cfg)
    time_window = scenario_cfg["time_window"]
    static_time = scenario_cfg.get("static_time", time_window[0])
    speed = float(scenario_cfg.get("speed_mps", 1.0))
    n_lat = int(scenario_cfg.get("graph_n_lat", 20))
    n_lon = int(scenario_cfg.get("graph_n_lon", 20))

    # Inset graph region by half a cell (consistent with shared infra graph).
    step_lat = (region_mm["lat_max"] - region_mm["lat_min"]) / n_lat
    step_lon = (region_mm["lon_max"] - region_mm["lon_min"]) / n_lon
    graph_region = {
        "lat_min": region_mm["lat_min"] + step_lat / 2,
        "lat_max": region_mm["lat_max"] - step_lat / 2,
        "lon_min": region_mm["lon_min"] + step_lon / 2,
        "lon_max": region_mm["lon_max"] - step_lon / 2,
    }

    # Receding horizon: replan ~4 times, incorporating new observations.
    horizon = T / 4
    execute = T / 4

    return {
        "seed": seed,
        "scenario": {
            "name": "compare_travel_budget",
            "variable": scenario_cfg.get("variable", "temp"),
            "region": graph_region,
            "time_range": (
                np.datetime64(time_window[0], "ns"),
                np.datetime64(time_window[1], "ns"),
            ),
            "metadata": {
                "product_id": scenario_cfg.get("product_id", ""),
            },
        },
        "task": {
            "graph": {
                "type": "grid",
                "seed": seed,
                "grid": {"n_lat": n_lat, "n_lon": n_lon},
                "start": {"node_id": 0},
                "goal": {"node_id": -1},  # dynamic: pick per horizon window
            },
            "robot": {"speed_mps": speed},
            "sampling": {
                "mode": "nodes+edges",
                "edge_spacing_m": 5000.0,
                "include_nodes": True,
                "measurement_noise_var": 0.01,
            },
            "time": {
                "dynamic": False,
                "tau0": str(static_time),
                "snap_to_provider_grid": False,
                "interp": "linear",
            },
            "objective": {
                "type": "emse",
                "eval_set": "dense_grid",
            },
            "eval": {
                "grid_n_lat": int(scenario_cfg.get("eval_n_lat", 40)),
                "grid_n_lon": int(scenario_cfg.get("eval_n_lon", 40)),
                "max_points": int(scenario_cfg.get("eval_max_points", 10_000)),
                "subsample_strategy": "random",
                "metrics": ["rmse", "mae"],
            },
            "covariance": {
                "backend": "kernel",
                "kernel": {
                    "kernel_type": "matern",
                    "nu": 2.5,
                    "lengthscale_space": 0.5,
                    "lengthscale_time": 86400.0,
                    "variance": 1.0,
                    "noise": 0.1,
                    "project_to_meters": False,
                    "use_scaling": False,
                    "fit": True,
                    "fit_max_iters": 10,
                    "fit_subsample_n": 200,
                },
            },
        },
        "policy": {
            "type": "grg",
            "planning": {
                "budget_seconds": T,
                "horizon_seconds": horizon,
                "execute_seconds": execute,
                "depth": 2,
                "n_splits": 5,
                "split_strategy": "uniform",
            },
        },
    }


def build_persistent_config(
    scenario_cfg: Mapping[str, Any],
    T: float,
    seed: int,
) -> dict[str, Any]:
    """Build the nested config dict for run_persistent_sampling."""
    region_mm = _region_minmax(scenario_cfg)
    time_window = scenario_cfg["time_window"]
    static_time = scenario_cfg.get("static_time", time_window[0])
    variable = scenario_cfg.get("variable", "temp")
    speed = float(scenario_cfg.get("speed_mps", 1.0))
    sample_interval_s = float(scenario_cfg.get("sample_interval_s", 60.0))
    max_samples = 500  # safety cap; travel_time_budget_s is the real constraint
    start_lat = 0.5 * (float(region_mm["lat_min"]) + float(region_mm["lat_max"]))
    start_lon = 0.5 * (float(region_mm["lon_min"]) + float(region_mm["lon_max"]))

    # Inset candidate grid region by half a cell (consistent with graph inset).
    cand_n_lat, cand_n_lon = 20, 20
    step_lat = (region_mm["lat_max"] - region_mm["lat_min"]) / cand_n_lat
    step_lon = (region_mm["lon_max"] - region_mm["lon_min"]) / cand_n_lon
    cand_region = {
        "lat_min": region_mm["lat_min"] + step_lat / 2,
        "lat_max": region_mm["lat_max"] - step_lat / 2,
        "lon_min": region_mm["lon_min"] + step_lon / 2,
        "lon_max": region_mm["lon_max"] - step_lon / 2,
    }

    return {
        "seed": seed,
        "scenario": {
            "name": "compare_travel_budget",
            "product_id": scenario_cfg.get("product_id", ""),
            "variable": variable,
            "region": cand_region,
            "time_window": list(time_window),
            "depth": scenario_cfg.get("depth"),
        },
        "candidates": {
            "type": "grid",
            "grid": {"n_lat": 20, "n_lon": 20},
            "max_points": 500,
            "seed": seed,
            "subsample_strategy": "random",
        },
        "eval": {
            "grid": {
                "fixed": True,
                "n_lat": int(scenario_cfg.get("eval_n_lat", 40)),
                "n_lon": int(scenario_cfg.get("eval_n_lon", 40)),
            },
            "max_points": int(scenario_cfg.get("eval_max_points", 10_000)),
            "subsample_strategy": "random",
            "times": "single",
            "sequence_length": 1,
            "error_time": "fixed_reference",
            "every_n_samples": max_samples,  # eval at end (budget or cap)
            "store_maps": False,
            "metrics": ["rmse", "mae"],
        },
        "planner": {
            "type": "mi_dp",
            "batch_size_n": 30,
            "dp_beam_width": 5,
            "max_candidates_for_dp": 50,
        },
        "mi": {
            "X_set": "candidate_grid",
            "X_subsample": {"max_points": 200},
            "jitter": 1e-6,
        },
        "routing": {
            "mode": "open_end_anywhere",
            "backend": "ortools",
            "metric": "haversine",
        },
        "mission": {
            "initial_samples": 5,
            "max_samples": max_samples,
            "travel_time_budget_s": T,
            "sample_interval_s": sample_interval_s,
            "speed_mps": speed,
            "measurement_noise_std": 0.1,
            "start": {"lat": start_lat, "lon": start_lon},
        },
        "field_model": {
            "backend": "sogp_paper",
            "params": {
                "max_basis_size": 100,
                "novelty_threshold": 0.1,
                "jitter": 1e-4,
                "include_time": False,
                "include_depth": False,
                "use_scaling": True,
                "lengthscale": 0.5,
                "variance": 2.0,
                "hyper_fit_method": "loo_cv",
                "hyper_fit_max_iters": 30,
                "max_history": 500,
            },
        },
        "truth": {
            "mode": "static",
            "time_mode": "interpolate",
            "frame_change_every_samples": 200,
            "static_time": str(static_time),
        },
        "hyperparams": {
            "mode": "rho_trigger",
            "rho0": 0.6,
            "period": 50,
            "on_update": "replan_immediately",
        },
        "replan": {
            "trigger": "end_of_batch",
        },
    }


# ── POMCP episode runner (no existing end-to-end runner) ────────────────


def run_pomcp_episode(
    infra: Mapping[str, Any],
    T: float,
    seed: int,
    variable: str = "temp",
) -> dict[str, Any]:
    """Run a single POMCP episode and return rmse + trajectory."""
    from oceanbench_models.belief.field import GPFieldModel
    from oceanbench_policies.pomdp.pomcp_adapter import POMCPConfig, POMCPPolicy
    from oceanbench_policies.pomdp.state_models import (
        BeliefAdapter,
        POMDPObservation,
        POMDPState,
    )

    graph: WaypointGraph = infra["graph"]
    truth_field: OceanTruthField = infra["truth_field"]
    eval_qp: QueryPoints = infra["eval_qp"]
    truth_values: np.ndarray = infra["truth_values"]
    scenario: Scenario = infra["scenario"]
    start_node: int = infra["start_node"]
    start_lat: float = infra["start_lat"]
    start_lon: float = infra["start_lon"]

    rng = np.random.default_rng(seed)

    # Build field model (exact GP)
    field_model = GPFieldModel(
        config={"kernel_type": "matern", "nu": 2.5,
                "lengthscale": 0.5, "variance": 2.0, "noise": 0.01,
                "include_time": False, "include_depth": False,
                "use_scaling": False, "training_iters": 20},
        seed=seed,
    )

    # Warm-start with shared prior observations.
    prior_obs: ObservationBatch = infra["prior_obs"]
    field_model.fit(prior_obs, scenario=scenario)

    # Belief adapter
    belief = BeliefAdapter(
        field_model, variable,
        objective_c=1.0, measurement_noise_var=0.01,
    )
    belief.seed_observations(prior_obs)

    # POMCP policy
    pomcp_cfg = POMCPConfig(
        max_depth=10,
        discount=0.95,
        uct_c=1.0,
        action_space="graph_neighbors",
        include_stay=False,
        objective_c=1.0,
        rollout_schedule="constant",
        rollout_kwargs={"n": 200},
        max_steps=200,
        seed=seed,
    )
    policy = POMCPPolicy(graph=graph, belief=belief, config=pomcp_cfg)

    # Static reference time
    static_time = scenario.time_range[0] if scenario.time_range else np.datetime64("2014-01-03", "ns")
    ref_time = np.datetime64(static_time, "ns")

    # Episode loop
    current_node = start_node
    travel_time = 0.0
    trajectory: list[dict[str, Any]] = [
        {"lat": start_lat, "lon": start_lon, "node_id": start_node, "time": 0.0},
    ]
    step = 0

    while travel_time < T:
        state = POMDPState(
            node_id=current_node,
            time=ref_time,
            step=step,
            remaining_budget=T - travel_time,
        )
        action = policy.act(state)
        target_node = action.target_node_id

        if target_node == current_node:
            break

        # Travel
        edge = graph.edge_attributes(current_node, target_node)
        dt = float(edge["time_s"])
        if travel_time + dt > T:
            break
        travel_time += dt
        current_node = target_node
        step += 1

        # Observe truth
        lat, lon = graph.node_coords(current_node)
        obs_qp = QueryPoints(
            lats=np.array([lat]), lons=np.array([lon]),
        )
        obs_val = float(
            np.asarray(
                truth_field.query_array(obs_qp, method="linear", bounds_mode="nan"),
                dtype=float,
            ).ravel()[0]
        )
        obs_val += float(rng.normal(0.0, np.sqrt(0.01)))  # measurement noise

        # Update belief
        pomcp_obs = POMDPObservation(
            value=obs_val, noise_var=0.01, lat=lat, lon=lon,
        )
        policy.observe(pomcp_obs)

        trajectory.append({
            "lat": lat, "lon": lon, "node_id": current_node, "time": travel_time,
        })

    # Final evaluation
    pred = field_model.predict(eval_qp)
    pred_mean = np.asarray(pred.mean, dtype=float).ravel()
    pred_std = np.asarray(pred.std, dtype=float).ravel() if pred.std is not None else np.full_like(pred_mean, np.nan)
    error = truth_values - pred_mean
    finite = np.isfinite(error)
    rmse = float(np.sqrt(np.mean(error[finite] ** 2))) if finite.any() else np.nan

    return {
        "rmse": rmse,
        "trajectory": trajectory,
        "n_samples": step,
        "travel_time": travel_time,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


# ── Dispatch: run one method for one (T, seed) ─────────────────────────


def run_one_method(
    method: str,
    T: float,
    seed: int,
    scenario_cfg: Mapping[str, Any],
    run_dir: Optional[Path] = None,
    shared_infra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Run a single method and return a result dict.

    Keys: method, T, seed, rmse, n_samples, distance, runtime, trajectory.
    On failure: rmse=NaN, error logged.
    """
    result: dict[str, Any] = {
        "method": method,
        "T": T,
        "seed": seed,
        "rmse": np.nan,
        "n_samples": 0,
        "distance": np.nan,
        "runtime": np.nan,
        "trajectory": [],
    }
    variable = str(scenario_cfg.get("variable", "temp"))

    try:
        t0 = time.perf_counter()

        if method == BINNEY:
            from .runner import run_binney_receding_horizon

            cfg = build_binney_config(scenario_cfg, T, seed)
            if shared_infra is not None:
                cfg["task"]["graph"]["start"]["node_id"] = shared_infra["start_node"]
                cfg["task"]["graph"]["goal"]["node_id"] = -1
            prior = shared_infra["prior_obs"] if shared_infra else None
            res = run_binney_receding_horizon(cfg, run_dir=None, prior_obs=prior)
            result["rmse"] = float(res["metrics"].get("rmse", np.nan))
            result["n_samples"] = int(res.get("n_samples", 0))
            result["pred_mean"] = res.get("pred_mean")
            result["pred_std"] = res.get("pred_std")
            # Trajectory from path (node IDs) -> lat/lon via Binney's own graph.
            # Reconstruct from the same config so node IDs match exactly.
            region_mm = _region_minmax(scenario_cfg)
            binney_graph = WaypointGraph.grid(
                region_mm,
                int(scenario_cfg.get("graph_n_lat", 20)),
                int(scenario_cfg.get("graph_n_lon", 20)),
                speed_mps=float(scenario_cfg.get("speed_mps", 1.0)),
                connectivity=str(scenario_cfg.get("connectivity", "4")),
            )
            path_nodes = res.get("path", [])
            if path_nodes:
                traj = []
                for nid in path_nodes:
                    lat, lon = binney_graph.node_coords(int(nid))
                    traj.append({"lat": lat, "lon": lon, "node_id": int(nid)})
                result["trajectory"] = traj
                total_time = binney_graph.path_travel_time(path_nodes) if len(path_nodes) > 1 else 0.0
                result["distance"] = total_time * float(scenario_cfg.get("speed_mps", 1.0))

        elif method == DLP:
            from .runner import run_persistent_sampling

            cfg = build_persistent_config(scenario_cfg, T, seed)
            prior = shared_infra["prior_obs"] if shared_infra else None
            res = run_persistent_sampling(cfg, run_dir=None, prior_obs=prior)
            result["rmse"] = float(res.get("final_metrics", {}).get("rmse", np.nan))
            result["n_samples"] = int(res.get("n_samples", 0))
            result["n_replans"] = int(res.get("n_replans", 0))
            result["distance"] = float(res.get("distance", np.nan))
            result["pred_mean"] = res.get("pred_mean")
            result["pred_std"] = res.get("pred_std")
            # Trajectory
            traj_pts = res.get("trajectory_points", [])
            result["trajectory"] = [
                {"lat": float(p[0]), "lon": float(p[1])}
                for p in traj_pts
            ]

        elif method == POMCP:
            if shared_infra is None:
                raise ValueError("POMCP requires shared_infra (call setup_shared_infrastructure first)")
            res = run_pomcp_episode(shared_infra, T, seed, variable=variable)
            result["rmse"] = float(res["rmse"])
            result["n_samples"] = int(res.get("n_samples", 0))
            result["travel_time"] = float(res.get("travel_time", 0.0))
            result["distance"] = result["travel_time"] * float(scenario_cfg.get("speed_mps", 1.0))
            result["trajectory"] = res["trajectory"]
            result["pred_mean"] = res.get("pred_mean")
            result["pred_std"] = res.get("pred_std")

        elif method == ADAPP:
            if shared_infra is None:
                raise ValueError("AdaPP requires shared_infra (call setup_shared_infrastructure first)")
            from oceanbench_bench.adapp_runner import run_adapp_episode
            from oceanbench_policies.ipp.adapp import AdaPPConfig

            adapp_cfg = AdaPPConfig(
                n_cell_lat=10,
                n_cell_lon=10,
                initial_variance=1.0,
                gamma=0.9,
                dp_max_iters=50,
                dp_tol=1e-8,
                eta=0.5,
                noise_variance=0.01,
                speed_mps=1.0,
                connectivity=str(scenario_cfg.get("connectivity", "4")),
                variance_backend="spgp_fitc",
                n_pseudo=50,
                refit_interval=1,
                hyperparams_mode="initial_only",
                seed=seed,
            )
            res = run_adapp_episode(
                shared_infra["graph"],
                shared_infra["region_minmax"],
                shared_infra["truth_field"],
                adapp_cfg,
                T,
                shared_infra["eval_qp"],
                shared_infra["truth_values"],
                start_lat=shared_infra["start_lat"],
                start_lon=shared_infra["start_lon"],
                variable=variable,
                prior_obs=shared_infra.get("prior_obs"),
            )
            result["rmse"] = float(res.get("rmse", np.nan))
            result["n_samples"] = int(res.get("n_steps", 0))
            result["travel_time"] = float(res.get("time_used", 0.0))
            result["distance"] = result["travel_time"] * float(scenario_cfg.get("speed_mps", 1.0))
            result["pred_mean"] = res.get("pred_mean")
            result["pred_std"] = res.get("pred_std")
            traj = res.get("trajectory", [])
            if traj and isinstance(traj[0], dict):
                result["trajectory"] = [
                    {"lat": float(t["lat"]), "lon": float(t["lon"])}
                    for t in traj
                ]
            else:
                result["trajectory"] = traj

        else:
            raise ValueError(f"Unknown method: {method!r}")

        result["runtime"] = time.perf_counter() - t0

        # Recompute RMSE on shared eval grid for fair cross-method comparison.
        if shared_infra is not None and result.get("pred_mean") is not None:
            truth_values = shared_infra["truth_values"]
            pred_mean = np.asarray(result["pred_mean"], dtype=float).ravel()
            error = pred_mean - truth_values
            finite = np.isfinite(error)
            if finite.any():
                result["rmse"] = float(np.sqrt(np.mean(error[finite] ** 2)))

    except Exception:
        logger.error("Method %s T=%s seed=%s FAILED:\n%s", method, T, seed, traceback.format_exc())
        print(f"  [FAIL] {method} T={T} seed={seed}: {traceback.format_exc()}")

    return result


# ── Sweep: run all methods × budgets × seeds ────────────────────────────


def run_sweep(
    scenario_cfg: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[str | Path] = None,
    methods: Optional[Sequence[str]] = None,
) -> Path:
    """Run the full comparison sweep and write results + plots.

    Parameters
    ----------
    scenario_cfg:
        Scenario configuration dict.  Falls back to default_scenario_config().
    output_dir:
        Base output directory.  A timestamped subdirectory is created.
    methods:
        Which methods to run.  Defaults to ALL_METHODS.

    Returns
    -------
    Path to the timestamped results directory.
    """
    if scenario_cfg is None:
        scenario_cfg = default_scenario_config()
    cfg = dict(scenario_cfg)
    if methods is None:
        methods = ALL_METHODS

    T_list: list[float] = [float(t) for t in cfg.get("T_list", [600, 900, 1200, 1500])]
    seeds: list[int] = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    T_plot: float = float(cfg.get("T_plot", T_list[-1]))
    if T_plot not in T_list:
        T_plot = min(T_list, key=lambda t: abs(t - T_plot))

    # Output directory
    scenario_name = "gulf_of_mexico"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("results") / "compare_travel_budget"
    out = Path(output_dir) / scenario_name / ts
    out.mkdir(parents=True, exist_ok=True)
    traj_dir = out / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    # Save config
    serialisable_cfg = {k: v for k, v in cfg.items()}
    serialisable_cfg["T_list"] = T_list
    serialisable_cfg["seeds"] = seeds
    serialisable_cfg["methods"] = list(methods)
    with open(out / "config.json", "w") as f:
        json.dump(serialisable_cfg, f, indent=2, default=str)

    # Build shared infra once (for POMCP & AdaPP, and Binney trajectory plotting)
    print("Building shared infrastructure...")
    shared_infra = setup_shared_infrastructure(cfg, seed=seeds[0])

    # Run sweep
    rows: list[dict[str, Any]] = []
    traj_cache: dict[str, list] = {}  # key: f"{method}_T{T}_seed{seed}"
    pred_cache: dict[str, dict] = {}  # key: same, value: {"pred_mean", "pred_std"}
    metrics_cache: dict[str, dict] = {}  # key: same, value: {"n_samples", "runtime", "rmse"}

    total_runs = len(methods) * len(T_list) * len(seeds)
    run_idx = 0
    for method in methods:
        for T in T_list:
            for seed in seeds:
                run_idx += 1
                style = _METHOD_STYLES.get(method, {"label": method})
                label = style.get("label", method)
                print(f"\n{'='*70}")
                print(f"[{run_idx}/{total_runs}] {label}  |  T={T:.0f}s  |  seed={seed}")
                print(f"{'='*70}")
                res = run_one_method(
                    method, T, seed, cfg,
                    run_dir=out,
                    shared_infra=shared_infra,
                )

                # Count distinct (lat, lon) pairs in trajectory.
                traj = res["trajectory"]
                if traj and isinstance(traj[0], dict):
                    coords = {(round(p["lat"], 4), round(p["lon"], 4)) for p in traj}
                    n_unique = len(coords)
                    node_ids = [p.get("node_id") for p in traj if "node_id" in p]
                    unique_nodes = len(set(node_ids)) if node_ids else n_unique
                else:
                    n_unique = 0
                    unique_nodes = 0

                # Travel time (from result or compute from trajectory)
                travel_time = res.get("travel_time", np.nan)
                if not np.isfinite(travel_time):
                    dist = res.get("distance", np.nan)
                    speed = float(cfg.get("speed_mps", 1.0))
                    if np.isfinite(dist) and speed > 0:
                        travel_time = dist / speed

                rmse = res["rmse"]
                n_samples = res["n_samples"]
                runtime = res.get("runtime", np.nan)
                distance = res.get("distance", np.nan)
                n_replans = res.get("n_replans", 0)

                rmse_str = f"{rmse:.4f}" if np.isfinite(rmse) else "NaN"
                rmse_per_sample = rmse / max(n_samples, 1) if (np.isfinite(rmse) and n_samples > 0) else np.nan

                print(f"  RMSE             : {rmse_str}")
                print(f"  Samples collected: {n_samples}")
                print(f"  Unique locations : {n_unique}")
                print(f"  Unique nodes     : {unique_nodes}")
                print(f"  Trajectory len   : {len(traj)} points")
                print(f"  Travel time used : {travel_time:.0f}s / {T:.0f}s budget ({100*travel_time/T:.1f}%)" if np.isfinite(travel_time) else f"  Travel time used : N/A / {T:.0f}s budget")
                print(f"  Distance         : {distance:.0f}m" if np.isfinite(distance) else "  Distance         : N/A")
                print(f"  Replans          : {n_replans}" if n_replans > 0 else "  Replans          : N/A")
                print(f"  RMSE/sample      : {rmse_per_sample:.6f}" if np.isfinite(rmse_per_sample) else "  RMSE/sample      : N/A")
                print(f"  Wall-clock time  : {runtime:.1f}s" if np.isfinite(runtime) else "  Wall-clock time  : N/A")
                print(f"{'-'*70}")

                rows.append({
                    "method": res["method"],
                    "T": res["T"],
                    "seed": res["seed"],
                    "rmse": res["rmse"],
                    "distance": res.get("distance", np.nan),
                    "n_samples": res["n_samples"],
                    "n_unique_locations": n_unique,
                    "n_unique_nodes": unique_nodes,
                    "n_replans": res.get("n_replans", 0),
                    "runtime": res["runtime"],
                    "travel_time": travel_time if np.isfinite(travel_time) else np.nan,
                    "traj_length": len(traj),
                })

                # Save trajectory and predictions
                traj_key = f"{method}_T{int(T)}_seed{seed}"
                traj_cache[traj_key] = res["trajectory"]
                pred_cache[traj_key] = {
                    "pred_mean": res.get("pred_mean"),
                    "pred_std": res.get("pred_std"),
                }
                metrics_cache[traj_key] = {
                    "n_samples": res["n_samples"],
                    "runtime": res.get("runtime", np.nan),
                    "rmse": res["rmse"],
                }
                traj_file = traj_dir / f"{traj_key}.json"
                with open(traj_file, "w") as f:
                    json.dump(res["trajectory"], f, default=str)

    # Write summary CSV
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)

    # Aggregated CSV (mean ± std)
    agg = (
        df.groupby(["method", "T"])["rmse"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std"})
    )
    agg.to_csv(out / "aggregated.csv", index=False)

    # Print final summary table
    print(f"\n{'='*90}")
    print(f"  FINAL SUMMARY  —  {len(seeds)} seed(s), {len(T_list)} budget(s), {len(methods)} method(s)")
    print(f"{'='*90}")
    header = f"{'Method':<22} {'T':>8} {'RMSE':>10} {'#Samples':>9} {'Unique':>7} {'TravTime':>9} {'Replans':>8} {'Runtime':>9} {'RMSE/samp':>10}"
    print(header)
    print(f"{'-'*90}")
    for _, row in df.iterrows():
        m_label = _METHOD_STYLES.get(row["method"], {}).get("label", row["method"])
        rmse_s = f"{row['rmse']:.4f}" if np.isfinite(row["rmse"]) else "NaN"
        tt_s = f"{row.get('travel_time', np.nan):.0f}s" if np.isfinite(row.get("travel_time", np.nan)) else "N/A"
        rp_s = f"{int(row.get('n_replans', 0))}" if row.get("n_replans", 0) > 0 else "-"
        rt_s = f"{row['runtime']:.1f}s" if np.isfinite(row["runtime"]) else "N/A"
        rps = row["rmse"] / max(row["n_samples"], 1) if (np.isfinite(row["rmse"]) and row["n_samples"] > 0) else np.nan
        rps_s = f"{rps:.6f}" if np.isfinite(rps) else "N/A"
        print(f"{m_label:<22} {row['T']:>8.0f} {rmse_s:>10} {row['n_samples']:>9} {row.get('n_unique_locations', 0):>7} {tt_s:>9} {rp_s:>8} {rt_s:>9} {rps_s:>10}")

    # Print aggregated means across seeds
    if len(seeds) > 1:
        print(f"\n{'='*90}")
        print(f"  AGGREGATED (mean ± std across {len(seeds)} seeds)")
        print(f"{'='*90}")
        agg_full = df.groupby(["method", "T"]).agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            samples_mean=("n_samples", "mean"), samples_std=("n_samples", "std"),
            unique_mean=("n_unique_locations", "mean"),
            runtime_mean=("runtime", "mean"), runtime_std=("runtime", "std"),
        ).reset_index()
        header2 = f"{'Method':<22} {'T':>8} {'RMSE (mean±std)':>20} {'#Samples (mean)':>16} {'Unique (mean)':>14} {'Runtime (mean)':>16}"
        print(header2)
        print(f"{'-'*90}")
        for _, r in agg_full.iterrows():
            m_label = _METHOD_STYLES.get(r["method"], {}).get("label", r["method"])
            rmse_s = f"{r['rmse_mean']:.4f}±{r['rmse_std']:.4f}" if np.isfinite(r["rmse_mean"]) else "NaN"
            print(f"{m_label:<22} {r['T']:>8.0f} {rmse_s:>20} {r['samples_mean']:>16.1f} {r['unique_mean']:>14.1f} {r['runtime_mean']:>13.1f}s")
    print(f"{'='*90}\n")

    # Plots
    print("Generating plots...")
    _plot_rmse_vs_T(agg, out)
    _plot_prior_baseline(shared_infra, out)
    _plot_trajectories(traj_cache, pred_cache, shared_infra, T_plot, out, methods, metrics_cache=metrics_cache)

    print(f"\nResults saved to: {out}")
    return out


# ── Plotting ─────────────────────────────────────────────────────────────

_METHOD_STYLES = {
    BINNEY: {"color": "tab:blue", "marker": "o", "label": "Binney (GRG)"},
    DLP: {"color": "tab:orange", "marker": "s", "label": "DLP (Persistent)"},
    POMCP: {"color": "tab:green", "marker": "^", "label": "POMCP"},
    ADAPP: {"color": "tab:red", "marker": "D", "label": "AdaPP"},
}


def _plot_rmse_vs_T(agg: pd.DataFrame, out: Path) -> None:
    """Plot RMSE vs travel-time budget with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("T")
        style = _METHOD_STYLES.get(method, {"color": "gray", "marker": "x", "label": method})
        ax.errorbar(
            sub["T"], sub["rmse_mean"], yerr=sub["rmse_std"],
            color=style["color"], marker=style["marker"],
            label=style["label"], capsize=4, linewidth=1.5,
        )
    ax.set_xlabel("Travel-time budget T (seconds)")
    ax.set_ylabel("RMSE (mean ± std across seeds)")
    ax.set_title("RMSE vs Travel-Time Budget — 4-Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_rmse_vs_T.png", dpi=150)
    plt.close(fig)


def _plot_prior_baseline(
    infra: Mapping[str, Any],
    out: Path,
) -> None:
    """Plot prior baseline: GT, prior prediction, prior std, prior error."""
    from oceanbench_models.belief.field import GPFieldModel

    eval_qp: QueryPoints = infra["eval_qp"]
    truth_values: np.ndarray = infra["truth_values"]
    prior_obs: ObservationBatch = infra["prior_obs"]

    lats = np.asarray(eval_qp.lats, dtype=float)
    lons = np.asarray(eval_qp.lons, dtype=float)
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))

    vmin = float(np.nanmin(truth_values))
    vmax = float(np.nanmax(truth_values))

    # Fit a GP on the prior observations to get baseline prediction.
    # Use RBF without scaling, minimal fitting — visualization only.
    prior_gp = GPFieldModel(
        config={"lengthscale": 1.0, "variance": 1.0, "noise": 0.01,
                "include_time": False, "include_depth": False,
                "use_scaling": False, "training_iters": 5},
        seed=0,
    )
    prior_gp.fit(prior_obs)
    pred = prior_gp.predict(eval_qp)
    pred_mean = np.asarray(pred.mean, dtype=float).ravel()
    pred_std = np.asarray(pred.std, dtype=float).ravel() if pred.std is not None else np.full_like(pred_mean, np.nan)
    error = pred_mean - truth_values

    prior_lats = np.asarray(prior_obs.lats, dtype=float)
    prior_lons = np.asarray(prior_obs.lons, dtype=float)

    # Interpolate fields.
    lon_gt, lat_gt, z_gt = _interp_to_grid(lats, lons, truth_values)
    lon_p, lat_p, z_p = _interp_to_grid(lats, lons, pred_mean)
    lon_s, lat_s, z_s = _interp_to_grid(lats, lons, pred_std)
    lon_e, lat_e, z_e = _interp_to_grid(lats, lons, error)

    err_max = float(np.nanmax(np.abs(error[np.isfinite(error)]))) if np.any(np.isfinite(error)) else 1.0
    rmse_val = float(np.sqrt(np.nanmean(error[np.isfinite(error)] ** 2)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    row_labels = ["Ground Truth", "Prior Prediction", "Prior Std", "Prior Error"]

    # GT + prior sample dots
    im_gt = axes[0].pcolormesh(lon_gt, lat_gt, z_gt, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    axes[0].scatter(prior_lons, prior_lats, c="red", s=20, edgecolors="white", linewidths=0.5, zorder=5)
    axes[0].set_title(f"Ground Truth\n({prior_obs.size} prior samples shown)")

    # Prior prediction + sample dots
    im_pred = axes[1].pcolormesh(lon_p, lat_p, z_p, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
    axes[1].scatter(prior_lons, prior_lats, c="red", s=20, edgecolors="white", linewidths=0.5, zorder=5)
    axes[1].set_title("Prior GP Prediction")

    # Prior std
    im_std = axes[2].pcolormesh(lon_s, lat_s, z_s, cmap="magma", shading="auto")
    axes[2].scatter(prior_lons, prior_lats, c="cyan", s=20, edgecolors="white", linewidths=0.5, zorder=5)
    axes[2].set_title("Prior Std (Uncertainty)")

    # Prior error
    im_err = axes[3].pcolormesh(lon_e, lat_e, z_e, cmap="RdBu_r", vmin=-err_max, vmax=err_max, shading="auto")
    axes[3].text(
        0.02, 0.02, f"RMSE={rmse_val:.2f}", transform=axes[3].transAxes,
        fontsize=10, color="black", va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, pad=2),
    )
    axes[3].set_title("Prior Error (Pred - Truth)")

    for ax in axes:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    fig.subplots_adjust(right=0.92, wspace=0.25)
    # Colorbars
    cax0 = fig.add_axes([0.93, 0.55, 0.012, 0.35])
    fig.colorbar(im_gt, cax=cax0, label="Temperature")
    cax1 = fig.add_axes([0.93, 0.10, 0.012, 0.35])
    fig.colorbar(im_std, cax=cax1, label="Std")

    fig.suptitle(f"Prior Baseline — {prior_obs.size} shared observations", fontsize=13, y=1.02)
    fig.savefig(out / "plot_prior_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_trajectory(ax: plt.Axes, traj: list, max_plot_pts: int = 200,
                        show_label: bool = False) -> None:
    """Overlay a trajectory on an axes with subsampling for dense paths."""
    if not traj:
        return
    t_lats = [p["lat"] for p in traj if isinstance(p, dict)]
    t_lons = [p["lon"] for p in traj if isinstance(p, dict)]
    if not t_lats:
        return
    if len(t_lats) > max_plot_pts:
        step = max(1, len(t_lats) // max_plot_pts)
        t_lats_p = t_lats[::step] + [t_lats[-1]]
        t_lons_p = t_lons[::step] + [t_lons[-1]]
    else:
        t_lats_p, t_lons_p = t_lats, t_lons
    ax.plot(t_lons_p, t_lats_p, "-", color="cyan", linewidth=1.8, alpha=0.9, zorder=3)
    # Plot unique locations as visible markers (yellow circles).
    unique_coords = list({(round(la, 6), round(lo, 6)) for la, lo in zip(t_lats, t_lons)})
    u_lats = [c[0] for c in unique_coords]
    u_lons = [c[1] for c in unique_coords]
    ax.scatter(u_lons, u_lats, c="yellow", s=30, edgecolors="black",
               linewidths=0.6, zorder=4, marker="o")
    ax.plot(t_lons_p[0], t_lats_p[0], "g*", markersize=14, zorder=5)
    ax.plot(t_lons_p[-1], t_lats_p[-1], "r*", markersize=14, zorder=5)
    if show_label:
        n_unique = len(unique_coords)
        ax.text(
            0.02, 0.02, f"n={len(t_lats)} ({n_unique} unique)", transform=ax.transAxes,
            fontsize=14, color="white", va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )


def _interp_to_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    grid_res: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate scattered points onto a regular grid for pcolormesh."""
    from scipy.interpolate import griddata

    lon_grid = np.linspace(float(np.nanmin(lons)), float(np.nanmax(lons)), grid_res)
    lat_grid = np.linspace(float(np.nanmin(lats)), float(np.nanmax(lats)), grid_res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    finite = np.isfinite(values)
    if not finite.any():
        return lon_mesh, lat_mesh, np.full_like(lon_mesh, np.nan)

    z = griddata(
        (lons[finite], lats[finite]),
        values[finite],
        (lon_mesh, lat_mesh),
        method="cubic",
    )
    return lon_mesh, lat_mesh, z


def _plot_trajectories(
    traj_cache: dict[str, list],
    pred_cache: dict[str, dict],
    infra: Mapping[str, Any],
    T_plot: float,
    out: Path,
    methods: Sequence[str],
    *,
    metrics_cache: Optional[dict[str, dict]] = None,
) -> None:
    """Plot 4-row comparison: GT+path, Prediction+path, Std, Error."""
    eval_qp: QueryPoints = infra["eval_qp"]
    truth_values: np.ndarray = infra["truth_values"]
    prior_obs = infra.get("prior_obs")

    lats = np.asarray(eval_qp.lats, dtype=float)
    lons = np.asarray(eval_qp.lons, dtype=float)
    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))

    n_methods = len(methods)
    fig, axes = plt.subplots(4, n_methods, figsize=(6.5 * n_methods, 22), squeeze=False)

    # Shared color limits for GT and Prediction (rows 0 & 1).
    vmin = float(np.nanmin(truth_values))
    vmax = float(np.nanmax(truth_values))

    # Pre-interpolate GT once (same for all columns).
    lon_gt, lat_gt, z_gt = _interp_to_grid(lats, lons, truth_values)

    # Collect std/error ranges across methods for shared colorbars.
    all_std, all_err_max = [], []
    for method in methods:
        traj_key = f"{method}_T{int(T_plot)}_seed0"
        preds = pred_cache.get(traj_key, {})
        ps = preds.get("pred_std")
        pm = preds.get("pred_mean")
        if ps is not None:
            all_std.append(ps)
        if pm is not None:
            err = np.asarray(pm, dtype=float) - truth_values
            finite_err = err[np.isfinite(err)]
            if finite_err.size > 0:
                all_err_max.append(float(np.nanmax(np.abs(finite_err))))
    std_vmin = 0.0
    std_vmax = float(np.nanmax(np.concatenate(all_std))) if all_std else 1.0
    err_max = max(all_err_max) if all_err_max else 1.0

    row_labels = ["Ground Truth + Path", "GP Prediction + Path", "Predictive Std", "Error (Pred - Truth)"]

    # Shared tick locators: 3-4 ticks on each axis.
    lon_ticks = np.round(np.linspace(lon_min, lon_max, 4), 1)
    lat_ticks = np.round(np.linspace(lat_min, lat_max, 4), 1)
    im_gt = im_pred = im_std = im_err = None

    for i, method in enumerate(methods):
        traj_key = f"{method}_T{int(T_plot)}_seed0"
        traj = traj_cache.get(traj_key, [])
        preds = pred_cache.get(traj_key, {})
        pred_mean = preds.get("pred_mean")
        pred_std = preds.get("pred_std")

        style = _METHOD_STYLES.get(method, {"label": method})
        label = style["label"]

        # -- Row 0: Ground Truth + Path + Prior samples --
        ax = axes[0][i]
        im_gt = ax.pcolormesh(lon_gt, lat_gt, z_gt, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        if prior_obs is not None:
            ax.scatter(
                np.asarray(prior_obs.lons, dtype=float),
                np.asarray(prior_obs.lats, dtype=float),
                c="red", s=15, edgecolors="white", linewidths=0.4, zorder=4,
            )
        _overlay_trajectory(ax, traj)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(f"{label}", fontsize=22, fontweight="bold")
        if i == 0:
            ax.set_ylabel(row_labels[0], fontsize=20)
        else:
            ax.set_yticklabels([])

        # -- Row 1: GP Prediction + Path --
        ax = axes[1][i]
        if pred_mean is not None:
            lon_p, lat_p, z_p = _interp_to_grid(lats, lons, np.asarray(pred_mean, dtype=float))
            im_pred = ax.pcolormesh(lon_p, lat_p, z_p, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        else:
            ax.text(0.5, 0.5, "No predictions", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        _overlay_trajectory(ax, traj, show_label=True)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        if i == 0:
            ax.set_ylabel(row_labels[1], fontsize=20)
        else:
            ax.set_yticklabels([])

        # -- Row 2: Predictive Std --
        ax = axes[2][i]
        if pred_std is not None:
            lon_s, lat_s, z_s = _interp_to_grid(lats, lons, np.asarray(pred_std, dtype=float))
            im_std = ax.pcolormesh(lon_s, lat_s, z_s, cmap="magma", vmin=std_vmin, vmax=std_vmax, shading="auto")
        else:
            ax.text(0.5, 0.5, "No std available", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        if i == 0:
            ax.set_ylabel(row_labels[2], fontsize=20)
        else:
            ax.set_yticklabels([])

        # -- Row 3: Error (Pred - Truth) --
        ax = axes[3][i]
        if pred_mean is not None:
            error = np.asarray(pred_mean, dtype=float) - truth_values
            lon_e, lat_e, z_e = _interp_to_grid(lats, lons, error)
            im_err = ax.pcolormesh(lon_e, lat_e, z_e, cmap="RdBu_r", vmin=-err_max, vmax=err_max, shading="auto")
            rmse_val = float(np.sqrt(np.nanmean(error[np.isfinite(error)] ** 2)))
            mc = (metrics_cache or {}).get(traj_key, {})
            runtime = mc.get("runtime", np.nan)
            # Compute unique locations from trajectory for fair efficiency metric.
            if traj and isinstance(traj[0], dict):
                n_unique_locs = len({(round(p["lat"], 4), round(p["lon"], 4)) for p in traj})
            else:
                n_unique_locs = mc.get("n_samples", 0)
            rmse_per_unique = rmse_val / max(n_unique_locs, 1) if n_unique_locs > 0 else np.nan
            lines = [f"RMSE={rmse_val:.2f}"]
            if n_unique_locs > 0:
                lines.append(f"RMSE/loc={rmse_per_unique:.4f} ({n_unique_locs} locs)")
            if np.isfinite(runtime):
                lines.append(f"Planning: {runtime:.1f}s")
            ax.text(
                0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=14, color="black", va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, pad=2),
            )
        else:
            ax.text(0.5, 0.5, "No predictions", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude", fontsize=19)
        if i == 0:
            ax.set_ylabel(row_labels[3], fontsize=20)
        else:
            ax.set_yticklabels([])

    # Configure ticks for all axes.
    for row in range(4):
        for col in range(n_methods):
            ax = axes[row][col]
            ax.set_xticks(lon_ticks)
            ax.set_yticks(lat_ticks)
            ax.tick_params(axis="both", which="major", labelsize=16,
                           length=9, width=1.3)
            # Remove x-tick labels from non-bottom rows.
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([f"{v:.1f}" for v in lon_ticks], rotation=30, ha="right")

    fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.06,
                        hspace=0.15, wspace=0.08)

    # Colorbars aligned to each row using axes positions.
    def _row_colorbar(fig, im, row_axes, label):
        """Add a colorbar aligned to a row of axes."""
        pos_top = row_axes[0].get_position()
        pos_bot = row_axes[-1].get_position()
        cax = fig.add_axes([0.90, pos_top.y0, 0.015, pos_top.y1 - pos_top.y0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=18)
        cb.ax.tick_params(labelsize=15)

    fig.canvas.draw()  # force layout so get_position() works
    if im_gt is not None:
        _row_colorbar(fig, im_gt, axes[0], "Temperature")
    if im_pred is not None:
        _row_colorbar(fig, im_pred, axes[1], "Temperature")
    if im_std is not None:
        _row_colorbar(fig, im_std, axes[2], "Std")
    if im_err is not None:
        _row_colorbar(fig, im_err, axes[3], "Error")

    fig.suptitle(f"4-Method Comparison — T={int(T_plot)}s, seed=0", fontsize=26, y=1)
    fig.savefig(out / f"plot_trajectories_T{int(T_plot)}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    run_sweep()
