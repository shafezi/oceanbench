"""
Adaptive Sampling POMCP demo.

Runs the POMCP policy from the paper "Adaptive Sampling using POMDPs with
Domain-Specific Considerations" on an OceanBench scenario.  Falls back to a
simple synthetic field if DataProvider is not available.

Usage
-----
    python examples/adaptive_sampling_pomcp_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure repo root is on sys.path when running from examples/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

def _load_config() -> dict:
    base = REPO_ROOT / "configs"
    with (base / "base.yaml").open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with (base / "scenarios" / "pomcp_region.yaml").open("r", encoding="utf-8") as f:
        scenario_cfg = yaml.safe_load(f) or {}
    with (base / "tasks" / "pomcp_sampling.yaml").open("r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f) or {}
    with (base / "policies" / "pomcp.yaml").open("r", encoding="utf-8") as f:
        policy_cfg = yaml.safe_load(f) or {}

    return {
        "seed": base_cfg.get("seed", 42),
        "scenario": scenario_cfg,
        "task": task_cfg,
        **policy_cfg,
    }


# -------------------------------------------------------------------------
# Synthetic fallback
# -------------------------------------------------------------------------

def _make_synthetic_field(region: dict, n_lat: int = 12, n_lon: int = 12, seed: int = 42):
    """Create a synthetic GP field model and truth arrays for demo purposes."""
    from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario
    from oceanbench_models.belief.field.gp import GPFieldModel

    rng = np.random.default_rng(seed)

    lat_min, lat_max = float(region["lat_min"]), float(region["lat_max"])
    lon_min, lon_max = float(region["lon_min"]), float(region["lon_max"])

    # Dense truth grid.
    lats_grid = np.linspace(lat_min, lat_max, 30)
    lons_grid = np.linspace(lon_min, lon_max, 30)
    lat_mesh, lon_mesh = np.meshgrid(lats_grid, lons_grid, indexing="ij")
    truth_lats = lat_mesh.ravel()
    truth_lons = lon_mesh.ravel()
    # Synthetic temperature field: smooth function + noise.
    truth_values = (
        20.0
        + 3.0 * np.sin(0.5 * (truth_lats - lat_min))
        + 2.0 * np.cos(0.7 * (truth_lons - lon_min))
        + 0.5 * rng.standard_normal(truth_lats.shape)
    )

    # Initial sparse observations.
    n_init = 15
    init_lats = rng.uniform(lat_min, lat_max, n_init)
    init_lons = rng.uniform(lon_min, lon_max, n_init)
    init_values = (
        20.0
        + 3.0 * np.sin(0.5 * (init_lats - lat_min))
        + 2.0 * np.cos(0.7 * (init_lons - lon_min))
        + 0.3 * rng.standard_normal(n_init)
    )

    obs_batch = ObservationBatch(
        lats=init_lats, lons=init_lons, values=init_values, variable="temp",
    )

    model = GPFieldModel(
        {"lengthscale": 1.5, "variance": 4.0, "noise": 0.1, "include_time": False, "include_depth": False},
        seed=seed,
    )
    model.fit(obs_batch)

    return model, obs_batch, truth_lats, truth_lons, truth_values


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    from oceanbench_core import WaypointGraph
    from oceanbench_core.types import QueryPoints
    from oceanbench_policies.pomdp import (
        BeliefAdapter,
        POMCPConfig,
        POMCPPolicy,
        POMDPObservation,
        POMDPState,
    )

    cfg = _load_config()
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    scenario_cfg = cfg["scenario"]
    task_cfg = cfg["task"]
    region = scenario_cfg["region"]

    # --- build graph ---
    graph_cfg = task_cfg.get("graph", {}).get("grid", {})
    n_lat = int(graph_cfg.get("n_lat", 12))
    n_lon = int(graph_cfg.get("n_lon", 12))
    connectivity = str(task_cfg.get("graph", {}).get("connectivity", "4"))
    speed = float(task_cfg.get("robot", {}).get("speed_mps", 1.0))

    graph = WaypointGraph.grid(
        region, n_lat, n_lon, speed_mps=speed, connectivity=connectivity, seed=seed,
    )
    print(f"Graph: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges")

    # --- build field model + belief ---
    model, init_obs, truth_lats, truth_lons, truth_values = _make_synthetic_field(
        region, n_lat=n_lat, n_lon=n_lon, seed=seed,
    )

    noise_var = float(task_cfg.get("sampling", {}).get("measurement_noise_var", 0.01))
    objective_c = float(cfg.get("pomcp", {}).get("belief", {}).get("objective_c", 1.0))
    belief = BeliefAdapter(
        model, variable="temp", objective_c=objective_c, measurement_noise_var=noise_var,
    )
    belief.seed_observations(init_obs)

    # --- build policy ---
    pomcp_config = POMCPConfig.from_mapping(cfg)
    max_steps = pomcp_config.max_steps
    policy = POMCPPolicy(graph=graph, belief=belief, config=pomcp_config)

    # --- run episode ---
    nodes = list(graph.graph.nodes)
    start_node = nodes[len(nodes) // 2]  # centre of graph
    time0 = np.datetime64("2014-01-01T00:00:00")

    state = POMDPState(node_id=start_node, time=time0, step=0, remaining_budget=None)
    rewards: list[float] = []

    print(f"\nRunning POMCP for {max_steps} steps...")
    t_start = time.perf_counter()

    for step in range(max_steps):
        actions = policy.plan_k_steps(state, task=None, constraints=None, rng=rng)
        for action in actions:
            if step >= max_steps:
                break
            # Simulate observation from truth.
            lat, lon = action.lat, action.lon
            # Find nearest truth point.
            dists = np.hypot(truth_lats - lat, truth_lons - lon)
            nearest = int(np.argmin(dists))
            true_val = truth_values[nearest]
            noisy_val = true_val + rng.normal(0, np.sqrt(noise_var))

            obs = POMDPObservation(
                value=noisy_val, noise_var=noise_var, lat=lat, lon=lon, time=state.time,
            )
            policy.observe(obs)
            reward = belief.reward_at(lat, lon)
            rewards.append(reward)

            state = POMDPState(
                node_id=action.target_node_id,
                time=state.time + np.timedelta64(1, "h"),
                step=step + 1,
            )
            step += 1

    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s, {len(rewards)} actions taken.")

    # --- save results ---
    run_dir = REPO_ROOT / "results" / "pomcp_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "trajectory.json").open("w") as f:
        json.dump(policy.trajectory, f, indent=2, default=str)
    with (run_dir / "rewards.json").open("w") as f:
        json.dump(rewards, f, indent=2)
    print(f"Results saved to {run_dir}")

    # --- simple plot ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Trajectory plot.
        traj = policy.trajectory
        traj_lats = [t["lat"] for t in traj]
        traj_lons = [t["lon"] for t in traj]

        ax = axes[0]
        ax.scatter(truth_lons, truth_lats, c=truth_values, cmap="viridis", s=8, alpha=0.4, label="Truth")
        ax.plot(traj_lons, traj_lats, "r-o", markersize=4, linewidth=1.5, label="Trajectory")
        ax.plot(traj_lons[0], traj_lats[0], "g*", markersize=12, label="Start")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("POMCP Trajectory over Truth Field")
        ax.legend(fontsize=8)

        # Reward curve.
        ax = axes[1]
        ax.plot(rewards, "b-", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward (mean + c*std)")
        ax.set_title("Reward Curve")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(run_dir / "pomcp_demo_plots.png"), dpi=150)
        print(f"Plot saved to {run_dir / 'pomcp_demo_plots.png'}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping plots.")


if __name__ == "__main__":
    main()
