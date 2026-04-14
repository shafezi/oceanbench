"""
AdaPP adaptive sampling example.

Runs AdaPP and lawnmower baseline on a synthetic field for multiple time
budgets and reports RMSE comparison.

Usage
-----
    python examples/run_adapp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _synth_field(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return 22.0 + 3.0 * np.sin(0.5 * lat) + 2.0 * np.cos(0.7 * lon)


def main() -> None:
    from oceanbench_core import WaypointGraph
    from oceanbench_core.types import QueryPoints
    from oceanbench_bench.adapp_runner import run_adapp_time_budget_sweep
    from oceanbench_policies.ipp.adapp import AdaPPConfig

    region = {"lat_min": 22.0, "lat_max": 28.0, "lon_min": -92.0, "lon_max": -86.0}
    seed = 42

    # Build graph.
    graph = WaypointGraph.grid(region, 15, 15, speed_mps=1.0, connectivity="4", seed=seed)
    print(f"Graph: {graph.graph.number_of_nodes()} nodes")

    # Eval grid.
    eval_lats = np.linspace(22.0, 28.0, 40)
    eval_lons = np.linspace(-92.0, -86.0, 40)
    lat_m, lon_m = np.meshgrid(eval_lats, eval_lons, indexing="ij")
    eval_qp = QueryPoints(lats=lat_m.ravel(), lons=lon_m.ravel())
    truth_values = _synth_field(lat_m.ravel(), lon_m.ravel())

    config = AdaPPConfig(
        n_cell_lat=5, n_cell_lon=5, initial_variance=1.0,
        gamma=0.8, eta=0.3, noise_variance=0.01,
        speed_mps=1.0, connectivity="4",
        variance_backend="spgp_fitc", n_pseudo=20,
        refit_interval=3, seed=seed,
    )

    time_budgets = [10.0, 20.0, 30.0, 40.0]
    print(f"Time budgets: {time_budgets}")

    results = run_adapp_time_budget_sweep(
        graph, region, None, config, time_budgets,
        eval_qp, truth_values, variable="temp",
    )

    # Print comparison.
    print("\n=== RMSE Comparison ===")
    print(f"{'T (s)':>8} {'AdaPP':>10} {'Lawnmower':>12}")
    for ar, lr in zip(results["adapp"], results["lawnmower"]):
        print(f"{ar['time_budget']:>8.0f} {ar['rmse']:>10.4f} {lr['rmse']:>12.4f}")

    # Save results.
    run_dir = REPO_ROOT / "results" / "adapp_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "time_budgets": time_budgets,
        "adapp_rmse": [r["rmse"] for r in results["adapp"]],
        "lawnmower_rmse": [r["rmse"] for r in results["lawnmower"]],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {run_dir}")

    # Plot if matplotlib available.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(time_budgets, summary["adapp_rmse"], "b-o", label="AdaPP")
        ax.plot(time_budgets, summary["lawnmower_rmse"], "r-s", label="Lawnmower")
        ax.set_xlabel("Time Budget (s)")
        ax.set_ylabel("RMSE")
        ax.set_title("AdaPP vs Lawnmower: RMSE vs Time Budget")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(run_dir / "rmse_vs_T.png"), dpi=150)
        print(f"Plot saved to {run_dir / 'rmse_vs_T.png'}")
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
