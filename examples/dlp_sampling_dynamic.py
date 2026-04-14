"""
Example: data-driven learning/planning in a dynamic field.

Run from repository root:
    python examples/dlp_sampling_dynamic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oceanbench_bench.runner import run_persistent_sampling


def build_config() -> dict:
    return {
        "seed": 7,
        "scenario": {
            "name": "dlp_dynamic",
            "product_id": "hycom_glbv0.08_reanalysis_53x",
            "variable": "temp",
            "region": {
                "lat_min": 24.0,
                "lat_max": 36.0,
                "lon_min": -86.0,
                "lon_max": -74.0,
            },
            "time_window": ["2014-01-01T00:00:00", "2014-01-07T00:00:00"],
        },
        "candidates": {
            "type": "grid",
            "grid": {"n_lat": 40, "n_lon": 40},
            "max_points": 1600,
            "seed": 7,
        },
        "field_model": {
            "backend": "svgp_gpytorch",
            "params": {
                "n_inducing": 96,
                "inducing_strategy": "kmeans",
                "learn_inducing_locations": True,
                "training_schedule": "per_replan",
                "fit_iters": 100,
                "update_iters": 5,
                "replan_iters": 30,
                "batch_size": 256,
                "lr": 0.01,
                "include_time": True,
                "include_depth": False,
                "use_scaling": True,
            },
        },
        "planner": {
            "type": "mi_greedy",
            "batch_size_n": 10,
            "max_candidates_for_greedy": 800,
        },
        "mi": {
            "X_set": "subsample",
            "X_subsample": {"max_points": 500},
            "jitter": 1.0e-8,
        },
        "routing": {
            "backend": "networkx",
            "mode": "open_end_anywhere",
            "metric": "haversine",
        },
        "truth": {
            "mode": "dynamic_piecewise",
            "time_mode": "interpolate",
            "frame_change_every_samples": 40,
        },
        "replan": {
            "trigger": "combined",
            "fixed_time_seconds": 3600.0,
            "uncertainty_threshold": 0.2,
            "rho_threshold": 0.6,
        },
        "hyperparams": {
            "mode": "periodic",
            "period": 50,
            "on_update": "replan_immediately",
            "rho0": 0.6,
        },
        "noise": {
            "mode": "estimate",
            "fixed_sigma2": 1e-3,
            "estimate_method": "gp_likelihood",
        },
        "mission": {
            "initial_samples": 10,
            "max_samples": 180,
            "sample_interval_s": 300.0,
            "speed_mps": 1.5,
            "measurement_noise_std": 0.01,
        },
        "eval": {
            "grid": {"fixed": True, "n_lat": 110, "n_lon": 110},
            "max_points": 10000,
            "subsample_strategy": "stratified",
            "times": "sequence",
            "sequence_length": 4,
            "every_n_samples": 20,
        },
    }


def main() -> None:
    cfg = build_config()
    out_dir = REPO_ROOT / "results" / "dlp_sampling_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_persistent_sampling(cfg, run_dir=out_dir)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": result.get("n_samples"),
                "n_replans": result.get("n_replans"),
                "final_metrics": result.get("final_metrics"),
            },
            f,
            indent=2,
            default=str,
        )
    print("Dynamic run completed.")
    print("Summary:", result.get("final_metrics", {}))


if __name__ == "__main__":
    main()

