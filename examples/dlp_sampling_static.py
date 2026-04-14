"""
Example: data-driven learning/planning in a static field.

Run from repository root:
    python examples/dlp_sampling_static.py
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
        "seed": 42,
        "scenario": {
            "name": "dlp_static",
            "product_id": "hycom_glbv0.08_reanalysis_53x",
            "variable": "temp",
            "region": {
                "lat_min": 24.0,
                "lat_max": 36.0,
                "lon_min": -86.0,
                "lon_max": -74.0,
            },
            "time_window": ["2014-01-01T00:00:00", "2014-01-03T00:00:00"],
        },
        "candidates": {
            "type": "grid",
            "grid": {"n_lat": 35, "n_lon": 35},
            "max_points": 1225,
            "seed": 42,
        },
        "field_model": {
            "backend": "sogp_paper",
            "params": {
                "max_basis_size": 100,
                "novelty_threshold": 1.0e-6,
                "lengthscale": 1.0,
                "variance": 1.0,
                "include_time": True,
                "include_depth": False,
            },
        },
        "planner": {
            "type": "mi_dp",
            "batch_size_n": 10,
            "dp_beam_width": 48,
        },
        "mi": {
            "X_set": "candidate_grid",
            "X_subsample": {"max_points": 500},
            "jitter": 1.0e-8,
        },
        "routing": {
            "backend": "ortools",
            "mode": "open_end_anywhere",
            "metric": "haversine",
        },
        "truth": {
            "mode": "static",
            "time_mode": "snap_to_provider",
            "static_time": "2014-01-01T00:00:00",
        },
        "replan": {"trigger": "end_of_batch"},
        "hyperparams": {
            "mode": "rho_trigger",
            "rho0": 0.6,
            "on_update": "replan_immediately",
        },
        "noise": {
            "mode": "fixed",
            "fixed_sigma2": 1e-3,
            "estimate_method": "residual",
        },
        "mission": {
            "initial_samples": 6,
            "max_samples": 150,
            "sample_interval_s": 300.0,
            "speed_mps": 1.5,
            "measurement_noise_std": 0.01,
        },
        "eval": {
            "grid": {"fixed": True, "n_lat": 110, "n_lon": 110},
            "max_points": 10000,
            "subsample_strategy": "stratified",
            "times": "single",
            "every_n_samples": 20,
        },
    }


def main() -> None:
    cfg = build_config()
    out_dir = REPO_ROOT / "results" / "dlp_sampling_static"
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
    print("Static run completed.")
    print("Summary:", result.get("final_metrics", {}))


if __name__ == "__main__":
    main()

