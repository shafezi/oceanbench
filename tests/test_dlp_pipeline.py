"""End-to-end tiny persistent sampling pipeline test."""

from __future__ import annotations

import numpy as np
import xarray as xr

from oceanbench_bench.runner import run_persistent_sampling
from oceanbench_data_provider import DataProvider


def _synthetic_dataset() -> xr.Dataset:
    lats = np.linspace(24.0, 26.0, 12)
    lons = np.linspace(-86.0, -84.0, 12)
    times = np.array(
        [
            np.datetime64("2014-01-01T00:00:00"),
            np.datetime64("2014-01-01T06:00:00"),
            np.datetime64("2014-01-01T12:00:00"),
            np.datetime64("2014-01-01T18:00:00"),
            np.datetime64("2014-01-02T00:00:00"),
        ],
        dtype="datetime64[ns]",
    )
    T, La, Lo = np.meshgrid(
        np.arange(times.size, dtype=float),
        lats,
        lons,
        indexing="ij",
    )
    data = np.sin(La * 0.2) + np.cos(Lo * 0.15) + 0.1 * T
    ds = xr.Dataset(
        data_vars={"temp": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds


def test_dlp_pipeline_tiny(monkeypatch):
    ds = _synthetic_dataset()

    def _fake_subset(self, *args, **kwargs):  # noqa: ANN001
        _ = self, args, kwargs
        return ds

    monkeypatch.setattr(DataProvider, "subset", _fake_subset)

    cfg = {
        "seed": 3,
        "scenario": {
            "name": "tiny_dlp",
            "product_id": "synthetic",
            "variable": "temp",
            "region": {
                "lat_min": 24.0,
                "lat_max": 26.0,
                "lon_min": -86.0,
                "lon_max": -84.0,
            },
            "time_window": ["2014-01-01T00:00:00", "2014-01-02T00:00:00"],
        },
        "candidates": {
            "type": "grid",
            "grid": {"n_lat": 10, "n_lon": 10},
            "max_points": 100,
            "seed": 3,
        },
        "field_model": {
            "backend": "sogp_paper",
            "params": {
                "max_basis_size": 20,
                "novelty_threshold": 1e-6,
                "include_time": True,
                "include_depth": False,
                "lengthscale": 1.0,
                "variance": 1.0,
            },
        },
        "planner": {
            "type": "mi_greedy",
            "batch_size_n": 6,
            "max_candidates_for_greedy": 100,
        },
        "mi": {"X_set": "candidate_grid", "jitter": 1e-8},
        "routing": {"backend": "networkx", "mode": "open_end_anywhere", "metric": "haversine"},
        "truth": {"mode": "dynamic_provider", "time_mode": "interpolate", "frame_change_every_samples": 8},
        "replan": {"trigger": "end_of_batch"},
        "hyperparams": {"mode": "rho_trigger", "rho0": 0.8, "on_update": "keep_current_plan"},
        "noise": {"mode": "fixed", "fixed_sigma2": 1e-3, "estimate_method": "residual"},
        "mission": {
            "initial_samples": 4,
            "max_samples": 28,
            "sample_interval_s": 900.0,
            "speed_mps": 1.0,
            "measurement_noise_std": 0.01,
        },
        "eval": {
            "grid": {"fixed": True, "n_lat": 20, "n_lon": 20},
            "max_points": 1000,
            "subsample_strategy": "stratified",
            "times": "sequence",
            "sequence_length": 3,
            "every_n_samples": 7,
        },
    }

    result = run_persistent_sampling(cfg, run_dir=None)
    assert result["n_samples"] >= cfg["mission"]["initial_samples"]
    assert result["n_samples"] <= cfg["mission"]["max_samples"]
    assert "final_metrics" in result
    fm = result["final_metrics"]
    assert np.isfinite(float(fm.get("rmse", np.nan)))
    assert np.isfinite(float(fm.get("mae", np.nan)))
