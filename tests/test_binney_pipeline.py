from __future__ import annotations

import numpy as np

from oceanbench_data_provider import DataProvider
from oceanbench_tasks.mapping.binney_task import BinneyMappingTask
from oceanbench_policies.ipp.grg import GRGConfig, GRGPlanner


def test_binney_pipeline_small_graph():
    # Minimal synthetic configs avoiding empirical covariance.
    scenario_cfg = {
        "name": "binney_test",
        "variable": "temp",
        "region": {
            "lat_min": 0.0,
            "lat_max": 1.0,
            "lon_min": 0.0,
            "lon_max": 1.0,
        },
        "time_range": (
            np.datetime64("2014-01-01T00:00:00"),
            np.datetime64("2014-01-02T00:00:00"),
        ),
        "metadata": {
            "product_id": "hycom_glbv0.08_reanalysis_53x",
        },
    }
    task_cfg = {
        "graph": {
            "type": "grid",
            "seed": 0,
            "grid": {"n_lat": 2, "n_lon": 2},
            "start": {"node_id": 0},
            "goal": {"node_id": 3},
        },
        "robot": {"speed_mps": 1.0},
        "sampling": {
            "mode": "nodes+edges",
            "edge_spacing_m": 0.0,
            "include_nodes": True,
            "measurement_noise_var": 0.01,
        },
        "time": {
            "dynamic": True,
            "tau0": "2014-01-01T00:00:00",
            "snap_to_provider_grid": False,
            "interp": "linear",
        },
        "objective": {"type": "emse", "eval_set": "dense_grid"},
        "eval": {
            "grid_n_lat": 4,
            "grid_n_lon": 4,
            "max_points": 100,
            "subsample_strategy": "random",
        },
        "covariance": {
            "backend": "kernel",
            "kernel": {
                "lengthscale_space": 1.0,
                "lengthscale_time": 1.0,
                "variance": 1.0,
                "noise": 0.01,
                "fit": False,
            },
        },
    }

    provider = DataProvider(config={"cache_dir": None})
    rng = np.random.default_rng(0)

    task = BinneyMappingTask.from_configs(
        scenario_cfg=scenario_cfg,
        task_cfg=task_cfg,
        provider=provider,
        rng=rng,
    )

    cfg = GRGConfig(depth=0, n_splits=3)
    planner = GRGPlanner(
        graph=task.graph,
        objective=task.objective,
        sampling_config=task.sampling_config,
        config=cfg,
    )

    B = task.graph.shortest_time(0, 3) + 1.0
    tau0 = np.datetime64(task.time_config.get("tau0", "2014-01-01T00:00:00"))
    path, samples, gain = planner.plan(0, 3, B, tau0, X_items=None)

    assert path is not None
    assert samples is not None
    assert isinstance(gain, float)

