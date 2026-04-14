"""End-to-end tests for AdaPP integration."""

import numpy as np
import pytest

from oceanbench_core import WaypointGraph
from oceanbench_core.types import QueryPoints
from oceanbench_policies.ipp.adapp import AdaPPConfig, AdaPPPlanner, LawnmowerBaseline, build_field_model


REGION = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0}
SEED = 42


def _synth_truth(lat, lon):
    return 20.0 + 2.0 * np.sin(0.5 * lat) + 1.5 * np.cos(0.3 * lon)


class TestAdaPPEndToEnd:
    def test_full_episode_with_spgp_fitc(self):
        """AdaPP with SPGP FITC backend completes and produces RMSE."""
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        config = AdaPPConfig(
            n_cell_lat=3, n_cell_lon=3, gamma=0.8, eta=0.3,
            noise_variance=0.01, speed_mps=1.0, connectivity="4",
            variance_backend="spgp_fitc", n_pseudo=10,
            refit_interval=2, seed=SEED,
        )
        model = build_field_model(config, seed=SEED)
        planner = AdaPPPlanner(graph, REGION, model, config)
        result = planner.run_episode(5.0, 5.0, time_budget=1_500_000.0, variable="temp")

        assert result["n_steps"] >= 1
        assert result["time_used"] > 0
        assert result["observations"] is not None

    def test_full_episode_with_svgp(self):
        """AdaPP with SVGP backend completes."""
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        config = AdaPPConfig(
            n_cell_lat=3, n_cell_lon=3, gamma=0.8, eta=0.3,
            noise_variance=0.01, speed_mps=1.0,
            variance_backend="svgp_gpytorch", n_pseudo=10,
            refit_interval=2, seed=SEED,
        )
        model = build_field_model(config, seed=SEED)
        planner = AdaPPPlanner(graph, REGION, model, config)
        result = planner.run_episode(5.0, 5.0, time_budget=1_500_000.0, variable="temp")

        assert result["n_steps"] >= 1

    def test_deterministic_under_seed(self):
        """Same seed → same trajectory."""
        results = []
        for _ in range(2):
            graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
            config = AdaPPConfig(
                n_cell_lat=3, n_cell_lon=3, gamma=0.8, eta=0.3,
                speed_mps=1.0, variance_backend="spgp_fitc", n_pseudo=10,
                refit_interval=999, seed=SEED,
            )
            model = build_field_model(config, seed=SEED)
            planner = AdaPPPlanner(graph, REGION, model, config)
            result = planner.run_episode(5.0, 5.0, time_budget=1_500_000.0)
            results.append(result)

        traj1 = results[0]["trajectory"]
        traj2 = results[1]["trajectory"]
        assert len(traj1) == len(traj2)
        for t1, t2 in zip(traj1, traj2):
            assert t1["cell_row"] == t2["cell_row"]
            assert t1["cell_col"] == t2["cell_col"]

    def test_lawnmower_episode(self):
        """Lawnmower baseline completes within budget."""
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        lm = LawnmowerBaseline(graph, speed_mps=1.0)
        result = lm.run_episode(5.0, 5.0, time_budget=1_500_000.0, variable="temp")
        assert result["n_steps"] >= 1

    def test_adapp_runner_sweep(self):
        """Sweep over two time budgets should produce results for both methods."""
        from oceanbench_bench.adapp_runner import run_adapp_time_budget_sweep

        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        config = AdaPPConfig(
            n_cell_lat=3, n_cell_lon=3, gamma=0.8, eta=0.3,
            speed_mps=1.0, variance_backend="spgp_fitc", n_pseudo=10,
            refit_interval=3, seed=SEED,
        )

        eval_lats = np.linspace(0.5, 9.5, 10)
        eval_lons = np.linspace(0.5, 9.5, 10)
        lat_m, lon_m = np.meshgrid(eval_lats, eval_lons, indexing="ij")
        eval_qp = QueryPoints(lats=lat_m.ravel(), lons=lon_m.ravel())
        truth_values = _synth_truth(lat_m.ravel(), lon_m.ravel())

        results = run_adapp_time_budget_sweep(
            graph, REGION, None, config, [750_000.0, 1_500_000.0],
            eval_qp, truth_values, variable="temp",
        )

        assert len(results["adapp"]) == 2
        assert len(results["lawnmower"]) == 2
        for r in results["adapp"]:
            assert "rmse" in r
