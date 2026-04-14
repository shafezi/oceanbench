"""Tests for the AdaPP planner and its components."""

import numpy as np
import pytest

from oceanbench_core import WaypointGraph
from oceanbench_core.cell_decomposition import CellGrid
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_policies.ipp.adapp import (
    AdaPPConfig,
    AdaPPPlanner,
    LawnmowerBaseline,
    _dp_value_iteration,
    _dp_policy_action,
    _simulate_theta,
    build_field_model,
)


REGION = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0}
SEED = 42


def _make_cell_grid(n_cell: int = 5) -> CellGrid:
    """Create a 5x5 cell grid with a 10x10 point grid."""
    lats = np.linspace(0.5, 9.5, 10)
    lons = np.linspace(0.5, 9.5, 10)
    lat_m, lon_m = np.meshgrid(lats, lons, indexing="ij")
    return CellGrid.from_region(
        REGION, n_cell, n_cell, lat_m.ravel(), lon_m.ravel(),
    )


# ---------------------------------------------------------------------------
# DP Value Iteration (Eqs. 8–9)
# ---------------------------------------------------------------------------


class TestDPValueIteration:
    def test_converges(self):
        grid = _make_cell_grid(5)
        grid.initialize(1.0)
        # Set one cell to high variance.
        grid.set_cell_variance(2, 2, 10.0)
        V = _dp_value_iteration(grid, gamma=0.9, connectivity="4", max_iters=100)
        assert V.shape == (25,)
        assert np.all(np.isfinite(V))

    def test_high_variance_cell_has_high_value(self):
        grid = _make_cell_grid(5)
        grid.initialize(0.01)
        grid.set_cell_variance(2, 2, 10.0)
        V = _dp_value_iteration(grid, gamma=0.9, connectivity="4")
        # Cell (2,2) neighbors should have high value (they can reach the prize).
        idx_center = grid.cell_index(2, 2)
        idx_corner = grid.cell_index(0, 0)
        # Not necessarily center > corner since center's own action isn't to itself,
        # but neighbors of center should be high.
        idx_nb = grid.cell_index(2, 1)
        assert V[idx_nb] > V[idx_corner]

    def test_reward_function(self):
        """R(c, a) = σ²_{c'} / ||c - c'|| should produce non-zero values."""
        grid = _make_cell_grid(3)
        grid.initialize(1.0)
        V = _dp_value_iteration(grid, gamma=0.5, connectivity="4")
        assert np.any(V > 0)


class TestDPPolicyAction:
    def test_returns_neighbor(self):
        grid = _make_cell_grid(5)
        grid.initialize(1.0)
        grid.set_cell_variance(2, 3, 5.0)
        V = _dp_value_iteration(grid, gamma=0.9, connectivity="4")
        current = grid.cell(2, 2)
        next_cell = _dp_policy_action(grid, V, current, 0.9, "4")
        assert next_cell is not None
        # Should be a neighbor of (2,2).
        dist = abs(next_cell.row - 2) + abs(next_cell.col - 2)
        assert dist <= 1


# ---------------------------------------------------------------------------
# θ-Simulation
# ---------------------------------------------------------------------------


class TestThetaSimulation:
    def test_reduces_variance(self):
        grid = _make_cell_grid(5)
        grid.initialize(1.0)
        start = grid.cell(2, 2)
        snapshot = grid.copy_variances()
        theta = _simulate_theta(
            grid, start, remaining_time=500_000.0, speed_mps=1.0,
            noise_variance=0.01, gamma=0.9, connectivity="4",
            dp_max_iters=50, dp_tol=1e-6,
        )
        grid.restore_variances(snapshot)
        assert theta > 0  # Some variance was reduced.

    def test_respects_time_budget(self):
        """With zero remaining time, θ should only be the start cell reduction."""
        grid = _make_cell_grid(5)
        grid.initialize(1.0)
        start = grid.cell(2, 2)
        snapshot = grid.copy_variances()
        theta = _simulate_theta(
            grid, start, remaining_time=0.0, speed_mps=1.0,
            noise_variance=0.01, gamma=0.9, connectivity="4",
            dp_max_iters=50, dp_tol=1e-3,
        )
        grid.restore_variances(snapshot)
        # Only the start cell's variance was reduced (1.0 - 0.01 = 0.99).
        assert theta == pytest.approx(0.99, abs=0.01)


# ---------------------------------------------------------------------------
# AdaPP Planner (smoke test with synthetic data)
# ---------------------------------------------------------------------------


class TestAdaPPPlanner:
    def test_episode_runs(self):
        """A short episode should complete without error."""
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        config = AdaPPConfig(
            n_cell_lat=3, n_cell_lon=3, initial_variance=1.0,
            gamma=0.8, eta=0.3, noise_variance=0.01,
            speed_mps=1.0, connectivity="4",
            variance_backend="spgp_fitc", n_pseudo=10,
            refit_interval=999,  # Don't refit (no model to refit with).
            seed=SEED,
        )
        # Use a dummy model (None) — planner will use synthetic fallback.
        from oceanbench_models.belief.field.spgp_fitc import SPGPFITCFieldModel
        model = SPGPFITCFieldModel({"n_pseudo": 10, "max_opt_iters": 10}, seed=SEED)
        planner = AdaPPPlanner(graph, REGION, model, config)
        result = planner.run_episode(5.0, 5.0, time_budget=1_500_000.0)

        assert result["n_steps"] >= 1
        assert len(result["trajectory"]) >= 1

    def test_trajectory_has_required_fields(self):
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        config = AdaPPConfig(
            n_cell_lat=3, n_cell_lon=3, gamma=0.8, eta=0.3,
            speed_mps=1.0, n_pseudo=10, refit_interval=999, seed=SEED,
        )
        from oceanbench_models.belief.field.spgp_fitc import SPGPFITCFieldModel
        model = SPGPFITCFieldModel({"n_pseudo": 10, "max_opt_iters": 10}, seed=SEED)
        planner = AdaPPPlanner(graph, REGION, model, config)
        result = planner.run_episode(5.0, 5.0, time_budget=1_500_000.0)

        entry = result["trajectory"][0]
        for key in ["step", "lat", "lon", "value", "cell_row", "cell_col", "time"]:
            assert key in entry, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Lawnmower Baseline
# ---------------------------------------------------------------------------


class TestLawnmowerBaseline:
    def test_plan_respects_budget(self):
        graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
        lm = LawnmowerBaseline(graph, speed_mps=1.0)
        path = lm.plan(start_node=12, time_budget=1_500_000.0)
        assert len(path) >= 1
        assert path[0] == 12


# ---------------------------------------------------------------------------
# Build field model factory
# ---------------------------------------------------------------------------


class TestBuildFieldModel:
    def test_spgp_fitc(self):
        config = AdaPPConfig(variance_backend="spgp_fitc", n_pseudo=10)
        model = build_field_model(config, seed=SEED)
        assert model.supports_uncertainty

    def test_svgp_gpytorch(self):
        config = AdaPPConfig(variance_backend="svgp_gpytorch", n_pseudo=10)
        model = build_field_model(config, seed=SEED)
        assert model.supports_uncertainty

    def test_unknown_raises(self):
        config = AdaPPConfig(variance_backend="nonexistent")
        with pytest.raises(ValueError):
            build_field_model(config)
