from __future__ import annotations

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.sampling import MeasurementItem
from oceanbench_models.belief.field.covariance_backends import CovarianceBackend
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective
from oceanbench_policies.ipp.grg import GRGConfig, GRGPlanner


class _DummyCovariance(CovarianceBackend):
    def cov_block(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        Xa = np.atleast_2d(Xa)
        Xb = np.atleast_2d(Xb)
        return Xa @ Xb.T


class _DummyObjective(BinneyObjective):
    """Simple objective: value is the number of unique samples."""

    def __init__(self) -> None:
        cov = _DummyCovariance()
        Y = np.zeros((1, 2), dtype=float)
        super().__init__(covariance=cov, eval_features=Y, noise_var=0.0)

    def value(self, A_features: np.ndarray) -> float:
        return float(A_features.shape[0])


def test_grg_returns_feasible_path():
    region = {"lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0}
    g = WaypointGraph.grid(region, n_lat=2, n_lon=2, speed_mps=1.0, seed=0)
    obj = _DummyObjective()
    cfg = GRGConfig(depth=1, n_splits=3)
    planner = GRGPlanner(graph=g, objective=obj, sampling_config={"edge_spacing_m": 0.0, "include_nodes": True}, config=cfg)

    tau = np.datetime64("2014-01-01T00:00:00", "ns")
    # Budget must allow at least one feasible split (e.g. B1 >= one-edge time).
    B = 4.0 * g.shortest_time(0, 3)
    path, samples, gain = planner.plan(0, 3, B, tau, X_items=None)

    assert path is not None
    assert path[0] == 0 and path[-1] == 3
    assert gain >= 0.0

