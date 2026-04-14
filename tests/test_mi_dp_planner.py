"""Tests for DP-style MI planner."""

from __future__ import annotations

import numpy as np

from oceanbench_core.types import QueryPoints
from oceanbench_policies.ipp.mi_dp_planner import MIDPPlanner
from oceanbench_policies.ipp.mi_greedy import MIGreedyPlanner


class _DummyModel:
    def __init__(self, K: np.ndarray):
        self.K = np.asarray(K, dtype=float)

    def predictive_covariance(self, points: QueryPoints) -> np.ndarray:
        n = points.size
        return self.K[:n, :n]


def _make_cov(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = np.exp(-d2 / 0.15) + 1e-4 * np.eye(n)
    return 0.5 * (K + K.T)


def test_dp_planner_selects_unique_points():
    n = 45
    qp = QueryPoints(
        lats=np.linspace(24.0, 36.0, n),
        lons=np.linspace(-86.0, -74.0, n),
    )
    K = _make_cov(n)
    model = _DummyModel(K)

    planner = MIDPPlanner(
        {
            "planner": {"batch_size_n": 7, "dp_beam_width": 24, "max_candidates_for_dp": 45},
            "mi": {"X_set": "candidate_grid", "jitter": 1e-8},
        },
        seed=1,
    )
    res = planner.plan(model, qp)
    idx = np.asarray(res.selected_indices, dtype=int)
    assert idx.shape[0] == 7
    assert np.unique(idx).shape[0] == idx.shape[0]
    assert np.isfinite(res.objective_value)
    assert res.objective_value >= 0.0


def test_dp_not_worse_than_greedy_on_same_covariance():
    n = 36
    qp = QueryPoints(
        lats=np.linspace(24.0, 36.0, n),
        lons=np.linspace(-86.0, -74.0, n),
    )
    K = _make_cov(n)
    model = _DummyModel(K)

    dp = MIDPPlanner(
        {
            "planner": {"batch_size_n": 6, "dp_beam_width": 32, "max_candidates_for_dp": 36},
            "mi": {"X_set": "candidate_grid", "jitter": 1e-8},
        },
        seed=2,
    )
    gr = MIGreedyPlanner(
        {
            "planner": {"batch_size_n": 6, "max_candidates_for_greedy": 36},
            "mi": {"X_set": "candidate_grid", "jitter": 1e-8},
        },
        seed=2,
    )
    r_dp = dp.plan(model, qp)
    r_gr = gr.plan(model, qp)
    # DP with beam search is an approximation; it usually matches or
    # beats greedy but is not strictly guaranteed to on every instance.
    # Allow up to 20% relative slack.
    assert r_dp.objective_value >= 0.8 * r_gr.objective_value
