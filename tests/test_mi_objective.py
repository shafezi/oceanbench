"""MI objective and numerical stability tests."""

from __future__ import annotations

import numpy as np

from oceanbench_core.types import QueryPoints
from oceanbench_policies.ipp.mi_dp_planner import (
    compute_mutual_information_from_covariance,
    predictive_covariance_from_model,
    stable_cholesky,
)
from oceanbench_policies.ipp.mi_greedy import MIGreedyPlanner


class _DummyModel:
    def __init__(self, covariance: np.ndarray):
        self._cov = np.asarray(covariance, dtype=float)

    def predictive_covariance(self, points: QueryPoints) -> np.ndarray:
        n = points.size
        return self._cov[:n, :n]

    def predict(self, points: QueryPoints):
        class _P:
            pass

        p = _P()
        p.mean = np.zeros(points.size, dtype=float)
        p.std = np.sqrt(np.maximum(np.diag(self._cov[: points.size, : points.size]), 1e-12))
        return p


def _toy_covariance(n: int = 25) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    d2 = (xs - xs.T) ** 2
    K = np.exp(-d2 / 0.2) + 1e-6 * np.eye(n)
    return 0.5 * (K + K.T)


def test_mi_computation_finite_stable():
    K = _toy_covariance(20)
    # Near-singular perturbation.
    K[:, 0] = K[:, 1]
    K[0, :] = K[1, :]
    L, used_jitter = stable_cholesky(K, jitter=1e-10, max_tries=8)
    assert L.shape == (20, 20)
    assert used_jitter >= 1e-10
    mi = compute_mutual_information_from_covariance(K, selected_indices=[0, 2, 4], jitter=1e-10)
    assert np.isfinite(mi)
    assert mi >= 0.0


def test_greedy_mi_beats_random_on_toy():
    n = 35
    K = _toy_covariance(n)
    model = _DummyModel(K)
    qp = QueryPoints(
        lats=np.linspace(24.0, 36.0, n),
        lons=np.linspace(-86.0, -74.0, n),
    )
    planner = MIGreedyPlanner(
        {
            "planner": {"batch_size_n": 6},
            "mi": {"X_set": "candidate_grid", "jitter": 1e-8},
        },
        seed=0,
    )
    res = planner.plan(model, qp)
    greedy_mi = float(res.objective_value)

    rng = np.random.default_rng(0)
    random_mi_vals = []
    for _ in range(25):
        idx = rng.choice(n, size=6, replace=False)
        mi = compute_mutual_information_from_covariance(K, selected_indices=idx, jitter=1e-8)
        random_mi_vals.append(mi)
    random_median = float(np.median(random_mi_vals))
    assert greedy_mi >= random_median

    cov = predictive_covariance_from_model(model, qp, jitter=1e-8)
    assert cov.shape == (n, n)
