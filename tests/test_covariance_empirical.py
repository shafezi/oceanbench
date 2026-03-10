from __future__ import annotations

import numpy as np

from oceanbench_core import EvalGrid, QueryPoints, Scenario
from oceanbench_models.belief.field.empirical_cov import EmpiricalCovarianceBackend


def test_empirical_covariance_basic_properties():
    scenario = Scenario(
        name="test",
        variable="temp",
        region={"lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0},
    )
    qp = QueryPoints(lats=[0.0, 0.5, 1.0], lons=[0.0, 0.5, 1.0])
    grid = EvalGrid(query_points=qp, scenario=scenario)

    # Build synthetic snapshots with a simple linear pattern plus noise.
    rng = np.random.default_rng(0)
    n_snapshots = 5
    base = np.linspace(0.0, 1.0, qp.size)
    values = np.stack(
        [base + 0.1 * rng.standard_normal(qp.size) for _ in range(n_snapshots)],
        axis=0,
    )
    # Inject some NaNs to test robustness.
    values[0, 0] = np.nan

    cov = EmpiricalCovarianceBackend(eval_grid=grid, values=values, config={"use_anomalies": True})

    X = np.column_stack([qp.lats, qp.lons])
    K = cov.cov_block(X, X)
    assert K.shape == (qp.size, qp.size)
    assert np.allclose(K, K.T)

    diag = cov.diag_cov(X)
    assert diag.shape == (qp.size,)
    assert np.all(diag >= 0.0)

