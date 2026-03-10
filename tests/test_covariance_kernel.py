from __future__ import annotations

import numpy as np

from oceanbench_models.belief.field.kernel_st import KernelSTCovariance


def test_kernel_covariance_symmetry_and_diagonal():
    cov = KernelSTCovariance(config={"lengthscale_space": 1.0, "lengthscale_time": 1.0, "variance": 2.0})

    # Simple 2D features [lat, lon]; no explicit time dimension needed here.
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    K = cov.cov_block(X, X)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)
    diag = np.diag(K)
    assert np.all(diag > 0.0)

    diag_direct = cov.diag_cov(X)
    assert np.allclose(diag, diag_direct, atol=1e-6)

