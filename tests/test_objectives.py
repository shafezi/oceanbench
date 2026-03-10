from __future__ import annotations

import numpy as np

from oceanbench_models.belief.field.covariance_backends import CovarianceBackend
from oceanbench_tasks.mapping.binney_objectives import (
    EMSEObjective,
    EntropyObjective,
    MutualInformationObjective,
)


class _IdentityCovariance(CovarianceBackend):
    """Simple covariance backend with Σ(S, T) = I when S == T."""

    def cov_block(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        Xa = np.atleast_2d(Xa)
        Xb = np.atleast_2d(Xb)
        if Xa.shape[0] == Xb.shape[0] and np.allclose(Xa, Xb):
            return np.eye(Xa.shape[0], dtype=float)
        return np.zeros((Xa.shape[0], Xb.shape[0]), dtype=float)


def test_emse_non_decreasing_with_more_samples():
    cov = _IdentityCovariance()
    Y = np.zeros((5, 1), dtype=float)
    obj = EMSEObjective(covariance=cov, eval_features=Y, noise_var=0.1)

    A1 = np.zeros((1, 1), dtype=float)
    A2 = np.zeros((2, 1), dtype=float)
    v1 = obj.value(A1)
    v2 = obj.value(A2)
    assert v2 >= v1


def test_entropy_and_mi_stable_on_small_matrices():
    cov = _IdentityCovariance()
    Y = np.zeros((3, 1), dtype=float)
    A = np.zeros((2, 1), dtype=float)

    ent = EntropyObjective(covariance=cov, eval_features=Y, noise_var=0.1)
    mi = MutualInformationObjective(covariance=cov, eval_features=Y, noise_var=0.1)

    v_ent = ent.value(A)
    v_mi = mi.value(A)
    assert np.isfinite(v_ent)
    assert np.isfinite(v_mi)

