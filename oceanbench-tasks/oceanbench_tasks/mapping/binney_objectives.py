from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_models.belief.field.covariance_backends import (
    ArrayLike,
    CovarianceBackend,
)


def _stable_logdet(K: ArrayLike, jitter: float = 1e-6) -> float:
    """Compute log(det(K)) in a numerically stable way."""
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    K = K + jitter * np.eye(n, dtype=float)
    sign, logdet = np.linalg.slogdet(K)
    if sign <= 0:
        # Fall back to a small value to avoid NaNs; this should be rare if
        # covariance construction is well-conditioned.
        return float(logdet)
    return float(logdet)


class BinneyObjective(ABC):
    """
    Abstract base class for Binney-style information objectives.

    Implementations operate on a fixed evaluation set Y (typically a dense
    evaluation grid) and use a covariance backend to compute prior / posterior
    covariances.
    """

    def __init__(
        self,
        covariance: CovarianceBackend,
        eval_features: ArrayLike,
        *,
        noise_var: float,
    ) -> None:
        self._cov = covariance
        self._Y = np.asarray(eval_features, dtype=float)
        self._noise_var = float(noise_var)

        # Precompute prior covariance blocks over Y once.
        self._Sigma_YY = self._cov.cov_block(self._Y, self._Y)
        self._trace_prior = float(np.trace(self._Sigma_YY))
        self._logdet_prior = _stable_logdet(self._Sigma_YY)

    @abstractmethod
    def value(self, A_features: ArrayLike) -> float:
        """
        Objective value f(A) for a set of measurement items A.

        Parameters
        ----------
        A_features:
            Feature matrix of shape (n_A, D) for the measurements in A.
        """

    def marginal_gain(self, X_features: ArrayLike, S_features: ArrayLike) -> float:
        """
        Marginal gain f_X(S) = f(X ∪ S) − f(X).
        """
        X_features = np.asarray(X_features, dtype=float)
        S_features = np.asarray(S_features, dtype=float)
        if X_features.size == 0:
            return self.value(S_features)
        if S_features.size == 0:
            return 0.0
        UX = np.concatenate([X_features, S_features], axis=0)
        return self.value(UX) - self.value(X_features)


class EMSEObjective(BinneyObjective):
    """
    EMSE / variance reduction objective:

        f(A) = tr(Σ_YY) − tr(Σ_YY|A)
    """

    def value(self, A_features: ArrayLike) -> float:
        A = np.asarray(A_features, dtype=float)
        if A.size == 0:
            return 0.0

        # Covariance blocks
        Sigma_AA = self._cov.cov_block(A, A)
        Sigma_YA = self._cov.cov_block(self._Y, A)
        Sigma_AY = Sigma_YA.T

        # Posterior covariance Σ_YY|A = Σ_YY − Σ_YA (Σ_AA + σ^2 I)^−1 Σ_AY
        nA = Sigma_AA.shape[0]
        Sigma_AA_noise = Sigma_AA + self._noise_var * np.eye(nA, dtype=float)
        try:
            inv_term = np.linalg.solve(Sigma_AA_noise, Sigma_AY)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(Sigma_AA_noise) @ Sigma_AY
        Sigma_post = self._Sigma_YY - Sigma_YA @ inv_term
        tr_post = float(np.trace(Sigma_post))
        return self._trace_prior - tr_post


class EntropyObjective(BinneyObjective):
    """
    Entropy-reduction objective:

        f(A) ∝ log det(Σ_YY) − log det(Σ_YY|A)
    """

    def value(self, A_features: ArrayLike) -> float:
        A = np.asarray(A_features, dtype=float)
        if A.size == 0:
            return 0.0

        Sigma_AA = self._cov.cov_block(A, A)
        Sigma_YA = self._cov.cov_block(self._Y, A)
        Sigma_AY = Sigma_YA.T

        nA = Sigma_AA.shape[0]
        Sigma_AA_noise = Sigma_AA + self._noise_var * np.eye(nA, dtype=float)
        try:
            inv_term = np.linalg.solve(Sigma_AA_noise, Sigma_AY)
        except np.linalg.LinAlgError:
            inv_term = np.linalg.pinv(Sigma_AA_noise) @ Sigma_AY
        Sigma_post = self._Sigma_YY - Sigma_YA @ inv_term
        logdet_post = _stable_logdet(Sigma_post)
        return self._logdet_prior - logdet_post


class MutualInformationObjective(EntropyObjective):
    """
    Mutual information objective I(Y; A).

    For jointly Gaussian variables, I(Y; A) = H(Y) − H(Y | A), which has the
    same computational form as entropy reduction. We therefore reuse the
    EntropyObjective implementation but expose a separate class for clarity.
    """

    # Inherits `value` from EntropyObjective.
    pass


def build_binney_objective(
    config: Mapping[str, Any],
    covariance: CovarianceBackend,
    eval_features: ArrayLike,
) -> BinneyObjective:
    """
    Factory for Binney objectives driven by a config mapping.

    The config is expected to contain (either at the top level or under
    an ``objective`` subsection):

    .. code-block:: yaml

        objective:
          type: emse | entropy | mutual_info
          noise_var: 0.01
    """
    root = config.get("objective", config)
    obj_type = str(root.get("type", "emse")).lower()
    noise_var = float(root.get("noise_var", root.get("measurement_noise_var", 1e-2)))

    if obj_type == "emse":
        return EMSEObjective(covariance=covariance, eval_features=eval_features, noise_var=noise_var)
    if obj_type in {"entropy", "entropy_reduction"}:
        return EntropyObjective(covariance=covariance, eval_features=eval_features, noise_var=noise_var)
    if obj_type in {"mi", "mutual_info", "mutual_information"}:
        return MutualInformationObjective(
            covariance=covariance, eval_features=eval_features, noise_var=noise_var
        )

    raise ValueError(f"Unknown objective.type: {obj_type!r}")

