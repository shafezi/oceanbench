"""
Pseudo-Input (FITC-style) Gaussian Process for scalable field estimation.

Uses a fixed set of M pseudo-inputs; the prior is approximated so that
inference is O(N M^2 + M^3) instead of O(N^3). Pseudo-input locations
are initialized from a subset of training data (e.g. k-means or uniform)
and are not optimized in this baseline implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import parse_gp_hyperparams, rbf_kernel


@dataclass
class PseudoInputGPConfig:
    """
    Configuration for the pseudo-input GP.

    Parameters
    ----------
    n_pseudo:
        Number of pseudo-input (inducing) points.
    lengthscale, variance, noise, jitter:
        Kernel and noise hyperparameters.
    include_time, include_depth:
        Whether to include time/depth in features.
    """

    n_pseudo: int = 100
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    jitter: float = 1e-8


class PseudoInputGPFieldModel(FieldBeliefModel):
    """
    Pseudo-input GP using a fixed inducing set (FITC-style approximation).

    Pseudo-inputs are chosen at fit time by sampling uniformly at random
    from the training inputs (without optimizing their positions). Predictive
    mean and variance use the standard FITC formulas. This model does not
    support efficient online updates; use fit() with the full dataset or
    SparseOnlineGPFieldModel for incremental updates.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = config or {}
        lengthscale, variance, noise = parse_gp_hyperparams(cfg)
        self._cfg = PseudoInputGPConfig(
            n_pseudo=int(cfg.get("n_pseudo", 100)),
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            jitter=float(cfg.get("jitter", 1e-8)),
        )
        self._X: Optional[ArrayLike] = None
        self._y: Optional[ArrayLike] = None
        self._Z: Optional[ArrayLike] = None  # pseudo-inputs
        self._Kzz_cho: Optional[tuple[ArrayLike, bool]] = None
        self._A_cho: Optional[tuple[ArrayLike, bool]] = None  # for FITC prediction variance
        self._alpha: Optional[ArrayLike] = None
        self._Lambda: Optional[ArrayLike] = None  # diag(K_ff - Q_ff) + noise
        self._variable: Optional[str] = None

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return False

    def _select_pseudo_inputs(self, X: ArrayLike) -> ArrayLike:
        """Choose n_pseudo pseudo-inputs from X (random subset)."""
        n = X.shape[0]
        m = min(self._cfg.n_pseudo, n)
        idx = self._rng.choice(n, size=m, replace=False)
        idx = np.sort(idx)
        return X[idx]

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        X = observations.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        y = observations.values.astype(float)
        self._X = X
        self._y = y
        self._variable = observations.variable

        Z = self._select_pseudo_inputs(X)
        self._Z = Z
        M = Z.shape[0]

        # K_zz
        Kzz = rbf_kernel(Z, Z, lengthscale=self._cfg.lengthscale, variance=self._cfg.variance)
        Kzz[np.diag_indices(M)] += self._cfg.jitter
        self._Kzz_cho = cho_factor(Kzz, lower=True, check_finite=False)

        # K_fz (n x M)
        Kfz = rbf_kernel(X, Z, lengthscale=self._cfg.lengthscale, variance=self._cfg.variance)
        # diag(K_ff - Q_ff) + noise, where Q_ff = K_fz K_zz^{-1} K_zf
        # v = K_zz^{-1} K_zf so v is (M, n), diag(K_fz @ v) = diag(Q_ff)
        v = cho_solve(self._Kzz_cho, Kfz.T, check_finite=False)  # (M, n)
        Qff_diag = np.einsum("ij,ji->i", Kfz, v)
        Kff_diag = np.full(X.shape[0], self._cfg.variance, dtype=float)
        Lambda_diag = np.maximum(Kff_diag - Qff_diag + self._cfg.noise, 1e-10)
        self._Lambda = Lambda_diag

        # FITC: (K_zf Lambda^{-1} K_fz + K_zz)^{-1} K_zf Lambda^{-1} y
        Lam_inv_Kfz = Kfz / Lambda_diag[:, None]  # (n, M)
        A = Kzz + Kfz.T @ Lam_inv_Kfz
        A[np.diag_indices(M)] += self._cfg.jitter
        self._A_cho = cho_factor(A, lower=True, check_finite=False)
        b = Kfz.T @ (y / Lambda_diag)
        self._alpha = cho_solve(self._A_cho, b, check_finite=False)
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "PseudoInputGPFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._Z is None or self._Kzz_cho is None or self._alpha is None:
            raise RuntimeError("Model must be fitted before prediction.")
        Xq = query_points.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        K_xz = rbf_kernel(
            Xq, self._Z,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        )
        mean = K_xz @ self._alpha

        # Predictive variance: k_ss - k_sz @ A^{-1} @ k_zs
        k_zs = K_xz.T  # (M, nq)
        v = cho_solve(self._A_cho, k_zs, check_finite=False)
        k_ss = np.full(Xq.shape[0], self._cfg.variance, dtype=float)
        var = k_ss - np.einsum("ij,ij->j", K_xz.T, v)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return FieldPrediction(mean=mean.astype(float), std=std.astype(float), metadata={})

    def reset(self) -> None:
        super().reset()
        self._X = None
        self._y = None
        self._Z = None
        self._Kzz_cho = None
        self._A_cho = None
        self._alpha = None
        self._Lambda = None
        self._variable = None
