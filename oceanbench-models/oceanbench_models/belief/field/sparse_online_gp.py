"""
Sparse Online Gaussian Process for scalable, incremental field estimation.

Uses a fixed-size sliding window of "inducing" points (the most recent
observations) and refits the GP on that subset at each update. This keeps
memory and compute bounded while supporting true incremental updates.
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
class SparseOnlineGPConfig:
    """
    Configuration for the sparse online GP.

    Parameters
    ----------
    max_points:
        Maximum number of observations to retain (sliding window size).
        When more observations are added via update(), the oldest are dropped.
    lengthscale, variance, noise, jitter:
        Same as full GP. Refit uses these on the current window.
    include_time, include_depth:
        Whether to include time/depth in the feature space.
    """

    max_points: int = 500
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    jitter: float = 1e-8


class SparseOnlineGPFieldModel(FieldBeliefModel):
    """
    Sparse online GP that maintains a bounded set of inducing points.

    Strategy: keep the last `max_points` observations in a buffer. On each
    update(), append new observations and drop the oldest if over the limit,
    then refit the GP on the current buffer. This gives O(max_points^3) per
    update and O(max_points) memory, suitable for long missions where full GP
    would be prohibitive.

    Uncertainty is supported. supports_online_update is True.
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
        self._cfg = SparseOnlineGPConfig(
            max_points=int(cfg.get("max_points", 500)),
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            jitter=float(cfg.get("jitter", 1e-8)),
        )
        self._X: Optional[ArrayLike] = None
        self._y: Optional[ArrayLike] = None
        self._cho: Optional[tuple[ArrayLike, bool]] = None
        self._alpha: Optional[ArrayLike] = None
        self._variable: Optional[str] = None

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return True

    def _refit(self, X: ArrayLike, y: ArrayLike) -> None:
        """Fit GP on current (X, y) and cache Cholesky and alpha."""
        K = rbf_kernel(
            X, X,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        )
        n = K.shape[0]
        K[np.diag_indices(n)] += self._cfg.noise + self._cfg.jitter
        cho = cho_factor(K, lower=True, overwrite_a=False, check_finite=False)
        alpha = cho_solve(cho, y, check_finite=False)
        self._cho = cho
        self._alpha = alpha

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
        n = X.shape[0]
        if n > self._cfg.max_points:
            # Keep the most recent max_points (last indices).
            start = n - self._cfg.max_points
            X = X[start:].copy()
            y = y[start:].copy()
        self._X = X
        self._y = y
        self._variable = observations.variable
        self._refit(self._X, self._y)
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        if not self.is_fitted or self._X is None or self._y is None:
            raise RuntimeError("Model must be fitted before update.")
        X_new = observations.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        y_new = observations.values.astype(float)
        X = np.vstack([self._X, X_new])
        y = np.concatenate([self._y, y_new])
        if X.shape[0] > self._cfg.max_points:
            start = X.shape[0] - self._cfg.max_points
            X = X[start:].copy()
            y = y[start:].copy()
        self._X = X
        self._y = y
        self._refit(self._X, self._y)

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._X is None or self._cho is None or self._alpha is None:
            raise RuntimeError("Model must be fitted before prediction.")
        Xq = query_points.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        K_xs_x = rbf_kernel(
            Xq, self._X,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        )
        mean = K_xs_x @ self._alpha
        K_xs_x_T = K_xs_x.T
        v = cho_solve(self._cho, K_xs_x_T, check_finite=False)
        K_ss_diag = np.full(Xq.shape[0], self._cfg.variance, dtype=float)
        var = K_ss_diag - np.einsum("ij,ij->j", K_xs_x_T, v)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return FieldPrediction(mean=mean.astype(float), std=std.astype(float), metadata={})

    def reset(self) -> None:
        super().reset()
        self._X = None
        self._y = None
        self._cho = None
        self._alpha = None
        self._variable = None
