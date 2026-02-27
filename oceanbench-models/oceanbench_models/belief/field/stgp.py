"""
Spatio-Temporal Gaussian Process for field estimation with explicit time.

Uses a separable space-time RBF kernel: k_space * k_time. Time is included
as a feature (e.g. seconds since epoch) so that predictions can vary
smoothly in time as well as in space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import (
    parse_gp_hyperparams,
    rbf_separable_space_time_kernel,
)


@dataclass
class STGPConfig:
    """
    Configuration for the spatio-temporal GP.

    Parameters
    ----------
    lengthscale_space:
        RBF lengthscale for (lat, lon).
    lengthscale_time:
        RBF lengthscale for time (in same units as time feature, e.g. seconds).
    variance, noise, jitter:
        Kernel and noise.
    include_depth:
        If True, depth is appended after time in the feature vector;
        the kernel treats it as part of "space" (same lengthscale_space).
    """

    lengthscale_space: float = 1.0
    lengthscale_time: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_depth: bool = True
    jitter: float = 1e-8


class STGPFieldModel(FieldBeliefModel):
    """
    Spatio-temporal GP with separable space-time kernel.

    Expects observations and query points to include time (and optionally
    depth). Time must be present in ObservationBatch/QueryPoints for this
    model; otherwise use the standard GP. Uncertainty is supported.
    Online updates are not implemented (refit for new data).
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = config or {}
        _, variance, noise = parse_gp_hyperparams(cfg)
        self._cfg = STGPConfig(
            lengthscale_space=float(cfg.get("lengthscale_space", 1.0)),
            lengthscale_time=float(cfg.get("lengthscale_time", 1.0)),
            variance=variance,
            noise=noise,
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
        return False

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        # Require at least [lat, lon, time]
        X = observations.as_features(include_time=True, include_depth=self._cfg.include_depth)
        if X.shape[1] < 3:
            raise ValueError(
                "STGPFieldModel requires time in observations; "
                "ensure ObservationBatch has times."
            )
        y = observations.values.astype(float)
        self._X = X
        self._y = y
        self._variable = observations.variable

        K = rbf_separable_space_time_kernel(
            X, X,
            lengthscale_space=self._cfg.lengthscale_space,
            lengthscale_time=self._cfg.lengthscale_time,
            variance=self._cfg.variance,
        )
        n = K.shape[0]
        K[np.diag_indices(n)] += self._cfg.noise + self._cfg.jitter
        cho = cho_factor(K, lower=True, overwrite_a=False, check_finite=False)
        alpha = cho_solve(cho, y, check_finite=False)
        self._cho = cho
        self._alpha = alpha
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "STGPFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._X is None or self._cho is None or self._alpha is None:
            raise RuntimeError("Model must be fitted before prediction.")
        Xq = query_points.as_features(include_time=True, include_depth=self._cfg.include_depth)
        if Xq.shape[1] < 3:
            raise ValueError(
                "STGPFieldModel requires time in query points; "
                "ensure QueryPoints has times."
            )
        K_xs_x = rbf_separable_space_time_kernel(
            Xq, self._X,
            lengthscale_space=self._cfg.lengthscale_space,
            lengthscale_time=self._cfg.lengthscale_time,
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
