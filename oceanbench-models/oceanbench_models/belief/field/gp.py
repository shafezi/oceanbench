from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import parse_gp_hyperparams, rbf_kernel


@dataclass
class GPConfig:
    """
    Configuration for the full Gaussian Process field model.

    Parameters
    ----------
    lengthscale:
        RBF kernel lengthscale (in feature space units).
    variance:
        Kernel signal variance.
    noise:
        Observation noise variance added to the diagonal of K.
    include_time:
        Whether to include time as a feature when available.
    include_depth:
        Whether to include depth as a feature when available.
    jitter:
        Small diagonal jitter for numerical stability in Cholesky.
    """

    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    jitter: float = 1e-8


class GPFieldModel(FieldBeliefModel):
    """
    Standard Gaussian Process regression model for scalar ocean fields.

    This implementation uses an isotropic RBF kernel over the feature space
    defined by latitude, longitude, and optionally time/depth. It provides
    predictive means and standard deviations at arbitrary query points.

    This reference implementation is primarily aimed at moderate training
    sizes where an O(N^3) Cholesky factorization is acceptable.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg_mapping = config or {}
        lengthscale, variance, noise = parse_gp_hyperparams(cfg_mapping)
        self._cfg = GPConfig(
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg_mapping.get("include_time", True)),
            include_depth=bool(cfg_mapping.get("include_depth", True)),
            jitter=float(cfg_mapping.get("jitter", 1e-8)),
        )

        self._X: Optional[ArrayLike] = None
        self._y: Optional[ArrayLike] = None
        self._cho: Optional[tuple[ArrayLike, bool]] = None
        self._alpha: Optional[ArrayLike] = None
        self._variable: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        # This implementation does not maintain low-rank updates of the
        # Cholesky factor; we encourage refitting for now.
        return False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

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

        K = rbf_kernel(
            X,
            X,
            lengthscale=self._cfg.lengthscale,
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
        """
        Full GP model does not implement efficient online updates.

        For clarity, we raise an error here rather than silently refitting.
        Call `fit` again with the desired observation set.
        """

        raise NotImplementedError(
            "GPFieldModel does not support online updates; please refit with "
            "the combined observations."
        )

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._X is None or self._cho is None or self._alpha is None:
            raise RuntimeError("Model must be fitted before prediction.")

        Xq = query_points.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )

        K_xs_x = rbf_kernel(
            Xq,
            self._X,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        )
        mean = K_xs_x @ self._alpha

        # Predictive variance: diag(K_ss - K_sx K_xx^{-1} K_xs)
        K_xs_x_T = K_xs_x.T
        v = cho_solve(self._cho, K_xs_x_T, check_finite=False)
        K_ss_diag = np.full(Xq.shape[0], self._cfg.variance, dtype=float)
        var = K_ss_diag - np.einsum("ij,ij->j", K_xs_x_T, v)

        # Numerical guard: variance must be non-negative.
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        return FieldPrediction(
            mean=mean.astype(float),
            std=std.astype(float),
            metadata={},
        )

    def reset(self) -> None:
        super().reset()
        self._X = None
        self._y = None
        self._cho = None
        self._alpha = None
        self._variable = None

