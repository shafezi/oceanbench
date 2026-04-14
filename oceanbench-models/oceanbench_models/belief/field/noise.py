from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


NOISE_MODES = ("fixed", "estimate")
NOISE_ESTIMATE_METHODS = ("gp_likelihood", "residual", "cv")


@dataclass
class NoiseConfig:
    """
    Shared noise configuration for field models.
    """

    mode: str = "fixed"
    fixed_sigma2: float = 1e-3
    estimate_method: str = "gp_likelihood"
    min_sigma2: float = 1e-8
    max_sigma2: float = 10.0
    ema_alpha: float = 0.25

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "NoiseConfig":
        cfg = dict(config or {})
        root = dict(cfg.get("noise", cfg))
        mode = str(root.get("mode", "fixed")).lower()
        estimate_method = str(root.get("estimate_method", "gp_likelihood")).lower()
        if mode not in NOISE_MODES:
            raise ValueError(f"Unknown noise mode {mode!r}; expected one of {NOISE_MODES}.")
        if estimate_method not in NOISE_ESTIMATE_METHODS:
            raise ValueError(
                f"Unknown noise.estimate_method {estimate_method!r}; expected one of {NOISE_ESTIMATE_METHODS}."
            )
        return cls(
            mode=mode,
            fixed_sigma2=float(root.get("fixed_sigma2", cfg.get("noise", 1e-3))),
            estimate_method=estimate_method,
            min_sigma2=float(root.get("min_sigma2", 1e-8)),
            max_sigma2=float(root.get("max_sigma2", 10.0)),
            ema_alpha=float(root.get("ema_alpha", 0.25)),
        )


class NoiseManager:
    """
    Maintains observation-noise variance under fixed/estimated modes.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = NoiseConfig.from_mapping(config)
        self._sigma2 = float(self.config.fixed_sigma2)

    @property
    def sigma2(self) -> float:
        return float(self._sigma2)

    def reset(self) -> None:
        self._sigma2 = float(self.config.fixed_sigma2)

    def resolve_sigma2(self) -> float:
        if self.config.mode == "fixed":
            return float(np.clip(self.config.fixed_sigma2, self.config.min_sigma2, self.config.max_sigma2))
        return float(np.clip(self._sigma2, self.config.min_sigma2, self.config.max_sigma2))

    def update_from_gp_likelihood(self, sigma2: float) -> float:
        if self.config.mode != "estimate":
            self._sigma2 = float(self.config.fixed_sigma2)
            return self.resolve_sigma2()
        clipped = float(np.clip(sigma2, self.config.min_sigma2, self.config.max_sigma2))
        self._sigma2 = _ema(self._sigma2, clipped, alpha=self.config.ema_alpha)
        return self.resolve_sigma2()

    def update_from_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        if self.config.mode != "estimate":
            self._sigma2 = float(self.config.fixed_sigma2)
            return self.resolve_sigma2()
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if y_true.size == 0:
            return self.resolve_sigma2()
        residuals = y_true - y_pred
        # Use mean squared residual instead of nanvar: nanvar of a single
        # element is always 0, which would drive the noise estimate to
        # min_sigma2 when called with one observation at a time (e.g. SOGP
        # online updates).  Mean squared residual is the correct unbiased
        # estimator of noise variance for zero-mean residuals.
        var = float(np.nanmean(residuals ** 2))
        var = float(np.clip(var, self.config.min_sigma2, self.config.max_sigma2))
        self._sigma2 = _ema(self._sigma2, var, alpha=self.config.ema_alpha)
        return self.resolve_sigma2()

    def update_from_cv(self, cv_error_values: np.ndarray) -> float:
        """
        Update from CV-like residual statistics.

        This intentionally remains lightweight: callers pass squared/absolute
        prediction residuals from a CV split, and we convert them to variance.
        """
        if self.config.mode != "estimate":
            self._sigma2 = float(self.config.fixed_sigma2)
            return self.resolve_sigma2()
        arr = np.asarray(cv_error_values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return self.resolve_sigma2()
        var = float(np.nanmean(arr**2))
        var = float(np.clip(var, self.config.min_sigma2, self.config.max_sigma2))
        self._sigma2 = _ema(self._sigma2, var, alpha=self.config.ema_alpha)
        return self.resolve_sigma2()

    def update(
        self,
        *,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        gp_likelihood_sigma2: Optional[float] = None,
        cv_error_values: Optional[np.ndarray] = None,
    ) -> float:
        method = self.config.estimate_method
        if self.config.mode == "fixed":
            self._sigma2 = float(self.config.fixed_sigma2)
            return self.resolve_sigma2()
        if method == "gp_likelihood":
            if gp_likelihood_sigma2 is not None:
                return self.update_from_gp_likelihood(gp_likelihood_sigma2)
            return self.resolve_sigma2()
        if method == "residual":
            if y_true is not None and y_pred is not None:
                return self.update_from_residuals(y_true, y_pred)
            return self.resolve_sigma2()
        if method == "cv":
            if cv_error_values is not None:
                return self.update_from_cv(cv_error_values)
            return self.resolve_sigma2()
        raise ValueError(f"Unsupported noise estimate method {method!r}.")


def _ema(previous: float, current: float, *, alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * float(previous) + alpha * float(current)

