from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.optimize import minimize


HYPERPARAM_MODES = ("fixed", "periodic", "rho_trigger", "continuous")
ON_UPDATE_MODES = ("replan_immediately", "keep_current_plan")


@dataclass
class HyperparameterScheduleConfig:
    """
    Schedule controlling when model hyperparameters are re-estimated.
    """

    mode: str = "rho_trigger"
    rho0: float = 0.6
    period: int = 200
    on_update: str = "replan_immediately"
    min_updates_apart: int = 1

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "HyperparameterScheduleConfig":
        cfg = dict(config or {})
        root = dict(cfg.get("hyperparams", cfg))
        mode = str(root.get("mode", "rho_trigger")).lower()
        on_update = str(root.get("on_update", "replan_immediately")).lower()
        if mode not in HYPERPARAM_MODES:
            raise ValueError(
                f"Unknown hyperparams.mode {mode!r}; expected one of {HYPERPARAM_MODES}."
            )
        if on_update not in ON_UPDATE_MODES:
            raise ValueError(
                f"Unknown hyperparams.on_update {on_update!r}; expected one of {ON_UPDATE_MODES}."
            )
        return cls(
            mode=mode,
            rho0=float(root.get("rho0", 0.6)),
            period=max(1, int(root.get("period", 200))),
            on_update=on_update,
            min_updates_apart=max(1, int(root.get("min_updates_apart", 1))),
        )


class HyperparameterScheduler:
    """
    Stateful schedule evaluator for online loops.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = HyperparameterScheduleConfig.from_mapping(config)
        self._last_update_step = -10**9

    @property
    def on_update(self) -> str:
        return self.config.on_update

    def should_update(
        self,
        *,
        step: int,
        rho: float,
        force: bool = False,
    ) -> bool:
        if force:
            return True
        if (step - self._last_update_step) < self.config.min_updates_apart:
            return False

        mode = self.config.mode
        if mode == "fixed":
            return False
        if mode == "continuous":
            return True
        if mode == "periodic":
            return step > 0 and (step % self.config.period == 0)
        if mode == "rho_trigger":
            return float(rho) >= float(self.config.rho0)
        raise ValueError(f"Unsupported hyperparameter mode {mode!r}.")

    def mark_updated(self, *, step: int) -> None:
        self._last_update_step = int(step)


@dataclass
class RBFHyperparameters:
    """
    ARD-RBF hyperparameter bundle.
    """

    lengthscales: np.ndarray
    variance: float
    noise: float

    def as_dict(self) -> dict[str, Any]:
        ls = np.asarray(self.lengthscales, dtype=float).ravel()
        if ls.size == 1:
            lengthscale: float | list[float] = float(ls[0])
        else:
            lengthscale = [float(x) for x in ls]
        return {
            "lengthscale": lengthscale,
            "variance": float(self.variance),
            "noise": float(self.noise),
        }


def fit_rbf_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    *,
    initial_lengthscale: float | np.ndarray = 1.0,
    initial_variance: float = 1.0,
    initial_noise: float = 1e-3,
    method: str = "mll",
    max_iters: int = 100,
    jitter: float = 1e-8,
) -> RBFHyperparameters:
    """
    Fit ARD-RBF hyperparameters with lightweight MLE / LOO-CV objective.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension.")
    if X.shape[0] < 2:
        return RBFHyperparameters(
            lengthscales=np.atleast_1d(float(np.asarray(initial_lengthscale).ravel()[0])),
            variance=float(initial_variance),
            noise=float(initial_noise),
        )

    # Centre targets so that the zero-mean GP MLL/LOO-CV objectives are
    # correct.  Without this, raw data with nonzero mean (e.g. ocean
    # temperature ~25°C) inflates the learned kernel variance.
    y_mean = float(np.mean(y))
    y = y - y_mean

    d = int(X.shape[1])
    ls0 = np.asarray(initial_lengthscale, dtype=float).ravel()
    if ls0.size == 1:
        ls0 = np.repeat(float(ls0[0]), d)
    elif ls0.size != d:
        raise ValueError("initial_lengthscale must be scalar or match X.shape[1].")

    theta0 = np.concatenate(
        [
            np.log(np.maximum(ls0, 1e-6)),
            np.array(
                [
                    np.log(max(float(initial_variance), 1e-8)),
                    np.log(max(float(initial_noise), 1e-8)),
                ],
                dtype=float,
            ),
        ]
    )

    method = str(method).lower()
    if method not in {"mll", "loo_cv"}:
        raise ValueError("method must be one of {'mll', 'loo_cv'}.")

    def objective(theta: np.ndarray) -> float:
        ls = np.exp(theta[:d])
        var = float(np.exp(theta[d]))
        noise = float(np.exp(theta[d + 1]))
        K = _rbf_kernel_ard(X, X, lengthscales=ls, variance=var)
        K = K + (noise + jitter) * np.eye(X.shape[0], dtype=float)
        if method == "mll":
            return _neg_log_marginal_likelihood(K, y)
        return _neg_log_loo_cv_likelihood(K, y)

    bounds = [(-10.0, 10.0)] * d + [(-12.0, 12.0), (-16.0, 4.0)]
    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iters)},
    )
    theta = np.asarray(res.x if res.success else theta0, dtype=float)
    ls = np.exp(theta[:d])
    var = float(np.exp(theta[d]))
    noise = float(np.exp(theta[d + 1]))
    return RBFHyperparameters(lengthscales=ls, variance=var, noise=noise)


def _rbf_kernel_ard(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    lengthscales: np.ndarray,
    variance: float,
) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    ls = np.asarray(lengthscales, dtype=float).ravel()
    if ls.size == 1:
        ls = np.repeat(float(ls[0]), X1.shape[1])
    X1s = X1 / ls
    X2s = X2 / ls
    d2 = np.sum((X1s[:, None, :] - X2s[None, :, :]) ** 2, axis=-1)
    return float(variance) * np.exp(-0.5 * d2)


def _neg_log_marginal_likelihood(K: np.ndarray, y: np.ndarray) -> float:
    n = int(y.shape[0])
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        return float("inf")


def _neg_log_loo_cv_likelihood(K: np.ndarray, y: np.ndarray) -> float:
    """
    LOO-CV objective using Eq.(5)-style identities from the paper.
    Uses Cholesky-based solve instead of direct inverse for stability.
    """
    try:
        L = np.linalg.cholesky(K)
        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0], dtype=float)))
    except np.linalg.LinAlgError:
        return float("inf")

    alpha = K_inv @ y
    diag = np.diag(K_inv)
    if np.any(diag <= 0.0):
        return float("inf")
    mu_loo = y - alpha / diag
    var_loo = 1.0 / diag
    ll = -0.5 * np.log(2.0 * np.pi * var_loo) - 0.5 * ((y - mu_loo) ** 2) / var_loo
    return float(-np.sum(ll))

