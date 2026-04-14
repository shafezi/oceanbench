"""Sparse Pseudo-inputs Gaussian Process (SPGP) using FITC approximation.

Paper-faithful implementation of Snelson & Ghahramani (2006):
  "Sparse Gaussian Processes using Pseudo-inputs"

as used in Mishra et al. (2018) "Online Informative Path Planning using
Sparse Gaussian Processes" (Eqs. 3-5).

Key equations:
  Predictive mean:  μ* = k*ᵀ Σ K_MN (Λ + σ²I)⁻¹ y          (Eq. 4)
  Predictive var:   σ²* = K** - k*ᵀ (K_M⁻¹ - Σ) k* + σ²    (Eq. 5)

where Σ = (K_M + K_MN Λ⁻¹_σ K_NM)⁻¹ and Λ_σ = Λ + σ²I,
      Λ = diag(K_NN - Q_NN),  Q_NN = K_NM K_M⁻¹ K_MN.

Both kernel hyperparameters AND pseudo-input locations Z are jointly
optimized by maximizing the FITC marginal log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.optimize import minimize

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import (
    FeatureScaler,
    observation_batch_to_numpy,
    query_points_to_numpy,
)

_JITTER = 1e-6


# ---------------------------------------------------------------------------
# ARD-RBF kernel  (Eq. 11 of Mishra et al.)
# ---------------------------------------------------------------------------

def _ard_rbf(X1: np.ndarray, X2: np.ndarray,
             log_alpha: float, log_lengthscales: np.ndarray) -> np.ndarray:
    """ARD-RBF kernel: K(x,x') = α exp(-0.5 Σ_l b_l (x_l - x'_l)²).

    Parameters are in log-space for unconstrained optimisation.
    Returns shape (N1, N2).
    """
    alpha = np.exp(np.clip(log_alpha, -20, 20))
    ls = np.exp(np.clip(log_lengthscales, -20, 20))  # (D,)
    # Scaled difference.
    X1s = X1 / ls[None, :]  # (N1, D)
    X2s = X2 / ls[None, :]  # (N2, D)
    dist_sq = (
        np.sum(X1s ** 2, axis=1, keepdims=True)
        - 2.0 * X1s @ X2s.T
        + np.sum(X2s ** 2, axis=1, keepdims=False)[None, :]
    )
    dist_sq = np.maximum(dist_sq, 0.0)
    return alpha * np.exp(-0.5 * dist_sq)


def _ard_rbf_diag(X: np.ndarray, log_alpha: float) -> np.ndarray:
    """Diagonal of ARD-RBF kernel: k(x,x) = α for all x."""
    return np.full(X.shape[0], np.exp(log_alpha))


# ---------------------------------------------------------------------------
# FITC log marginal likelihood
# ---------------------------------------------------------------------------

def _fitc_nll(params: np.ndarray, X: np.ndarray, y: np.ndarray,
              M: int, D: int) -> float:
    """Negative log marginal likelihood under FITC approximation.

    params layout: [log_alpha, log_ls (D), log_sigma2, Z (M*D)]
    """
    N = X.shape[0]

    log_alpha = params[0]
    log_ls = params[1:1 + D]
    log_sigma2 = params[1 + D]
    Z = params[2 + D:].reshape(M, D)

    sigma2 = np.exp(log_sigma2)
    alpha = np.exp(log_alpha)

    # K_M = K(Z, Z) + jitter
    K_M = _ard_rbf(Z, Z, log_alpha, log_ls) + _JITTER * np.eye(M)
    K_MN = _ard_rbf(Z, X, log_alpha, log_ls)  # (M, N)
    K_diag = _ard_rbf_diag(X, log_alpha)       # (N,)

    # Cholesky of K_M.
    try:
        L_M = np.linalg.cholesky(K_M)
    except np.linalg.LinAlgError:
        return 1e10

    # V = L_M⁻¹ K_MN, shape (M, N)
    V = np.linalg.solve(L_M, K_MN)

    # Λ = diag(K_NN - Q_NN) where Q_NN = K_NMᵀ K_M⁻¹ K_MN
    Q_diag = np.sum(V ** 2, axis=0)  # (N,)
    Lambda = np.maximum(K_diag - Q_diag, _JITTER)

    # Λ_σ = Λ + σ²
    Lambda_sigma = Lambda + sigma2

    # B = I + V diag(1/Λ_σ) Vᵀ, shape (M, M)
    V_scaled = V / Lambda_sigma[None, :]  # (M, N)
    B = np.eye(M) + V_scaled @ V.T

    try:
        L_B = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        return 1e10

    # β = L_B⁻¹ V diag(1/Λ_σ) y
    r = V_scaled @ y  # (M,)
    beta = np.linalg.solve(L_B, r)

    # NLL = 0.5 [N log(2π) + log|B| + log|Λ_σ| + yᵀ Λ_σ⁻¹ y - βᵀ β]
    log_det_B = 2.0 * np.sum(np.log(np.diag(L_B)))
    log_det_Ls = np.sum(np.log(Lambda_sigma))
    data_fit = np.sum(y ** 2 / Lambda_sigma) - np.sum(beta ** 2)

    nll = 0.5 * (N * np.log(2.0 * np.pi) + log_det_B + log_det_Ls + data_fit)

    if not np.isfinite(nll):
        return 1e10
    return float(nll)


# ---------------------------------------------------------------------------
# FITC prediction
# ---------------------------------------------------------------------------

def _fitc_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, Z: np.ndarray,
    log_alpha: float, log_ls: np.ndarray, log_sigma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """FITC predictive mean and variance (Eqs. 4-5)."""
    M = Z.shape[0]
    sigma2 = np.exp(log_sigma2)

    K_M = _ard_rbf(Z, Z, log_alpha, log_ls) + _JITTER * np.eye(M)
    K_MN = _ard_rbf(Z, X_train, log_alpha, log_ls)
    K_diag = _ard_rbf_diag(X_train, log_alpha)

    L_M = np.linalg.cholesky(K_M)
    V = np.linalg.solve(L_M, K_MN)  # L_M⁻¹ K_MN

    Q_diag = np.sum(V ** 2, axis=0)
    Lambda = np.maximum(K_diag - Q_diag, _JITTER)
    Lambda_sigma = Lambda + sigma2

    V_scaled = V / Lambda_sigma[None, :]
    B = np.eye(M) + V_scaled @ V.T
    L_B = np.linalg.cholesky(B)

    # Σ = (K_M + K_MN Λ_σ⁻¹ K_NM)⁻¹ = L_M⁻ᵀ L_B⁻ᵀ L_B⁻¹ L_M⁻¹
    # For prediction we need:
    #   μ* = k*ᵀ Σ K_MN Λ_σ⁻¹ y
    #   σ²* = K** - k*ᵀ (K_M⁻¹ - Σ) k* + σ²

    # Precompute α_vec = Σ K_MN Λ_σ⁻¹ y
    r = K_MN @ (y_train / Lambda_sigma)  # (M,)
    # Σ r = K_M⁻¹ B⁻¹ r  (using Woodbury)
    # Actually: Σ = (L_M L_M^T + K_MN Λ_σ⁻¹ K_NM)⁻¹
    # So Σ r: solve (K_M + K_MN Λ_σ⁻¹ K_NM) α = r
    # = L_M (L_M^T) + V_scaled V^T -> B in L_M⁻¹ space
    # Simplify: L_M^T α = L_B⁻ᵀ L_B⁻¹ L_M⁻¹ r
    tmp1 = np.linalg.solve(L_M, r)
    tmp2 = np.linalg.solve(L_B, tmp1)
    tmp3 = np.linalg.solve(L_B.T, tmp2)
    alpha_vec = np.linalg.solve(L_M.T, tmp3)

    # Test points.
    k_star = _ard_rbf(Z, X_test, log_alpha, log_ls)  # (M, N*)
    k_diag = _ard_rbf_diag(X_test, log_alpha)          # (N*,)

    # Mean: μ* = k*ᵀ α_vec
    mu = k_star.T @ alpha_vec  # (N*,)

    # Variance: σ²* = K** - k*ᵀ (K_M⁻¹ - Σ) k* + σ²
    # = K** - k*ᵀ K_M⁻¹ k* + k*ᵀ Σ k* + σ²
    # Term 1: k*ᵀ K_M⁻¹ k*
    v1 = np.linalg.solve(L_M, k_star)   # (M, N*)
    term_prior = np.sum(v1 ** 2, axis=0)  # k*ᵀ K_M⁻¹ k*

    # Term 2: k*ᵀ Σ k* = ||L_B⁻¹ L_M⁻¹ k*||²
    v2 = np.linalg.solve(L_B, v1)
    term_posterior = np.sum(v2 ** 2, axis=0)

    var = k_diag - term_prior + term_posterior + sigma2
    var = np.maximum(var, 1e-10)

    return mu, var


# ---------------------------------------------------------------------------
# FieldBeliefModel implementation
# ---------------------------------------------------------------------------


@dataclass
class SPGPFITCConfig:
    """Configuration for the SPGP (FITC) field model.

    Parameters
    ----------
    n_pseudo:
        Number of pseudo-input points M.
    lengthscale:
        Initial lengthscale(s) for ARD-RBF kernel.
    variance:
        Initial signal variance (α).
    noise:
        Initial noise variance (σ²).
    include_time, include_depth:
        Whether to include time/depth in features.
    use_scaling:
        Standardise features before fitting.
    max_opt_iters:
        Maximum L-BFGS-B iterations for FITC likelihood optimisation.
    """

    n_pseudo: int = 50
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-2
    include_time: bool = False
    include_depth: bool = False
    use_scaling: bool = True
    max_opt_iters: int = 200


class SPGPFITCFieldModel(FieldBeliefModel):
    """Sparse Pseudo-inputs GP with FITC approximation.

    This is the paper-faithful model from Snelson & Ghahramani (2006),
    used as the SPGP backend in AdaPP.  Both kernel hyperparameters
    AND pseudo-input locations Z are optimised jointly via L-BFGS-B
    on the FITC marginal log-likelihood.

    Supports uncertainty; does NOT support online updates (refit needed).
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = dict(config or {})
        self._cfg = SPGPFITCConfig(
            n_pseudo=int(cfg.get("n_pseudo", 50)),
            lengthscale=float(cfg.get("lengthscale", 1.0)),
            variance=float(cfg.get("variance", 1.0)),
            noise=float(cfg.get("noise", 1e-2)),
            include_time=bool(cfg.get("include_time", False)),
            include_depth=bool(cfg.get("include_depth", False)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            max_opt_iters=int(cfg.get("max_opt_iters", 200)),
        )
        self._scaler: Optional[FeatureScaler] = None
        # Fitted parameters (all in log-space for optimisation).
        self._log_alpha: float = 0.0
        self._log_ls: Optional[np.ndarray] = None
        self._log_sigma2: float = 0.0
        self._Z: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

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
        X, y = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X)
            X = self._scaler.transform(X)
        else:
            self._scaler = None

        N, D = X.shape
        M = min(self._cfg.n_pseudo, N)

        # Initialise pseudo-inputs: random subset of training data.
        idx = self._rng.choice(N, size=M, replace=False)
        Z_init = X[idx].copy()

        # Initialise hyperparams in log-space.
        log_alpha = np.log(max(self._cfg.variance, 1e-6))
        log_ls = np.full(D, np.log(max(self._cfg.lengthscale, 1e-6)))
        log_sigma2 = np.log(max(self._cfg.noise, 1e-6))

        # Pack parameters: [log_alpha, log_ls(D), log_sigma2, Z(M*D)]
        params0 = np.concatenate([
            [log_alpha], log_ls, [log_sigma2], Z_init.ravel(),
        ])

        # Optimise.
        result = minimize(
            _fitc_nll,
            params0,
            args=(X, y, M, D),
            method="L-BFGS-B",
            options={"maxiter": self._cfg.max_opt_iters, "disp": False},
        )

        # Unpack.
        p = result.x
        self._log_alpha = float(p[0])
        self._log_ls = p[1:1 + D].copy()
        self._log_sigma2 = float(p[1 + D])
        self._Z = p[2 + D:].reshape(M, D).copy()
        self._X_train = X
        self._y_train = y

        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "SPGPFITCFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._Z is None:
            raise RuntimeError("Model must be fitted before prediction.")

        Xq = query_points_to_numpy(
            query_points,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        if self._scaler is not None:
            Xq = self._scaler.transform(Xq)

        mu, var = _fitc_predict(
            self._X_train, self._y_train, Xq, self._Z,
            self._log_alpha, self._log_ls, self._log_sigma2,
        )

        # Subtract noise variance for latent field std (consistent with
        # other OceanBench models).
        sigma2 = np.exp(self._log_sigma2)
        latent_var = np.maximum(var - sigma2, 0.0)

        return FieldPrediction(
            mean=mu,
            std=np.sqrt(latent_var),
            metadata={"noise_variance": sigma2},
        )

    @property
    def noise_variance(self) -> float:
        """Return the fitted noise variance σ²."""
        return float(np.exp(self._log_sigma2))

    def reset(self) -> None:
        super().reset()
        self._scaler = None
        self._Z = None
        self._X_train = None
        self._y_train = None
        self._log_ls = None
