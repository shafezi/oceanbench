from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .hyperparams import RBFHyperparameters, fit_rbf_hyperparams
from .noise import NoiseManager
from .utils import FeatureScaler, observation_batch_to_numpy, query_points_to_numpy


@dataclass
class SOGPPaperConfig:
    """
    Paper-faithful Sparse Online GP configuration.

    Note on coordinate space: when ``use_scaling=True`` (default), features
    are standardized to zero-mean/unit-variance before kernel evaluation.
    Lengthscales are therefore in **standardized units** (1.0 ≈ one
    standard deviation of the training data).  This differs from the
    Binney kernel backend which projects (lat, lon) to meters.  The SOGP
    convention is intentional: the ARD lengthscales are learned online via
    LOO-CV, so the initial values only matter as starting points for
    optimisation.
    """

    max_basis_size: int = 100
    novelty_threshold: float = 1e-6  # omega in the paper
    jitter: float = 1e-8
    lengthscale: float | list[float] = 1.0
    variance: float = 1.0
    include_time: bool = True
    include_depth: bool = True
    use_scaling: bool = True
    hyper_fit_method: str = "loo_cv"
    hyper_fit_max_iters: int = 80
    max_history: int = 10000
    q_clip: float = 1e4


class SOGPPaperFieldModel(FieldBeliefModel):
    """
    Sparse Online GP matching the update and pruning logic in the paper.

    Key components:
    - novelty test gamma with threshold omega (BV insertion rule),
    - bounded BV-set size m,
    - pruning score epsilon_i = |alpha_i| / Q_ii,
    - downdate formulas for alpha, C, Q when pruning.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = dict(config or {})
        self._cfg = SOGPPaperConfig(
            max_basis_size=max(1, int(cfg.get("max_basis_size", cfg.get("m", 100)))),
            novelty_threshold=float(cfg.get("novelty_threshold", cfg.get("omega", 1e-6))),
            jitter=float(cfg.get("jitter", 1e-8)),
            lengthscale=cfg.get("lengthscale", 1.0),
            variance=float(cfg.get("variance", 1.0)),
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            hyper_fit_method=str(cfg.get("hyper_fit_method", "loo_cv")).lower(),
            hyper_fit_max_iters=int(cfg.get("hyper_fit_max_iters", 80)),
            max_history=max(100, int(cfg.get("max_history", 10000))),
            q_clip=float(cfg.get("q_clip", 1e4)),
        )
        self._noise = NoiseManager(cfg)
        self._scaler: Optional[FeatureScaler] = None
        self._variable: Optional[str] = None

        # Constant prior mean.  The SOGP is a zero-mean GP internally;
        # we subtract this mean before updates and add it back at
        # prediction time so that predictions far from observed data
        # revert to the training mean, not to zero.
        self._prior_mean: float = 0.0

        # Core SOGP state over current BV-set.
        self._X_bv = np.zeros((0, 0), dtype=float)
        self._alpha = np.zeros((0,), dtype=float)
        self._C = np.zeros((0, 0), dtype=float)
        self._Q = np.zeros((0, 0), dtype=float)  # inverse kernel matrix over BV-set

        # History used for optional hyperparameter re-estimation.
        # _y_history stores **raw** (un-centred) targets so that the prior
        # mean can be recomputed when hyperparameters are re-estimated.
        self._X_history = np.zeros((0, 0), dtype=float)
        self._y_history = np.zeros((0,), dtype=float)
        self._n_added_since_hyper = 0

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        _ = scenario
        self.reset()
        X_raw, y = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        X_raw = _sanitize_features(X_raw)
        y = np.asarray(y, dtype=float).ravel()
        if X_raw.shape[0] == 0:
            raise ValueError("fit requires at least one observation.")

        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X_raw)
            X = self._scaler.transform(X_raw)
        else:
            self._scaler = None
            X = X_raw

        # Compute and subtract prior mean so the internal SOGP operates
        # on zero-mean residuals.
        self._prior_mean = float(np.mean(y))
        y_centred = y - self._prior_mean

        self._append_history(X, y)  # store raw targets for hyper re-estimation
        for x_i, y_i in zip(X, y_centred):
            self._update_single(np.asarray(x_i, dtype=float), float(y_i))

        self._variable = observations.variable
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before update.")

        X_raw, y = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        X_raw = _sanitize_features(X_raw)
        y = np.asarray(y, dtype=float).ravel()
        if self._scaler is not None:
            X = self._scaler.transform(X_raw)
        else:
            X = X_raw

        y_centred = y - self._prior_mean

        self._append_history(X, y)  # store raw targets
        for x_i, y_i in zip(X, y_centred):
            self._update_single(np.asarray(x_i, dtype=float), float(y_i))

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        Xq = query_points_to_numpy(
            query_points,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        Xq = _sanitize_features(Xq)
        Xq = self._align_query_features(Xq)
        if self._scaler is not None:
            Xq = self._scaler.transform(Xq)

        mean, var = self._predict_mean_var_from_features(Xq)
        mean = mean + self._prior_mean  # add back the constant prior mean
        std = np.sqrt(np.maximum(var, 0.0))
        return FieldPrediction(
            mean=mean,
            std=std,
            metadata={
                "basis_size": int(self._alpha.shape[0]),
                "noise_sigma2": float(self._noise.resolve_sigma2()),
                "rho": float(self.rho_since_last_hyper_update()),
            },
        )

    # ------------------------------------------------------------------
    # SOGP-specific helpers
    # ------------------------------------------------------------------

    def predictive_covariance(self, query_points: QueryPoints) -> np.ndarray:
        """
        Full predictive covariance for MI planners.

        Returns a symmetric positive-semidefinite matrix.  The diagonal is
        floored at ``jitter`` to prevent degenerate covariance blocks from
        destabilising downstream MI computations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predictive_covariance.")
        Xq = query_points_to_numpy(
            query_points,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        Xq = _sanitize_features(Xq)
        Xq = self._align_query_features(Xq)
        if self._scaler is not None:
            Xq = self._scaler.transform(Xq)

        n = Xq.shape[0]
        Kqq = self._kernel(Xq, Xq)
        if self._X_bv.shape[0] == 0:
            return _symmetrize(Kqq + self._cfg.jitter * np.eye(n, dtype=float))
        Kbq = self._kernel(self._X_bv, Xq)
        cov = Kqq + Kbq.T @ self._C @ Kbq
        cov = _symmetrize(cov)

        # Ensure the covariance is positive semi-definite.  After many
        # online updates the C matrix can accumulate large negative
        # eigenvalues, making the raw covariance non-PSD.  Project to the
        # PSD cone via eigenvalue clipping rather than just fixing the
        # diagonal, which can leave off-diagonal structure invalid.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, self._cfg.jitter)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov = _symmetrize(cov)
        cov = cov + self._cfg.jitter * np.eye(n, dtype=float)
        return cov

    def rho_since_last_hyper_update(self) -> float:
        m = max(1, int(self._cfg.max_basis_size))
        return float(self._n_added_since_hyper) / float(m)

    def fit_hyperparameters(self) -> dict[str, Any]:
        """
        Re-estimate kernel hyperparameters from buffered data and rebuild state.
        """
        if self._X_history.shape[0] < 4:
            return {
                "updated": False,
                "reason": "insufficient_history",
            }
        hp = fit_rbf_hyperparams(
            self._X_history,
            self._y_history,
            initial_lengthscale=self._cfg.lengthscale,
            initial_variance=self._cfg.variance,
            initial_noise=self._noise.resolve_sigma2(),
            method=self._cfg.hyper_fit_method,
            max_iters=self._cfg.hyper_fit_max_iters,
            jitter=self._cfg.jitter,
        )
        self._apply_hyperparameters(hp)
        self._rebuild_from_history()
        self._n_added_since_hyper = 0
        return {
            "updated": True,
            "lengthscale": hp.as_dict()["lengthscale"],
            "variance": float(hp.variance),
            "noise": float(hp.noise),
        }

    def _apply_hyperparameters(self, hp: RBFHyperparameters) -> None:
        ls = np.asarray(hp.lengthscales, dtype=float).ravel()
        self._cfg.lengthscale = float(ls[0]) if ls.size == 1 else [float(v) for v in ls]
        self._cfg.variance = float(hp.variance)
        if self._noise.config.mode == "fixed":
            self._noise.config.fixed_sigma2 = float(hp.noise)
            self._noise.reset()
        else:
            self._noise.update_from_gp_likelihood(float(hp.noise))

    def _rebuild_from_history(self) -> None:
        X = np.asarray(self._X_history, dtype=float)
        y_raw = np.asarray(self._y_history, dtype=float).ravel()
        if X.shape[0] == 0:
            return
        # Recompute prior mean from raw history and centre targets.
        self._prior_mean = float(np.mean(y_raw))
        y_centred = y_raw - self._prior_mean

        self._X_bv = np.zeros((0, X.shape[1]), dtype=float)
        self._alpha = np.zeros((0,), dtype=float)
        self._C = np.zeros((0, 0), dtype=float)
        self._Q = np.zeros((0, 0), dtype=float)
        for x_i, y_i in zip(X, y_centred):
            self._update_single(np.asarray(x_i, dtype=float), float(y_i))

    def _predict_mean_var_from_features(self, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_q = int(Xq.shape[0])
        if self._X_bv.shape[0] == 0:
            mean = np.zeros(n_q, dtype=float)
            var = np.full(n_q, float(self._cfg.variance), dtype=float)
            return mean, var
        Kbq = self._kernel(self._X_bv, Xq)
        mean = self._alpha @ Kbq
        # Use kernel diagonal directly (O(n)) instead of computing the
        # full n×n matrix and extracting the diagonal (O(n²)).
        base = self._kernel_diag(Xq)
        var = base + np.einsum("iq,ij,jq->q", Kbq, self._C, Kbq)
        var = np.maximum(var, self._cfg.jitter)
        return mean.astype(float).ravel(), var.astype(float).ravel()

    def _update_single(self, x: np.ndarray, y: float) -> None:
        m = int(self._alpha.shape[0])
        if m == 0:
            self._initialize_first_basis(x, y)
            self._n_added_since_hyper += 1
            return

        k = self._kernel(self._X_bv, x[None, :]).reshape(-1)
        kxx = float(self._kernel(x[None, :], x[None, :])[0, 0])
        mu = float(self._alpha @ k)
        var = float(kxx + k @ self._C @ k)
        noise = float(self._noise.resolve_sigma2())
        denom = max(var + noise, self._cfg.jitter)
        q = (float(y) - mu) / denom
        r = -1.0 / denom
        if not np.isfinite(q):
            q = 0.0
        if not np.isfinite(r):
            r = -1.0 / max(noise, self._cfg.jitter)
        q = float(np.clip(q, -self._cfg.q_clip, self._cfg.q_clip))
        r = float(np.clip(r, -self._cfg.q_clip, -1.0 / (10.0 * self._cfg.q_clip)))

        e_hat = self._Q @ k
        e_hat = np.nan_to_num(e_hat, nan=0.0, posinf=0.0, neginf=0.0)
        gamma = float(kxx - k @ e_hat)
        if not np.isfinite(gamma):
            gamma = self._cfg.jitter

        if gamma <= self._cfg.novelty_threshold:
            # Existing BV span is sufficient: renormalize posterior only.
            s_hat = self._C @ k + e_hat
            self._alpha = self._alpha + q * s_hat
            self._C = self._C + r * np.outer(s_hat, s_hat)
        else:
            # Add a new BV, then prune if memory cap exceeded.
            s = np.concatenate([self._C @ k, np.array([1.0])], axis=0)
            alpha_ext = np.concatenate([self._alpha, np.array([0.0])], axis=0)
            C_ext = np.zeros((m + 1, m + 1), dtype=float)
            C_ext[:m, :m] = self._C

            self._alpha = alpha_ext + q * s
            self._C = C_ext + r * np.outer(s, s)
            self._Q = _append_inverse_kernel(self._Q, e_hat, gamma, jitter=self._cfg.jitter)
            self._X_bv = np.vstack([self._X_bv, x.reshape(1, -1)])
            self._n_added_since_hyper += 1

            if self._alpha.shape[0] > self._cfg.max_basis_size:
                eps = np.abs(self._alpha) / np.maximum(np.diag(self._Q), self._cfg.jitter)
                j = int(np.argmin(eps))
                self._prune_basis(j)

        self._C = _symmetrize(self._C)
        self._Q = _symmetrize(self._Q)
        self._stabilize_state()

        if self._noise.config.mode == "estimate" and self._noise.config.estimate_method == "residual":
            self._noise.update(y_true=np.array([y]), y_pred=np.array([mu]))

    def _initialize_first_basis(self, x: np.ndarray, y: float) -> None:
        kxx = float(self._kernel(x[None, :], x[None, :])[0, 0])
        noise = float(self._noise.resolve_sigma2())
        denom = max(kxx + noise, self._cfg.jitter)
        q = float(y) / denom
        r = -1.0 / denom
        self._X_bv = x.reshape(1, -1)
        self._alpha = np.array([q], dtype=float)
        self._C = np.array([[r]], dtype=float)
        self._Q = np.array([[1.0 / max(kxx, self._cfg.jitter)]], dtype=float)

    def _prune_basis(self, j: int) -> None:
        """
        Paper Eq.(21) downdate when BV size exceeds cap.
        """
        m = int(self._alpha.shape[0])
        if m <= 1:
            return
        keep = [i for i in range(m) if i != j]

        alpha_j = float(self._alpha[j])
        q_j = float(self._Q[j, j])
        q_j = np.sign(q_j) * max(abs(q_j), self._cfg.jitter)
        Q_j = self._Q[keep, j]
        C_j = self._C[keep, j]
        c_j = float(self._C[j, j])

        alpha_t = self._alpha[keep]
        C_t = self._C[np.ix_(keep, keep)]
        Q_t = self._Q[np.ix_(keep, keep)]

        self._alpha = alpha_t - (alpha_j / q_j) * Q_j
        self._C = (
            C_t
            + (c_j / (q_j**2)) * np.outer(Q_j, Q_j)
            - (1.0 / q_j) * (np.outer(Q_j, C_j) + np.outer(C_j, Q_j))
        )
        self._Q = Q_t - (1.0 / q_j) * np.outer(Q_j, Q_j)
        self._X_bv = self._X_bv[keep, :]

    def _append_history(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if self._X_history.size == 0:
            self._X_history = X.copy()
            self._y_history = y.copy()
        else:
            self._X_history = np.vstack([self._X_history, X])
            self._y_history = np.concatenate([self._y_history, y], axis=0)
        if self._X_history.shape[0] > self._cfg.max_history:
            self._X_history = self._X_history[-self._cfg.max_history :, :]
            self._y_history = self._y_history[-self._cfg.max_history :]

    def _stabilize_state(self) -> None:
        self._alpha = np.nan_to_num(self._alpha, nan=0.0, posinf=0.0, neginf=0.0)
        self._C = np.nan_to_num(self._C, nan=0.0, posinf=0.0, neginf=0.0)
        self._Q = np.nan_to_num(self._Q, nan=0.0, posinf=0.0, neginf=0.0)
        lim = 1e8
        self._alpha = np.clip(self._alpha, -lim, lim)
        self._C = np.clip(self._C, -lim, lim)
        self._Q = np.clip(self._Q, -lim, lim)

    def _kernel_diag(self, X: np.ndarray) -> np.ndarray:
        """Kernel self-variance k(x_i, x_i) for each row — always σ² for RBF."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.full(X.shape[0], float(self._cfg.variance), dtype=float)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        ls = np.asarray(self._cfg.lengthscale, dtype=float).ravel()
        if ls.size == 1:
            ls = np.repeat(float(ls[0]), X1.shape[1])
        if ls.size != X1.shape[1]:
            ls = np.repeat(float(ls[0]), X1.shape[1])
        X1s = X1 / np.maximum(ls, self._cfg.jitter)
        X2s = X2 / np.maximum(ls, self._cfg.jitter)
        d2 = np.sum((X1s[:, None, :] - X2s[None, :, :]) ** 2, axis=-1)
        return float(self._cfg.variance) * np.exp(-0.5 * d2)

    def _align_query_features(self, X: np.ndarray) -> np.ndarray:
        """
        Align query feature dimensionality to the training feature space.

        If query points omit time/depth dimensions used during fit, pad missing
        columns with training-feature means so transformed values become zero.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Query feature matrix must be 2-D.")
        if self._scaler is not None:
            d_ref = int(self._scaler.mean_.shape[0])
            if X.shape[1] < d_ref:
                pad = np.tile(self._scaler.mean_[X.shape[1] :], (X.shape[0], 1))
                X = np.column_stack([X, pad])
            elif X.shape[1] > d_ref:
                X = X[:, :d_ref]
            return X
        if self._X_bv.size > 0:
            d_ref = int(self._X_bv.shape[1])
            if X.shape[1] < d_ref:
                pad = np.zeros((X.shape[0], d_ref - X.shape[1]), dtype=float)
                X = np.column_stack([X, pad])
            elif X.shape[1] > d_ref:
                X = X[:, :d_ref]
        return X

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._noise.reset()
        self._scaler = None
        self._variable = None
        self._prior_mean = 0.0
        self._X_bv = np.zeros((0, 0), dtype=float)
        self._alpha = np.zeros((0,), dtype=float)
        self._C = np.zeros((0, 0), dtype=float)
        self._Q = np.zeros((0, 0), dtype=float)
        self._X_history = np.zeros((0, 0), dtype=float)
        self._y_history = np.zeros((0,), dtype=float)
        self._n_added_since_hyper = 0

    def get_state(self) -> Mapping[str, Any]:
        return {
            "X_bv": self._X_bv.tolist(),
            "alpha": self._alpha.tolist(),
            "C": self._C.tolist(),
            "Q": self._Q.tolist(),
            "X_history": self._X_history.tolist(),
            "y_history": self._y_history.tolist(),
            "prior_mean": float(self._prior_mean),
            "noise_sigma2": float(self._noise.resolve_sigma2()),
            "n_added_since_hyper": int(self._n_added_since_hyper),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self._X_bv = np.asarray(state.get("X_bv", []), dtype=float)
        self._alpha = np.asarray(state.get("alpha", []), dtype=float)
        self._C = np.asarray(state.get("C", []), dtype=float)
        self._Q = np.asarray(state.get("Q", []), dtype=float)
        self._X_history = np.asarray(state.get("X_history", []), dtype=float)
        self._y_history = np.asarray(state.get("y_history", []), dtype=float)
        sigma2 = float(state.get("noise_sigma2", self._noise.resolve_sigma2()))
        self._noise.update_from_gp_likelihood(sigma2)
        self._prior_mean = float(state.get("prior_mean", 0.0))
        self._n_added_since_hyper = int(state.get("n_added_since_hyper", 0))
        self._mark_fitted(self._alpha.size > 0)


def _append_inverse_kernel(
    Q: np.ndarray,
    e_hat: np.ndarray,
    gamma: float,
    *,
    jitter: float,
) -> np.ndarray:
    m = int(Q.shape[0])
    gamma = float(np.sign(gamma) * max(abs(gamma), jitter))
    out = np.zeros((m + 1, m + 1), dtype=float)
    outer = np.outer(e_hat, e_hat) / gamma
    out[:m, :m] = Q + outer
    out[:m, m] = -e_hat / gamma
    out[m, :m] = -e_hat / gamma
    out[m, m] = 1.0 / gamma
    return _symmetrize(out)


def _symmetrize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    return 0.5 * (M + M.T)


def _sanitize_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Feature array must be 2-D.")
    if X.size == 0:
        return X
    # Replace non-finite values with per-column medians (or zeros if needed).
    Xc = X.copy()
    for j in range(Xc.shape[1]):
        col = Xc[:, j]
        mask = np.isfinite(col)
        if np.all(mask):
            continue
        if np.any(mask):
            fill = float(np.median(col[mask]))
        else:
            fill = 0.0
        col[~mask] = fill
        Xc[:, j] = col
    return Xc

