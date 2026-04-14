from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import gpytorch
import numpy as np
import torch
from sklearn.cluster import KMeans

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .noise import NoiseManager
from .utils import (
    FeatureScaler,
    get_torch_device,
    numpy_to_torch,
    observation_batch_to_numpy,
    query_points_to_numpy,
)


INDUCING_STRATEGIES = ("fixed_grid", "kmeans", "learnable", "random")
TRAINING_SCHEDULES = ("every_step", "per_batch", "per_replan")


@dataclass
class SVGPConfig:
    n_inducing: int = 128
    inducing_strategy: str = "random"
    learn_inducing_locations: bool = True
    training_schedule: str = "per_batch"
    fit_iters: int = 100
    update_iters: int = 10
    replan_iters: int = 40
    batch_size: int = 256
    lr: float = 0.01
    include_time: bool = True
    include_depth: bool = True
    use_scaling: bool = True
    lengthscale: float | list[float] = 1.0
    variance: float = 1.0
    jitter: float = 1e-6


class _SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        *,
        lengthscale: float | list[float],
        variance: float,
        learn_inducing_locations: bool,
    ) -> None:
        m = int(inducing_points.shape[0])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(m)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=bool(learn_inducing_locations),
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        d = int(inducing_points.shape[1])
        rbf = gpytorch.kernels.RBFKernel(ard_num_dims=d)
        ls = np.asarray(lengthscale, dtype=float).ravel()
        if ls.size == 1:
            rbf.lengthscale = float(ls[0])
        else:
            ls = ls[:d] if ls.size >= d else np.pad(ls, (0, d - ls.size), mode="edge")
            rbf.lengthscale = torch.as_tensor(ls.reshape(1, -1), dtype=inducing_points.dtype)

        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)
        self.covar_module.outputscale = float(variance)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPGPyTorchFieldModel(FieldBeliefModel):
    """
    Sparse variational GP baseline with configurable inducing and training schedules.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = dict(config or {})
        self._cfg = SVGPConfig(
            n_inducing=max(1, int(cfg.get("n_inducing", cfg.get("m", 128)))),
            inducing_strategy=str(cfg.get("inducing_strategy", "random")).lower(),
            learn_inducing_locations=bool(cfg.get("learn_inducing_locations", True)),
            training_schedule=str(cfg.get("training_schedule", "per_batch")).lower(),
            fit_iters=max(0, int(cfg.get("fit_iters", 100))),
            update_iters=max(0, int(cfg.get("update_iters", 10))),
            replan_iters=max(0, int(cfg.get("replan_iters", 40))),
            batch_size=max(1, int(cfg.get("batch_size", 256))),
            lr=float(cfg.get("lr", 0.01)),
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            lengthscale=cfg.get("lengthscale", 1.0),
            variance=float(cfg.get("variance", 1.0)),
            jitter=float(cfg.get("jitter", 1e-6)),
        )
        if self._cfg.inducing_strategy not in INDUCING_STRATEGIES:
            raise ValueError(
                f"Unknown inducing_strategy {self._cfg.inducing_strategy!r}; "
                f"expected one of {INDUCING_STRATEGIES}."
            )
        if self._cfg.training_schedule not in TRAINING_SCHEDULES:
            raise ValueError(
                f"Unknown training_schedule {self._cfg.training_schedule!r}; "
                f"expected one of {TRAINING_SCHEDULES}."
            )

        self._noise = NoiseManager(cfg)
        self._device: torch.device = get_torch_device(cfg)
        self._scaler: Optional[FeatureScaler] = None

        # Constant prior mean (same approach as SOGP): centre targets
        # before training so the GP operates on residuals, then add
        # the mean back during prediction.
        self._prior_mean: float = 0.0

        self._model: Optional[_SVGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._variable: Optional[str] = None

        # Replay buffer for schedule-controlled updates.
        # Stores **raw** (un-centred) targets.
        self._X_replay = np.zeros((0, 0), dtype=float)
        self._y_replay = np.zeros((0,), dtype=float)

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return True

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        _ = scenario
        if self._seed is not None:
            torch.manual_seed(int(self._seed))
            np.random.seed(int(self._seed))

        X_raw, y = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        X_raw = _sanitize_features(X_raw)
        y = np.asarray(y, dtype=float).ravel()
        if X_raw.shape[0] == 0:
            raise ValueError("fit requires non-empty observations.")

        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X_raw)
            X = self._scaler.transform(X_raw)
        else:
            self._scaler = None
            X = X_raw

        self._prior_mean = float(np.mean(y))
        y_centred = y - self._prior_mean

        self._X_replay = X.copy()
        self._y_replay = y.copy()  # store raw targets
        self._initialize_model(X)
        self._variable = observations.variable
        self._train_on_numpy(X, y_centred, iters=self._cfg.fit_iters)
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        if not self.is_fitted or self._model is None or self._likelihood is None:
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

        if self._X_replay.size == 0:
            self._X_replay = X.copy()
            self._y_replay = y.copy()  # raw targets
        else:
            self._X_replay = np.vstack([self._X_replay, X])
            self._y_replay = np.concatenate([self._y_replay, y], axis=0)

        y_centred = y - self._prior_mean

        schedule = self._cfg.training_schedule
        if schedule == "every_step":
            for x_i, y_i in zip(X, y_centred):
                self._train_on_numpy(
                    x_i.reshape(1, -1),
                    np.array([y_i], dtype=float),
                    iters=max(1, self._cfg.update_iters),
                )
        elif schedule == "per_batch":
            self._train_on_numpy(X, y_centred, iters=self._cfg.update_iters)
        elif schedule == "per_replan":
            pass
        else:
            raise ValueError(f"Unsupported training schedule {schedule!r}.")

    def train_replan(self) -> None:
        """
        Optional hook for online loop: train when replanning triggers.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before train_replan.")
        if self._X_replay.shape[0] == 0:
            return
        y_centred = self._y_replay - self._prior_mean
        self._train_on_numpy(self._X_replay, y_centred, iters=self._cfg.replan_iters)

    def fit_hyperparameters(self) -> dict[str, Any]:
        """
        Treat additional ELBO optimization as hyperparameter re-estimation.
        """
        if not self.is_fitted:
            return {"updated": False, "reason": "not_fitted"}
        self.train_replan()
        return {"updated": True}

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._model is None:
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
        xq_t = numpy_to_torch(Xq, device=self._device)

        self._model.eval()
        assert self._likelihood is not None
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent = self._model(xq_t)
            mean = latent.mean.detach().cpu().numpy().astype(float).reshape(-1)
            var = latent.variance.detach().cpu().numpy().astype(float).reshape(-1)

        mean = mean + self._prior_mean  # add back the constant prior mean
        std = np.sqrt(np.maximum(var, 0.0))
        return FieldPrediction(
            mean=mean,
            std=std,
            metadata={
                "noise_sigma2": float(self._noise.resolve_sigma2()),
                "n_inducing": int(self._model.variational_strategy.inducing_points.shape[0]),
                "training_schedule": self._cfg.training_schedule,
            },
        )

    def predictive_covariance(self, query_points: QueryPoints) -> np.ndarray:
        """
        Full latent predictive covariance for MI planners.
        """
        if not self.is_fitted or self._model is None:
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
        xq_t = numpy_to_torch(Xq, device=self._device)
        self._model.eval()
        with torch.no_grad():
            latent = self._model(xq_t)
            cov = latent.covariance_matrix.detach().cpu().numpy().astype(float)
        cov = 0.5 * (cov + cov.T)
        cov = cov + self._cfg.jitter * np.eye(cov.shape[0], dtype=float)
        return cov

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialize_model(self, X: np.ndarray) -> None:
        m = min(self._cfg.n_inducing, int(X.shape[0]))
        Z = self._select_inducing(X, m=m)
        z_t = numpy_to_torch(Z, device=self._device)
        learn_inducing = bool(self._cfg.learn_inducing_locations)
        if self._cfg.inducing_strategy == "learnable":
            learn_inducing = True

        self._model = _SVGPModel(
            z_t,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
            learn_inducing_locations=learn_inducing,
        ).to(self._device)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)
        sigma2 = float(self._noise.resolve_sigma2())
        self._likelihood.noise = sigma2
        if self._noise.config.mode == "fixed":
            self._likelihood.raw_noise.requires_grad_(False)
        self._optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self._cfg.lr,
        )

    def _select_inducing(self, X: np.ndarray, *, m: int) -> np.ndarray:
        if m <= 0:
            raise ValueError("Number of inducing points must be positive.")
        strat = self._cfg.inducing_strategy
        n, d = X.shape
        if m >= n:
            return X.copy()
        if strat == "random" or strat == "learnable":
            idx = self._rng.choice(n, size=m, replace=False)
            return X[np.sort(idx)]
        if strat == "kmeans":
            km = KMeans(n_clusters=m, random_state=self._seed if self._seed is not None else 0, n_init=5)
            km.fit(X)
            return np.asarray(km.cluster_centers_, dtype=float)
        if strat == "fixed_grid":
            # Build approximately m points across a bounding grid in feature space.
            side = int(np.ceil(m ** (1.0 / d)))
            axes = []
            for j in range(d):
                lo = float(np.min(X[:, j]))
                hi = float(np.max(X[:, j]))
                if np.isclose(lo, hi):
                    axes.append(np.array([lo], dtype=float))
                else:
                    axes.append(np.linspace(lo, hi, side))
            mesh = np.meshgrid(*axes, indexing="ij")
            grid = np.column_stack([g.ravel() for g in mesh])
            if grid.shape[0] > m:
                idx = np.linspace(0, grid.shape[0] - 1, m).astype(int)
                grid = grid[idx]
            return grid.astype(float)
        raise ValueError(f"Unknown inducing strategy {strat!r}.")

    def _train_on_numpy(self, X: np.ndarray, y: np.ndarray, *, iters: int) -> None:
        if iters <= 0:
            return
        if self._model is None or self._likelihood is None or self._optimizer is None:
            raise RuntimeError("Model must be initialized before training.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] == 0:
            return

        train_x = numpy_to_torch(X, device=self._device)
        train_y = numpy_to_torch(y, device=self._device)
        num_data = max(1, int(self._X_replay.shape[0]))
        mll = gpytorch.mlls.VariationalELBO(self._likelihood, self._model, num_data=num_data)

        self._model.train()
        self._likelihood.train()

        bs = min(self._cfg.batch_size, int(train_x.shape[0]))
        for _ in range(int(iters)):
            if bs < train_x.shape[0]:
                idx = self._rng.choice(train_x.shape[0], size=bs, replace=False)
                x_b = train_x[idx]
                y_b = train_y[idx]
            else:
                x_b = train_x
                y_b = train_y
            self._optimizer.zero_grad(set_to_none=True)
            output = self._model(x_b)
            loss = -mll(output, y_b)
            loss.backward()
            self._optimizer.step()

        self._model.eval()
        self._likelihood.eval()

        if self._noise.config.mode == "estimate":
            self._noise.update(gp_likelihood_sigma2=float(self._likelihood.noise.detach().cpu()))
        if self._noise.config.mode == "fixed":
            sigma2 = float(self._noise.resolve_sigma2())
            self._likelihood.noise = sigma2

    def _align_query_features(self, X: np.ndarray) -> np.ndarray:
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
        if self._X_replay.size > 0:
            d_ref = int(self._X_replay.shape[1])
            if X.shape[1] < d_ref:
                pad = np.zeros((X.shape[0], d_ref - X.shape[1]), dtype=float)
                X = np.column_stack([X, pad])
            elif X.shape[1] > d_ref:
                X = X[:, :d_ref]
        return X

    def reset(self) -> None:
        super().reset()
        self._noise.reset()
        self._scaler = None
        self._prior_mean = 0.0
        self._model = None
        self._likelihood = None
        self._optimizer = None
        self._variable = None
        self._X_replay = np.zeros((0, 0), dtype=float)
        self._y_replay = np.zeros((0,), dtype=float)


def _sanitize_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Feature array must be 2-D.")
    if X.size == 0:
        return X
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

