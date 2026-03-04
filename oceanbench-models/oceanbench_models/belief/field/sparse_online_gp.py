"""
Sparse Online Gaussian Process for scalable, incremental field estimation.

Implements an online strategy on top of a variational inducing-point GP in
GPyTorch. The model maintains fixed inducing points and variational
parameters; `update()` performs additional ELBO steps on incoming mini-batches,
supporting genuine online learning while keeping memory and compute bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import gpytorch
import torch

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import (
    FeatureScaler,
    get_torch_device,
    numpy_to_torch,
    observation_batch_to_numpy,
    parse_gp_hyperparams,
    query_points_to_numpy,
)


@dataclass
class SparseOnlineGPConfig:
    """
    Configuration for the sparse online GP.

    Parameters
    ----------
    n_pseudo:
        Number of inducing points.
    lengthscale, variance, noise:
        Initial kernel and likelihood hyperparameters.
    include_time, include_depth:
        Whether to include time/depth in the feature space.
    use_scaling:
        If True, standardize features using a per-dimension mean/std.
    training_iters:
        Number of initial optimization steps on the first batch.
    update_iters:
        Number of optimization steps per `update()` mini-batch.
    """

    n_pseudo: int = 100
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    use_scaling: bool = True
    training_iters: int = 50
    update_iters: int = 10


class _SparseOnlineVariationalGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        *,
        lengthscale: float,
        variance: float,
    ) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.RBFKernel()
        base_kernel.lengthscale = float(lengthscale)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.outputscale = float(variance)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseOnlineGPFieldModel(FieldBeliefModel):
    """
    Sparse online GP that maintains a bounded set of inducing points.

    Initial training (`fit`) runs variational inference on the first batch.
    Subsequent `update` calls perform a small number of ELBO steps on new
    observations, updating the variational parameters in-place.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = dict(config or {})
        lengthscale, variance, noise = parse_gp_hyperparams(cfg)
        self._cfg = SparseOnlineGPConfig(
            n_pseudo=int(cfg.get("n_pseudo", 100)),
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            training_iters=int(cfg.get("training_iters", 50)),
            update_iters=int(cfg.get("update_iters", 10)),
        )

        self._device: torch.device = get_torch_device(cfg)
        self._scaler: Optional[FeatureScaler] = None

        self._model: Optional[_SparseOnlineVariationalGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._num_data: int = 0
        self._variable: Optional[str] = None

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

    def _select_inducing_points(self, X: ArrayLike) -> ArrayLike:
        """Choose inducing points from X (random subset)."""
        n = X.shape[0]
        m = min(self._cfg.n_pseudo, n)
        idx = self._rng.choice(n, size=m, replace=False)
        idx = np.sort(idx)
        return X[idx]

    def _ensure_optimizer(self) -> None:
        if self._model is None or self._likelihood is None:
            raise RuntimeError("Model and likelihood must be initialized before optimization.")
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                list(self._model.parameters()) + list(self._likelihood.parameters()),
                lr=float(self.config.get("lr", 0.01)),
            )

    def _elbo_objective(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> gpytorch.mlls.VariationalELBO:
        if self._model is None or self._likelihood is None:
            raise RuntimeError("Model and likelihood must be initialized before ELBO construction.")
        return gpytorch.mlls.VariationalELBO(
            self._likelihood,
            self._model,
            num_data=self._num_data,
        )

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        X_np, y_np = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X_np)
            X_np = self._scaler.transform(X_np)
        else:
            self._scaler = None

        Z_np = self._select_inducing_points(X_np)

        train_x = numpy_to_torch(X_np, device=self._device)
        train_y = numpy_to_torch(y_np.reshape(-1), device=self._device)
        inducing_points = numpy_to_torch(Z_np, device=self._device)

        model = _SparseOnlineVariationalGP(
            inducing_points,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        ).to(self._device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)
        likelihood.noise = float(self._cfg.noise)

        self._model = model
        self._likelihood = likelihood
        self._num_data = int(train_y.numel())
        self._variable = observations.variable

        self._ensure_optimizer()
        assert self._optimizer is not None

        model.train()
        likelihood.train()
        mll = self._elbo_objective(train_x, train_y)
        for _ in range(self._cfg.training_iters):
            self._optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            self._optimizer.step()

        model.eval()
        likelihood.eval()
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        """
        Incrementally update the variational parameters using a new mini-batch.

        The inducing set is kept fixed; only variational and hyper-parameters
        are nudged using a few ELBO steps on the new batch.
        """
        if not self.is_fitted or self._model is None or self._likelihood is None:
            raise RuntimeError("Model must be fitted before update.")

        X_np, y_np = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        if self._scaler is not None:
            X_np = self._scaler.transform(X_np)

        batch_x = numpy_to_torch(X_np, device=self._device)
        batch_y = numpy_to_torch(y_np.reshape(-1), device=self._device)

        # Update total data count for ELBO scaling.
        self._num_data += int(batch_y.numel())

        self._ensure_optimizer()
        assert self._optimizer is not None

        self._model.train()
        self._likelihood.train()
        mll = self._elbo_objective(batch_x, batch_y)
        for _ in range(self._cfg.update_iters):
            self._optimizer.zero_grad(set_to_none=True)
            output = self._model(batch_x)
            loss = -mll(output, batch_y)
            loss.backward()
            self._optimizer.step()

        self._model.eval()
        self._likelihood.eval()

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if (
            not self.is_fitted
            or self._model is None
            or self._likelihood is None
        ):
            raise RuntimeError("Model must be fitted before prediction.")

        Xq_np = query_points_to_numpy(
            query_points,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        if self._scaler is not None:
            Xq_np = self._scaler.transform(Xq_np)

        test_x = numpy_to_torch(Xq_np, device=self._device)

        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self._likelihood(self._model(test_x))
            mean = pred_dist.mean.detach().cpu().numpy()
            var_obs = pred_dist.variance.detach().cpu().numpy()

        # Remove homoskedastic observation noise to obtain latent field variance.
        noise_var = float(self._likelihood.noise.detach().cpu())
        var = np.maximum(var_obs.astype(float).reshape(-1) - noise_var, 0.0)
        mean = mean.astype(float).reshape(-1)
        std = np.sqrt(var)

        return FieldPrediction(mean=mean, std=std, metadata={})

    def reset(self) -> None:
        super().reset()
        self._scaler = None
        self._model = None
        self._likelihood = None
        self._optimizer = None
        self._num_data = 0
        self._variable = None

