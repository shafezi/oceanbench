"""
Variational (inducing point) Gaussian Process for scalable field estimation.

Uses a fixed set of M inducing points and a variational approximation built
with GPyTorch. This replaces the previous FITC-style custom math with a
library-backed implementation while preserving the FieldBeliefModel API.
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
class PseudoInputGPConfig:
    """
    Configuration for the pseudo-input (variational) GP.

    Parameters
    ----------
    n_pseudo:
        Number of inducing points.
    lengthscale, variance, noise:
        Initial kernel and likelihood hyperparameters.
    include_time, include_depth:
        Whether to include time/depth in features.
    use_scaling:
        If True, standardize features using a per-dimension mean/std.
    training_iters:
        Number of optimization steps for variational ELBO.
    """

    n_pseudo: int = 100
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    use_scaling: bool = True
    training_iters: int = 50


class _VariationalGPModel(gpytorch.models.ApproximateGP):
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


class PseudoInputGPFieldModel(FieldBeliefModel):
    """
    Variational inducing-point GP using GPyTorch.

    Inducing points are chosen at fit time by sampling uniformly at random
    from the training inputs. Positions and variational parameters are
    optimized via an ELBO objective. Uncertainty is supported; online updates
    are not.
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
        self._cfg = PseudoInputGPConfig(
            n_pseudo=int(cfg.get("n_pseudo", 100)),
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            training_iters=int(cfg.get("training_iters", 50)),
        )

        self._device: torch.device = get_torch_device(cfg)
        self._scaler: Optional[FeatureScaler] = None

        self._model: Optional[_VariationalGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._variable: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return False

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

        model = _VariationalGPModel(
            inducing_points,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
        ).to(self._device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self._device)
        likelihood.noise = float(self._cfg.noise)

        self._model = model
        self._likelihood = likelihood
        self._train_x = train_x
        self._train_y = train_y
        self._variable = observations.variable

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=float(self.config.get("lr", 0.01)),
        )
        mll = gpytorch.mlls.VariationalELBO(
            likelihood,
            model,
            num_data=train_y.numel(),
        )
        for _ in range(self._cfg.training_iters):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "PseudoInputGPFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

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

        # Convert observation variance to latent field variance by subtracting
        # the learned homoskedastic noise from the Gaussian likelihood.
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
        self._train_x = None
        self._train_y = None
        self._variable = None

