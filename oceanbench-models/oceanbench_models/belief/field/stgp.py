"""
Spatio-Temporal Gaussian Process for field estimation with explicit time.

Implements a separable space-time GP using a product kernel in GPyTorch.
If time is not available in the data, the model transparently falls back to
an ordinary spatial GP on (lat, lon[, depth]) while preserving the same
FieldBeliefModel interface.
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
class STGPConfig:
    """
    Configuration for the spatio-temporal GP.

    Parameters
    ----------
    lengthscale_space:
        RBF lengthscale for spatial dimensions (lat, lon[, depth]).
    lengthscale_time:
        RBF lengthscale for time.
    variance, noise:
        Kernel and likelihood hyperparameters.
    include_depth:
        Whether to include depth as a spatial feature dimension.
    use_scaling:
        If True, standardize features before passing them to the GP backend.
    training_iters:
        Number of optimization steps for hyperparameters (0 = no training).
    """

    lengthscale_space: float = 1.0
    lengthscale_time: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_depth: bool = True
    use_scaling: bool = True
    training_iters: int = 0


class _STExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        lengthscale_space: float,
        lengthscale_time: float,
        variance: float,
        n_space_dims: int,
        use_time: bool,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Spatial kernel acts on the first n_space_dims.
        space_kernel = gpytorch.kernels.RBFKernel(active_dims=tuple(range(n_space_dims)))
        space_kernel.lengthscale = float(lengthscale_space)

        if use_time:
            time_kernel = gpytorch.kernels.RBFKernel(
                active_dims=(n_space_dims,),
            )
            time_kernel.lengthscale = float(lengthscale_time)
            base_kernel = gpytorch.kernels.ProductKernel(space_kernel, time_kernel)
        else:
            base_kernel = space_kernel

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.outputscale = float(variance)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class STGPFieldModel(FieldBeliefModel):
    """
    Spatio-temporal GP with separable space-time kernel using GPyTorch.

    If both observations and query points include time, a product kernel over
    (space x time) is used. If time is not available, the model falls back to
    a purely spatial GP with the same backend and interface.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = dict(config or {})
        _, variance, noise = parse_gp_hyperparams(cfg)
        self._cfg = STGPConfig(
            lengthscale_space=float(cfg.get("lengthscale_space", 1.0)),
            lengthscale_time=float(cfg.get("lengthscale_time", 1.0)),
            variance=variance,
            noise=noise,
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            training_iters=int(cfg.get("training_iters", 0)),
        )

        self._device: torch.device = get_torch_device(cfg)
        self._scaler: Optional[FeatureScaler] = None

        self._model: Optional[_STExactGPModel] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._train_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._variable: Optional[str] = None
        self._uses_time: bool = False
        self._n_space_dims: int = 0

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

    def _split_space_time_dims(self, X: ArrayLike) -> None:
        """
        Infer how many leading dimensions belong to space vs time.

        Assumes features are ordered as [lat, lon(, depth)[, time]].
        """
        n_dims = X.shape[1]
        if n_dims < 2:
            raise ValueError("Expected at least [lat, lon] features.")

        # Base spatial dims: lat, lon.
        n_space = 2
        if self._cfg.include_depth and n_dims >= 3:
            # Treat depth as an additional spatial dim.
            n_space += 1

        # If there are more dims beyond space dims, treat the next one as time.
        self._uses_time = n_dims > n_space
        self._n_space_dims = n_space

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        # Always include time in features; we will ignore it if not present.
        X_np, y_np = observation_batch_to_numpy(
            observations,
            include_time=True,
            include_depth=self._cfg.include_depth,
        )
        if X_np.shape[1] < 2:
            raise ValueError("STGPFieldModel requires at least latitude and longitude.")

        self._split_space_time_dims(X_np)

        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X_np)
            X_np = self._scaler.transform(X_np)
        else:
            self._scaler = None

        train_x = numpy_to_torch(X_np, device=self._device)
        train_y = numpy_to_torch(y_np.reshape(-1), device=self._device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = float(self._cfg.noise)
        model = _STExactGPModel(
            train_x,
            train_y,
            likelihood,
            lengthscale_space=self._cfg.lengthscale_space,
            lengthscale_time=self._cfg.lengthscale_time,
            variance=self._cfg.variance,
            n_space_dims=self._n_space_dims,
            use_time=self._uses_time,
        ).to(self._device)
        likelihood = likelihood.to(self._device)

        self._train_x = train_x
        self._train_y = train_y
        self._model = model
        self._likelihood = likelihood
        self._variable = observations.variable

        if self._cfg.training_iters > 0:
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(
                list(model.parameters()) + 
                list(likelihood.parameters()),
                lr=float(self.config.get("lr", 0.1)),
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
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
            "STGPFieldModel does not support online updates; "
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
            include_time=True,
            include_depth=self._cfg.include_depth,
        )
        if Xq_np.shape[1] < 2:
            raise ValueError("STGPFieldModel requires at least latitude and longitude in query points.")

        # If the model was trained with time but query points lack it, raise.
        if self._uses_time and Xq_np.shape[1] <= self._n_space_dims:
            raise ValueError(
                "STGPFieldModel was trained with time; QueryPoints must include time as well."
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

        # Observation variance includes homoskedastic noise from the likelihood;
        # subtract it to obtain the latent field variance.
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
        self._uses_time = False
        self._n_space_dims = 0

