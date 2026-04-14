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
class GPConfig:
    """
    Configuration for the full Gaussian Process field model.

    Parameters
    ----------
    lengthscale:
        Initial RBF kernel lengthscale (in feature space units).
    variance:
        Initial kernel signal variance.
    noise:
        Initial observation noise variance.
    include_time:
        Whether to include time as a feature when available.
    include_depth:
        Whether to include depth as a feature when available.
    use_scaling:
        If True, standardize features using a per-dimension mean/std.
    training_iters:
        Number of optimization steps for hyperparameters (0 = no training).
    """

    kernel_type: str = "rbf"
    nu: float = 2.5
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_time: bool = True
    include_depth: bool = True
    use_scaling: bool = True
    training_iters: int = 0


class _ExactGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        lengthscale: float,
        variance: float,
        kernel_type: str = "rbf",
        nu: float = 2.5,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=nu)
        else:
            base_kernel = gpytorch.kernels.RBFKernel()
        base_kernel.lengthscale = float(lengthscale)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.outputscale = float(variance)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPFieldModel(FieldBeliefModel):
    """
    Gaussian Process regression model for scalar ocean fields using GPyTorch.

    The model uses an ExactGP with an RBF kernel and Gaussian likelihood.
    It provides predictive means and standard deviations at arbitrary query
    points. Hyperparameters can optionally be optimized via marginal
    likelihood using a small number of training iterations.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg_mapping = dict(config or {})
        lengthscale, variance, noise = parse_gp_hyperparams(cfg_mapping)
        self._cfg = GPConfig(
            kernel_type=str(cfg_mapping.get("kernel_type", "rbf")).lower(),
            nu=float(cfg_mapping.get("nu", 2.5)),
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            include_time=bool(cfg_mapping.get("include_time", True)),
            include_depth=bool(cfg_mapping.get("include_depth", True)),
            use_scaling=bool(cfg_mapping.get("use_scaling", True)),
            training_iters=int(cfg_mapping.get("training_iters", 0)),
        )

        self._device: torch.device = get_torch_device(cfg_mapping)
        self._scaler: Optional[FeatureScaler] = None

        self._model: Optional[_ExactGPRegressionModel] = None
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

        train_x = numpy_to_torch(X_np, device=self._device)
        train_y = numpy_to_torch(y_np.reshape(-1), device=self._device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = float(self._cfg.noise)
        model = _ExactGPRegressionModel(
            train_x,
            train_y,
            likelihood,
            lengthscale=self._cfg.lengthscale,
            variance=self._cfg.variance,
            kernel_type=self._cfg.kernel_type,
            nu=self._cfg.nu,
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

        # Convert observation variance to latent field variance by removing
        # the (homoskedastic) noise variance from the Gaussian likelihood.
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

