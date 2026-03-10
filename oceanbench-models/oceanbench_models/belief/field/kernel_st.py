from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import gpytorch
import torch

from .covariance_backends import ArrayLike, CovarianceBackend
from .utils import FeatureScaler, get_torch_device, numpy_to_torch


@dataclass
class KernelSTConfig:
    """
    Configuration for the separable space-time RBF kernel backend.

    Parameters
    ----------
    lengthscale_space:
        RBF lengthscale for spatial dimensions (lat, lon[, depth]).
    lengthscale_time:
        RBF lengthscale for time.
    variance:
        Kernel output scale σ_f^2.
    noise:
        Observation noise variance used during hyperparameter fitting.
    include_depth:
        Whether to treat depth as an additional spatial dimension when present.
    use_scaling:
        If True, standardize features before passing them to the GP backend.
    fit:
        Whether hyperparameter fitting via marginal likelihood is enabled.
    fit_max_iters:
        Number of optimization steps when `fit` is True.
    fit_subsample_n:
        Maximum number of training points used when fitting.
    """

    lengthscale_space: float = 1.0
    lengthscale_time: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_depth: bool = True
    use_scaling: bool = True
    fit: bool = False
    fit_max_iters: int = 0
    fit_subsample_n: int = 512


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


class KernelSTCovariance(CovarianceBackend):
    """
    Separable spatio-temporal RBF kernel backend.

    This backend exposes covariance blocks over arbitrary feature matrices
    using a product kernel:

        k((x, t), (x', t')) = σ_f^2 * k_s(x, x') * k_t(t, t')

    with RBF kernels in space and time. When either training or query data
    lack time information, the backend gracefully falls back to a purely
    spatial RBF kernel.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        cfg = dict(config or {})
        self._cfg = KernelSTConfig(
            lengthscale_space=float(cfg.get("lengthscale_space", 1.0)),
            lengthscale_time=float(cfg.get("lengthscale_time", 1.0)),
            variance=float(cfg.get("variance", 1.0)),
            noise=float(cfg.get("noise", 1e-3)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            fit=bool(cfg.get("fit", False)),
            fit_max_iters=int(
                cfg.get("fit_max_iters", cfg.get("fit", {}).get("max_iters", 0))
                if isinstance(cfg.get("fit"), Mapping)
                else cfg.get("fit_max_iters", cfg.get("kernel.fit.max_iters", 0))
            ),
            fit_subsample_n=int(
                cfg.get("fit_subsample_n", cfg.get("fit", {}).get("subsample_n", 512))
                if isinstance(cfg.get("fit"), Mapping)
                else cfg.get("fit_subsample_n", cfg.get("kernel.fit.subsample_n", 512))
            ),
        )
        self._seed: Optional[int] = seed
        self._device: torch.device = get_torch_device(cfg)
        self._scaler: Optional[FeatureScaler] = None

        # Effective hyperparameters used in cov_block; may be updated by fit().
        self._lengthscale_space: float = self._cfg.lengthscale_space
        self._lengthscale_time: float = self._cfg.lengthscale_time
        self._variance: float = self._cfg.variance

        self._n_space_dims: int = 0
        self._uses_time: bool = False
        self._split_initialized: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_space_time_dims(self, X: ArrayLike) -> None:
        """
        Infer how many leading dimensions belong to space vs time.

        Assumes features are ordered as [lat, lon(, depth)[, time]].
        """
        X = np.asarray(X, dtype=float)
        n_dims = X.shape[1]
        if n_dims < 2:
            raise ValueError("Expected at least [lat, lon] features.")

        n_space = 2
        if self._cfg.include_depth and n_dims >= 3:
            n_space += 1

        self._n_space_dims = n_space
        self._uses_time = n_dims > n_space
        self._split_initialized = True

    def _ensure_split(self, X: ArrayLike) -> None:
        if not self._split_initialized:
            self._split_space_time_dims(X)

    # ------------------------------------------------------------------
    # Core CovarianceBackend API
    # ------------------------------------------------------------------

    def cov_block(self, Xa: ArrayLike, Xb: ArrayLike) -> ArrayLike:
        Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))

        if Xa.shape[1] != Xb.shape[1]:
            raise ValueError("Xa and Xb must have the same number of feature dimensions.")

        self._ensure_split(Xa)
        n_space = self._n_space_dims

        Xa_space = Xa[:, :n_space]
        Xb_space = Xb[:, :n_space]

        dists_sq_space = np.sum(
            (Xa_space[:, None, :] - Xb_space[None, :, :]) ** 2,
            axis=-1,
        )
        K_space = np.exp(-0.5 * dists_sq_space / (self._lengthscale_space**2))

        if self._uses_time:
            Xa_time = Xa[:, n_space : n_space + 1]
            Xb_time = Xb[:, n_space : n_space + 1]
            dists_sq_time = np.sum(
                (Xa_time[:, None, :] - Xb_time[None, :, :]) ** 2,
                axis=-1,
            )
            K_time = np.exp(-0.5 * dists_sq_time / (self._lengthscale_time**2))
            return self._variance * K_space * K_time

        return self._variance * K_space

    def diag_cov(self, X: ArrayLike) -> ArrayLike:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # For an RBF kernel, the marginal variance is constant σ_f^2.
        return np.full(X.shape[0], self._variance, dtype=float)

    # ------------------------------------------------------------------
    # Hyperparameter fitting via GPyTorch (optional)
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        if not self._cfg.fit or self._cfg.fit_max_iters <= 0:
            return

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        # Optional subsampling for efficiency.
        n = X.shape[0]
        if n > self._cfg.fit_subsample_n:
            rng = np.random.default_rng(self._seed)
            idx = rng.choice(n, size=self._cfg.fit_subsample_n, replace=False)
            X = X[idx]
            y = y[idx]

        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X)
            X = self._scaler.transform(X)
        else:
            self._scaler = None

        self._split_space_time_dims(X)

        train_x = numpy_to_torch(X, device=self._device)
        train_y = numpy_to_torch(y.reshape(-1), device=self._device)

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

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self._cfg.__dict__.get("lr", 0.1)),
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self._cfg.fit_max_iters):
            optimizer.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Extract fitted hyperparameters back into NumPy scalars.
        model.eval()
        with torch.no_grad():
            if self._uses_time:
                space_kernel = model.covar_module.base_kernel.kernels[0]
                time_kernel = model.covar_module.base_kernel.kernels[1]
            else:
                space_kernel = model.covar_module.base_kernel
                time_kernel = None

            self._lengthscale_space = float(space_kernel.lengthscale.detach().cpu())
            if time_kernel is not None:
                self._lengthscale_time = float(time_kernel.lengthscale.detach().cpu())
            self._variance = float(model.covar_module.outputscale.detach().cpu())

