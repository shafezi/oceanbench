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
    Configuration for the separable space-time kernel backend.

    Parameters
    ----------
    kernel_type:
        Kernel family: ``"rbf"`` (squared exponential) or ``"matern"``.
    nu:
        Matern smoothness parameter (0.5, 1.5, or 2.5). Ignored for RBF.
    lengthscale_space:
        Lengthscale for spatial dimensions.  Units depend on
        ``project_to_meters``: meters if True, degrees if False.
    lengthscale_time:
        Lengthscale for time (seconds).
    variance:
        Kernel output scale σ_f^2.
    noise:
        Observation noise variance used during hyperparameter fitting.
    include_depth:
        Whether to treat depth as an additional spatial dimension when present.
    use_scaling:
        If True, standardize features before passing them to the GP backend
        during hyperparameter fitting.
    project_to_meters:
        If True (default), project (lat, lon) from degrees to meters using
        an equirectangular approximation before computing spatial distances.
        When False, distances are computed in raw degree space.
    fit:
        Whether hyperparameter fitting via marginal likelihood is enabled.
    fit_max_iters:
        Number of optimization steps when `fit` is True.
    fit_subsample_n:
        Maximum number of training points used when fitting.
    """

    kernel_type: str = "rbf"
    nu: float = 2.5
    lengthscale_space: float = 1.0
    lengthscale_time: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3
    include_depth: bool = True
    use_scaling: bool = True
    project_to_meters: bool = True
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
        kernel_type: str = "rbf",
        nu: float = 2.5,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        def _make_kernel(active_dims: tuple, ls: float) -> gpytorch.kernels.Kernel:
            if kernel_type == "matern":
                k = gpytorch.kernels.MaternKernel(nu=nu, active_dims=active_dims)
            else:
                k = gpytorch.kernels.RBFKernel(active_dims=active_dims)
            k.lengthscale = float(ls)
            return k

        space_kernel = _make_kernel(tuple(range(n_space_dims)), lengthscale_space)

        if use_time:
            time_kernel = _make_kernel((n_space_dims,), lengthscale_time)
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
            kernel_type=str(cfg.get("kernel_type", "rbf")).lower(),
            nu=float(cfg.get("nu", 2.5)),
            lengthscale_space=float(cfg.get("lengthscale_space", 1.0)),
            lengthscale_time=float(cfg.get("lengthscale_time", 1.0)),
            variance=float(cfg.get("variance", 1.0)),
            noise=float(cfg.get("noise", 1e-3)),
            include_depth=bool(cfg.get("include_depth", True)),
            use_scaling=bool(cfg.get("use_scaling", True)),
            project_to_meters=bool(cfg.get("project_to_meters", True)),
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

        # Reference latitude for equirectangular projection (set on first
        # call to cov_block from the mean latitude of the input data).
        self._ref_lat_rad: Optional[float] = None

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
    # Projection helpers
    # ------------------------------------------------------------------

    _DEG_TO_M = 111_320.0  # meters per degree of latitude (approximate)

    def _project_space_to_meters(self, X_space: np.ndarray) -> np.ndarray:
        """
        Project (lat, lon[, depth]) from degrees to meters using an
        equirectangular approximation.

        The first column is latitude, the second is longitude.  Both are
        converted to meters.  Depth (if present as a third column) is
        already in meters and left unchanged.

        The reference latitude for the cos-correction is computed once from
        the first batch of data seen and then held constant to ensure
        Σ(A, B) is consistent regardless of call order.
        """
        X = X_space.copy()
        if self._ref_lat_rad is None:
            self._ref_lat_rad = float(np.deg2rad(np.mean(X[:, 0])))

        cos_ref = np.cos(self._ref_lat_rad)

        # lat → northing (meters)
        X[:, 0] = X[:, 0] * self._DEG_TO_M
        # lon → easting (meters), corrected for latitude
        X[:, 1] = X[:, 1] * self._DEG_TO_M * cos_ref
        # depth (col 2 if present) is already in meters — no change needed
        return X

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

        if self._cfg.project_to_meters:
            Xa_space = self._project_space_to_meters(Xa_space)
            Xb_space = self._project_space_to_meters(Xb_space)

        # Apply feature scaling if a scaler was fitted (ensures cov_block
        # operates in the same coordinate system as the fitted hyperparameters).
        if self._scaler is not None:
            Xa_scaled = np.column_stack(
                [Xa_space] + ([Xa[:, n_space:]] if Xa.shape[1] > n_space else [])
            )
            Xb_scaled = np.column_stack(
                [Xb_space] + ([Xb[:, n_space:]] if Xb.shape[1] > n_space else [])
            )
            Xa_scaled = self._scaler.transform(Xa_scaled)
            Xb_scaled = self._scaler.transform(Xb_scaled)
            Xa_space = Xa_scaled[:, :n_space]
            Xb_space = Xb_scaled[:, :n_space]
            if self._uses_time:
                Xa_time = Xa_scaled[:, n_space : n_space + 1]
                Xb_time = Xb_scaled[:, n_space : n_space + 1]
        elif self._uses_time:
            Xa_time = Xa[:, n_space : n_space + 1]
            Xb_time = Xb[:, n_space : n_space + 1]

        dists_sq_space = np.sum(
            (Xa_space[:, None, :] - Xb_space[None, :, :]) ** 2,
            axis=-1,
        )
        K_space = self._kernel_1d(dists_sq_space, self._lengthscale_space)

        if self._uses_time:
            dists_sq_time = np.sum(
                (Xa_time[:, None, :] - Xb_time[None, :, :]) ** 2,
                axis=-1,
            )
            K_time = self._kernel_1d(dists_sq_time, self._lengthscale_time)
            return self._variance * K_space * K_time

        return self._variance * K_space

    def _kernel_1d(self, dists_sq: np.ndarray, lengthscale: float) -> np.ndarray:
        """Evaluate 1-D kernel given squared distances."""
        if self._cfg.kernel_type == "matern":
            r = np.sqrt(np.clip(dists_sq, 0.0, None)) / lengthscale
            nu = self._cfg.nu
            if nu == 0.5:
                return np.exp(-r)
            elif nu == 1.5:
                s = np.sqrt(3.0) * r
                return (1.0 + s) * np.exp(-s)
            else:  # 2.5
                s = np.sqrt(5.0) * r
                return (1.0 + s + 5.0 / 3.0 * r ** 2) * np.exp(-s)
        # RBF
        return np.exp(-0.5 * dists_sq / (lengthscale ** 2))

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

        self._split_space_time_dims(X)

        # Project spatial dimensions to meters so that fitted lengthscales
        # are in a consistent metric space (matches cov_block behaviour).
        if self._cfg.project_to_meters:
            n_space = self._n_space_dims
            X_space = self._project_space_to_meters(X[:, :n_space].copy())
            if X.shape[1] > n_space:
                X = np.column_stack([X_space, X[:, n_space:]])
            else:
                X = X_space

        if self._cfg.use_scaling:
            self._scaler = FeatureScaler.from_data(X)
            X = self._scaler.transform(X)
        else:
            self._scaler = None

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
            kernel_type=self._cfg.kernel_type,
            nu=self._cfg.nu,
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

