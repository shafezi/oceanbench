from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

try:  # Torch / GPyTorch are optional at import time for type-checkers.
    import torch
except Exception:  # pragma: no cover - exercised only when torch is missing
    torch = None  # type: ignore[assignment]

from oceanbench_core.types import ObservationBatch, QueryPoints

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Legacy kernel helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def rbf_kernel(
    X1: ArrayLike,
    X2: ArrayLike,
    *,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> ArrayLike:
    """
    Isotropic RBF (squared-exponential) kernel.

    k(x, x') = variance * exp(-0.5 * ||x - x'||^2 / lengthscale^2)
    """

    X1 = np.atleast_2d(X1).astype(float)
    X2 = np.atleast_2d(X2).astype(float)

    dists_sq = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    return variance * np.exp(-0.5 * dists_sq / (lengthscale**2))


def rbf_separable_space_time_kernel(
    X1: ArrayLike,
    X2: ArrayLike,
    *,
    lengthscale_space: float = 1.0,
    lengthscale_time: float = 1.0,
    variance: float = 1.0,
    n_space_dims: int = 2,
) -> ArrayLike:
    """
    Simple separable RBF kernel in space and time.

    Assumes features are ordered as [lat, lon, (depth,) time] where time
    is already scaled to a reasonable numeric range (e.g., seconds or days).

    Parameters
    ----------
    n_space_dims:
        Number of leading spatial dimensions (2 for lat/lon, 3 for
        lat/lon/depth).  The time column is assumed to follow immediately.
    """

    X1 = np.atleast_2d(X1).astype(float)
    X2 = np.atleast_2d(X2).astype(float)

    min_cols = n_space_dims + 1
    if X1.shape[1] < min_cols or X2.shape[1] < min_cols:
        raise ValueError(
            f"Expected at least {min_cols} feature dimensions "
            f"[{n_space_dims} spatial + 1 time] for spatio-temporal kernel."
        )

    X1_space = X1[:, :n_space_dims]
    X2_space = X2[:, :n_space_dims]
    X1_time = X1[:, n_space_dims:n_space_dims + 1]
    X2_time = X2[:, n_space_dims:n_space_dims + 1]

    dists_sq_space = np.sum(
        (X1_space[:, None, :] - X2_space[None, :, :]) ** 2,
        axis=-1,
    )
    dists_sq_time = np.sum(
        (X1_time[:, None, :] - X2_time[None, :, :]) ** 2,
        axis=-1,
    )

    K_space = np.exp(-0.5 * dists_sq_space / (lengthscale_space**2))
    K_time = np.exp(-0.5 * dists_sq_time / (lengthscale_time**2))
    return variance * K_space * K_time


def parse_gp_hyperparams(
    config: Mapping[str, float],
    *,
    default_lengthscale: float = 1.0,
    default_variance: float = 1.0,
    default_noise: float = 1e-3,
) -> Tuple[float, float, float]:
    """Helper to extract standard GP hyperparameters from a config mapping."""

    lengthscale = float(config.get("lengthscale", default_lengthscale))
    variance = float(config.get("variance", default_variance))
    noise = float(config.get("noise", default_noise))
    return lengthscale, variance, noise


# ---------------------------------------------------------------------------
# Shared helpers for data conversion, scaling, and device handling
# ---------------------------------------------------------------------------


def observation_batch_to_numpy(
    observations: ObservationBatch,
    *,
    include_time: bool = True,
    include_depth: bool = True,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Convert an ObservationBatch to (X, y) NumPy arrays suitable for backends.

    X has shape (n_obs, n_features); y has shape (n_obs,).
    """

    X = observations.as_features(
        include_time=include_time,
        include_depth=include_depth,
    ).astype(float)
    y = np.asarray(observations.values, dtype=float)
    return X, y


def query_points_to_numpy(
    query_points: QueryPoints,
    *,
    include_time: bool = True,
    include_depth: bool = True,
) -> ArrayLike:
    """
    Convert QueryPoints to a NumPy feature matrix.

    Mirrors observation_batch_to_numpy for consistency.
    """

    Xq = query_points.as_features(
        include_time=include_time,
        include_depth=include_depth,
    ).astype(float)
    return Xq


@dataclass
class FeatureScaler:
    """
    Simple feature standardizer for lat/lon (and optionally other features).

    Center and scale each feature dimension using statistics computed on the
    training features. This keeps model internals numeric and backend-friendly
    while remaining NumPy-based at the API boundary.
    """

    mean_: ArrayLike
    scale_: ArrayLike

    @classmethod
    def from_data(cls, X: ArrayLike) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean_=mean, scale_=scale)

    def transform(self, X: ArrayLike) -> ArrayLike:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def transform_torch(self, X: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        if torch is None:
            raise RuntimeError("torch is required for transform_torch but is not available.")
        mean = torch.as_tensor(self.mean_, dtype=X.dtype, device=X.device)
        scale = torch.as_tensor(self.scale_, dtype=X.dtype, device=X.device)
        return (X - mean) / scale


def get_torch_device(config: Mapping[str, object] | None = None) -> "torch.device":  # type: ignore[name-defined]
    """
    Resolve the torch device to use for GP backends.

    Falls back to CPU by default; config may optionally provide:
        - config["device"]: e.g. "cpu", "cuda", torch.device(...)
    """

    if torch is None:
        raise RuntimeError("torch is required but not available; install torch to use GP models.")

    if config is None:
        return torch.device("cpu")

    dev = config.get("device")  # type: ignore[assignment]
    if dev is None:
        return torch.device("cpu")
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device(dev)


def numpy_to_torch(
    X: ArrayLike,
    *,
    device: "torch.device",  # type: ignore[name-defined]
    dtype: "torch.dtype | None" = None,  # type: ignore[name-defined]
) -> "torch.Tensor":  # type: ignore[name-defined]
    """Convert NumPy array to a torch Tensor on the requested device."""

    if torch is None:
        raise RuntimeError("torch is required but not available; install torch to use GP models.")
    if dtype is None:
        dtype = torch.float32
    return torch.as_tensor(X, dtype=dtype, device=device)

