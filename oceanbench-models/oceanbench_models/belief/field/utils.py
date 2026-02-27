from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

ArrayLike = np.ndarray


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
    return variance * np.exp(-0.5 * dists_sq / (lengthscale ** 2))


def rbf_separable_space_time_kernel(
    X1: ArrayLike,
    X2: ArrayLike,
    *,
    lengthscale_space: float = 1.0,
    lengthscale_time: float = 1.0,
    variance: float = 1.0,
) -> ArrayLike:
    """
    Simple separable RBF kernel in space and time.

    Assumes features are ordered as [lat, lon, time] where time is already
    scaled to a reasonable numeric range (e.g., seconds or days).
    """

    X1 = np.atleast_2d(X1).astype(float)
    X2 = np.atleast_2d(X2).astype(float)

    if X1.shape[1] < 3 or X2.shape[1] < 3:
        raise ValueError(
            "Expected at least 3 feature dimensions [lat, lon, time] "
            "for spatio-temporal kernel."
        )

    X1_space = X1[:, :2]
    X2_space = X2[:, :2]
    X1_time = X1[:, 2:3]
    X2_time = X2[:, 2:3]

    dists_sq_space = np.sum(
        (X1_space[:, None, :] - X2_space[None, :, :]) ** 2,
        axis=-1,
    )
    dists_sq_time = np.sum(
        (X1_time[:, None, :] - X2_time[None, :, :]) ** 2,
        axis=-1,
    )

    K_space = np.exp(-0.5 * dists_sq_space / (lengthscale_space ** 2))
    K_time = np.exp(-0.5 * dists_sq_time / (lengthscale_time ** 2))
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

