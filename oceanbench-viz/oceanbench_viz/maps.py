"""
Plotting helpers for truth, prediction, uncertainty, and error maps.

All functions accept NumPy arrays; coordinates can be 1D (lat, lon) or 2D grids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _check_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for oceanbench_viz.maps")


def plot_field(
    values: np.ndarray,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    *,
    title: str = "Field",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[Any] = None,
    colorbar: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot a 2D field (e.g. truth, prediction, or uncertainty).

    Parameters
    ----------
    values : 2D array (ny, nx) or 1D array of size n. If 1D, lats and lons must be 1D and define points.
    lats, lons : If values is 2D, lats/lons can be 1D edges or 2D grids. If 1D, used for extent.
    title, cmap, vmin, vmax : Plot options.
    ax : Matplotlib axes. If None, current axes or new figure.
    colorbar : Whether to add a colorbar.
    **kwargs : Passed to pcolormesh or scatter.
    """
    _check_mpl()
    if ax is None:
        ax = plt.gca()

    values = np.asarray(values)
    if values.ndim == 1:
        if lats is None or lons is None:
            raise ValueError("For 1D values, lats and lons must be provided")
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        if lats.size != values.size or lons.size != values.size:
            raise ValueError("lats and lons must have the same length as values")
        sc = ax.scatter(lons, lats, c=values, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        if colorbar:
            plt.colorbar(sc, ax=ax, label=title)
        return sc

    # 2D
    ny, nx = values.shape
    if lats is None:
        lats = np.arange(ny + 1)
    if lons is None:
        lons = np.arange(nx + 1)
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    if lats.ndim == 1 and lats.size == ny:
        lats = np.concatenate([lats, [lats[-1] + (lats[-1] - lats[-2]) if len(lats) > 1 else 1]])
    if lons.ndim == 1 and lons.size == nx:
        lons = np.concatenate([lons, [lons[-1] + (lons[-1] - lons[-2]) if len(lons) > 1 else 1]])
    if lats.ndim == 1:
        Lats, Lons = np.meshgrid(lats, lons, indexing="ij")
    else:
        Lats, Lons = lats, lons
    pc = ax.pcolormesh(Lons, Lats, values, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(pc, ax=ax, label=title)
    return pc


def plot_truth_prediction_uncertainty(
    lats: np.ndarray,
    lons: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    *,
    obs_lats: Optional[np.ndarray] = None,
    obs_lons: Optional[np.ndarray] = None,
    title_prefix: str = "",
    figsize: Tuple[float, float] = (14, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Create a 3-panel figure: truth, prediction, and uncertainty (or error).

    All arrays are 1D (same length) and interpreted as point-wise values.
    Optionally overlay observation locations.
    """
    _check_mpl()
    n = np.asarray(truth).size
    prediction = np.asarray(prediction).ravel()
    if prediction.size != n:
        raise ValueError("prediction must have the same size as truth")
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty).ravel()
        if uncertainty.size != n:
            raise ValueError("uncertainty must have the same size as truth")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    vmin = min(np.nanmin(truth), np.nanmin(prediction))
    vmax = max(np.nanmax(truth), np.nanmax(prediction))

    ax = axes[0]
    ax.scatter(lons, lats, c=truth, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title_prefix} Truth".strip())
    plt.colorbar(ax.collections[0], ax=ax)

    ax = axes[1]
    ax.scatter(lons, lats, c=prediction, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title_prefix} Prediction".strip())
    plt.colorbar(ax.collections[0], ax=ax)

    ax = axes[2]
    if uncertainty is not None:
        sc = ax.scatter(lons, lats, c=uncertainty, cmap="plasma")
        ax.set_title(f"{title_prefix} Uncertainty (std)".strip())
    else:
        err = np.abs(np.asarray(truth).ravel() - prediction)
        sc = ax.scatter(lons, lats, c=err, cmap="Reds")
        ax.set_title(f"{title_prefix} |Error|".strip())
    plt.colorbar(sc, ax=ax)

    if obs_lats is not None and obs_lons is not None:
        for ax in axes:
            ax.scatter(obs_lons, obs_lats, s=20, c="k", marker="x", label="observations")
            ax.legend()

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
