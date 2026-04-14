"""
Plotting helpers for truth, prediction, uncertainty, and error maps.

All functions accept NumPy arrays; coordinates can be 1D (lat, lon) or 2D grids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

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


def plot_map_sequence(
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    truth_list: Sequence[np.ndarray],
    pred_list: Sequence[np.ndarray],
    std_list: Optional[Sequence[np.ndarray]] = None,
    times: Optional[Sequence[Any]] = None,
    max_frames: int = 6,
    figsize_per_frame: Tuple[float, float] = (4.0, 3.2),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot truth / prediction / uncertainty / error over multiple time frames.
    """
    _check_mpl()
    n_total = min(len(truth_list), len(pred_list))
    n_frames = min(max_frames, n_total)
    if n_frames <= 0:
        raise ValueError("plot_map_sequence requires at least one frame.")

    idx = np.linspace(0, n_total - 1, n_frames).astype(int)
    n_cols = n_frames
    n_rows = 4 if std_list is not None else 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_frame[0] * n_cols, figsize_per_frame[1] * n_rows),
        squeeze=False,
    )

    for c, i in enumerate(idx):
        truth = np.asarray(truth_list[i], dtype=float).ravel()
        pred = np.asarray(pred_list[i], dtype=float).ravel()
        if truth.size != pred.size:
            raise ValueError("Each truth/pred frame pair must have matching size.")
        title_t = f"t={times[i]}" if times is not None and i < len(times) else f"frame {i}"

        ax = axes[0, c]
        sc = ax.scatter(lons, lats, c=truth, cmap="viridis")
        ax.set_title(f"Truth ({title_t})")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, c]
        sc = ax.scatter(lons, lats, c=pred, cmap="viridis")
        ax.set_title(f"Prediction ({title_t})")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        row_err = 2
        if std_list is not None:
            std = np.asarray(std_list[i], dtype=float).ravel()
            if std.size != truth.size:
                raise ValueError("Each std frame must match truth frame size.")
            ax = axes[2, c]
            sc = ax.scatter(lons, lats, c=std, cmap="plasma")
            ax.set_title(f"Uncertainty ({title_t})")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            row_err = 3

        err = np.abs(pred - truth)
        ax = axes[row_err, c]
        sc = ax.scatter(lons, lats, c=err, cmap="Reds")
        ax.set_title(f"|Error| ({title_t})")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_field_depth_slices(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    *,
    n_slices: int = 4,
    title: str = "Field",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize_per_slice: Tuple[float, float] = (4.0, 3.5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot a 3-D field as multiple depth-slice panels.

    Parameters
    ----------
    values, lats, lons, depths:
        1-D arrays of the same length.
    n_slices:
        Number of depth slices to show.
    """
    _check_mpl()
    values = np.asarray(values, dtype=float).ravel()
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    depths = np.asarray(depths, dtype=float).ravel()

    unique_depths = np.unique(depths)
    if len(unique_depths) <= n_slices:
        slice_depths = unique_depths
    else:
        idx = np.linspace(0, len(unique_depths) - 1, n_slices).astype(int)
        slice_depths = unique_depths[idx]

    n = len(slice_depths)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_slice[0] * n, figsize_per_slice[1]),
        squeeze=False,
    )

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    for col, d in enumerate(slice_depths):
        ax = axes[0, col]
        mask = np.isclose(depths, d, atol=1e-6)
        if not mask.any():
            ax.axis("off")
            continue
        sc = ax.scatter(
            lons[mask], lats[mask], c=values[mask],
            cmap=cmap, vmin=vmin, vmax=vmax, s=10,
        )
        ax.set_title(f"{title}\ndepth={d:.1f}m")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_curves(
    metrics_history: Sequence[Mapping[str, Any]],
    *,
    x_key: str = "sample_count",
    keys: Sequence[str] = ("rmse", "mae", "uncertainty_mean_std"),
    figsize: Tuple[float, float] = (8.0, 4.0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot metric curves over mission progress.
    """
    _check_mpl()
    if len(metrics_history) == 0:
        raise ValueError("metrics_history is empty.")
    xs = np.array([float(m.get(x_key, i)) for i, m in enumerate(metrics_history)], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for key in keys:
        ys = np.array([float(m.get(key, np.nan)) for m in metrics_history], dtype=float)
        ax.plot(xs, ys, marker="o", label=str(key))
    ax.set_xlabel(x_key)
    ax.set_ylabel("metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
