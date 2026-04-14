"""Trajectory plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _check_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for oceanbench_viz.trajectories")


def plot_route(
    route_points: np.ndarray,
    *,
    selected_points: Optional[np.ndarray] = None,
    candidate_points: Optional[np.ndarray] = None,
    title: str = "Route",
    annotate_order: bool = False,
    figsize: Tuple[float, float] = (6.0, 6.0),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot a single route over optional candidate/selected points.
    """
    _check_mpl()
    route = np.asarray(route_points, dtype=float)
    if route.ndim != 2 or route.shape[1] < 2:
        raise ValueError("route_points must have shape (n, 2+) with [lat, lon, ...].")
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if candidate_points is not None:
        cand = np.asarray(candidate_points, dtype=float)
        ax.scatter(cand[:, 1], cand[:, 0], s=10, c="lightgray", alpha=0.6, label="candidates")
    if selected_points is not None:
        sel = np.asarray(selected_points, dtype=float)
        ax.scatter(sel[:, 1], sel[:, 0], s=25, c="royalblue", alpha=0.9, label="selected")

    ax.plot(route[:, 1], route[:, 0], "-o", c="crimson", lw=1.8, ms=4, label="route")
    ax.scatter(route[0, 1], route[0, 0], c="green", s=60, marker="s", label="start")
    ax.scatter(route[-1, 1], route[-1, 0], c="black", s=60, marker="X", label="end")
    if annotate_order:
        for i, p in enumerate(route):
            ax.text(float(p[1]), float(p[0]), str(i), fontsize=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_route_sequence(
    routes: Sequence[np.ndarray],
    *,
    title_prefix: str = "Route",
    max_routes: int = 8,
    figsize_per_plot: Tuple[float, float] = (4.0, 3.8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot multiple routes (e.g. replans) in a grid layout.
    """
    _check_mpl()
    if len(routes) == 0:
        raise ValueError("routes is empty.")
    n = min(max_routes, len(routes))
    idx = np.linspace(0, len(routes) - 1, n).astype(int)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        squeeze=False,
    )
    for j in range(n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        ax = axes[r, c]
        if j >= n:
            ax.axis("off")
            continue
        route = np.asarray(routes[idx[j]], dtype=float)
        if route.ndim != 2 or route.shape[1] < 2:
            ax.axis("off")
            continue
        ax.plot(route[:, 1], route[:, 0], "-o", c="crimson", lw=1.2, ms=3)
        ax.scatter(route[0, 1], route[0, 0], c="green", s=30, marker="s")
        ax.scatter(route[-1, 1], route[-1, 0], c="black", s=30, marker="X")
        ax.set_title(f"{title_prefix} {idx[j]}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
