"""
Gaussian Markov Random Field (GMRF) for scalable spatial field estimation.

Discretizes the domain on a regular grid (2-D or 3-D) and uses a sparse
precision matrix (Laplacian-style prior). Observations are binned to grid
cells; posterior mean is computed via sparse Cholesky. Prediction at
arbitrary points is by bilinear (2-D) or trilinear (3-D) interpolation
from the grid. Uncertainty is approximated from the posterior marginal
variances at grid nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction


@dataclass
class GMRFConfig:
    """
    Configuration for the GMRF model.

    Parameters
    ----------
    n_lat, n_lon:
        Grid resolution along latitude and longitude.
    n_depth:
        Grid resolution along depth.  When ``1`` (default) the model
        operates as a standard 2-D GMRF.
    obs_noise:
        Observation noise variance.
    prior_precision_scale:
        Scale of the Laplacian prior precision (larger = smoother).
    include_time, include_depth:
        Whether to use time / depth from observations.  ``include_depth``
        activates the 3-D grid when ``n_depth > 1``.
    """

    n_lat: int = 30
    n_lon: int = 30
    n_depth: int = 1
    obs_noise: float = 1e-2
    prior_precision_scale: float = 1.0
    include_time: bool = False
    include_depth: bool = True


# ---------------------------------------------------------------------------
# Laplacian precision (2-D or 3-D)
# ---------------------------------------------------------------------------


def _build_laplacian_precision(
    n_lat: int,
    n_lon: int,
    scale: float,
    n_depth: int = 1,
) -> sparse.csr_matrix:
    """Sparse Laplacian precision for a regular 2-D or 3-D grid.

    2-D uses a 5-point stencil; 3-D uses a 7-point stencil (face-connected
    neighbours only).
    """
    n_horiz = n_lat * n_lon
    n = n_horiz * n_depth
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for idx in range(n):
        if n_depth > 1:
            k = idx // n_horiz
            rem = idx % n_horiz
            i = rem // n_lon
            j = rem % n_lon
        else:
            k = 0
            i = idx // n_lon
            j = idx % n_lon

        count = 0

        # Lat neighbours.
        if i > 0:
            nb = k * n_horiz + (i - 1) * n_lon + j
            rows.append(idx); cols.append(nb); data.append(-scale); count += 1
        if i < n_lat - 1:
            nb = k * n_horiz + (i + 1) * n_lon + j
            rows.append(idx); cols.append(nb); data.append(-scale); count += 1

        # Lon neighbours.
        if j > 0:
            nb = k * n_horiz + i * n_lon + (j - 1)
            rows.append(idx); cols.append(nb); data.append(-scale); count += 1
        if j < n_lon - 1:
            nb = k * n_horiz + i * n_lon + (j + 1)
            rows.append(idx); cols.append(nb); data.append(-scale); count += 1

        # Depth neighbours (3-D only).
        if n_depth > 1:
            if k > 0:
                nb = (k - 1) * n_horiz + i * n_lon + j
                rows.append(idx); cols.append(nb); data.append(-scale); count += 1
            if k < n_depth - 1:
                nb = (k + 1) * n_horiz + i * n_lon + j
                rows.append(idx); cols.append(nb); data.append(-scale); count += 1

        # Diagonal (self).
        rows.append(idx); cols.append(idx); data.append(scale * count + 1e-8)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


# ---------------------------------------------------------------------------
# Coordinate ↔ grid-index helpers
# ---------------------------------------------------------------------------


def _coord_to_grid(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
    depth: float | None = None,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
    n_depth: int = 1,
) -> tuple[int, ...]:
    """Map (lat, lon[, depth]) to grid cell indices. Clamp to valid range."""
    if lat_max <= lat_min or lon_max <= lon_min:
        raise ValueError("Invalid domain: lat_max > lat_min and lon_max > lon_min")
    i = int((lat - lat_min) / (lat_max - lat_min) * n_lat)
    j = int((lon - lon_min) / (lon_max - lon_min) * n_lon)
    i = max(0, min(n_lat - 1, i))
    j = max(0, min(n_lon - 1, j))
    if n_depth > 1 and depth is not None:
        if depth_max <= depth_min:
            depth_max = depth_min + 1.0
        k = int((depth - depth_min) / (depth_max - depth_min) * n_depth)
        k = max(0, min(n_depth - 1, k))
        return i, j, k
    return i, j


def _grid_to_coord(
    i: int,
    j: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
    k: int | None = None,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
    n_depth: int = 1,
) -> tuple[float, ...]:
    """Cell center coordinates."""
    lat = lat_min + (i + 0.5) / n_lat * (lat_max - lat_min)
    lon = lon_min + (j + 0.5) / n_lon * (lon_max - lon_min)
    if k is not None and n_depth > 1:
        dep = depth_min + (k + 0.5) / n_depth * (depth_max - depth_min)
        return lat, lon, dep
    return lat, lon


# ---------------------------------------------------------------------------
# GMRF model
# ---------------------------------------------------------------------------


class GMRFFieldModel(FieldBeliefModel):
    """
    GMRF on a regular 2-D or 3-D grid with Laplacian prior.

    When ``n_depth > 1`` in the config, the model builds a 3-D grid and
    uses a 7-point Laplacian stencil.  Otherwise it falls back to the
    classical 2-D 5-point stencil.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = config or {}
        self._cfg = GMRFConfig(
            n_lat=int(cfg.get("n_lat", 30)),
            n_lon=int(cfg.get("n_lon", 30)),
            n_depth=int(cfg.get("n_depth", 1)),
            obs_noise=float(cfg.get("obs_noise", 1e-2)),
            prior_precision_scale=float(cfg.get("prior_precision_scale", 1.0)),
            include_time=bool(cfg.get("include_time", False)),
            include_depth=bool(cfg.get("include_depth", True)),
        )
        self._n_lat = self._cfg.n_lat
        self._n_lon = self._cfg.n_lon
        self._n_depth = self._cfg.n_depth if self._cfg.include_depth else 1
        self._use_3d = self._n_depth > 1

        self._lat_min = self._lat_max = 0.0
        self._lon_min = self._lon_max = 0.0
        self._depth_min = self._depth_max = 0.0
        self._grid_mean: Optional[np.ndarray] = None
        self._grid_std: Optional[np.ndarray] = None
        self._variable: Optional[str] = None

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        lats = np.asarray(observations.lats, dtype=float)
        lons = np.asarray(observations.lons, dtype=float)
        values = np.asarray(observations.values, dtype=float)
        self._variable = observations.variable

        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        if lat_max - lat_min < 1e-6:
            lat_min -= 0.01; lat_max += 0.01
        if lon_max - lon_min < 1e-6:
            lon_min -= 0.01; lon_max += 0.01
        self._lat_min, self._lat_max = lat_min, lat_max
        self._lon_min, self._lon_max = lon_min, lon_max

        n_lat, n_lon, n_depth = self._n_lat, self._n_lon, self._n_depth

        # Depth bounds (from observations or scenario).
        depths: np.ndarray | None = None
        if self._use_3d and observations.depths is not None:
            depths = np.asarray(observations.depths, dtype=float)
            self._depth_min = float(depths.min())
            self._depth_max = float(depths.max())
            if self._depth_max - self._depth_min < 1e-6:
                self._depth_min -= 1.0; self._depth_max += 1.0
        elif self._use_3d and scenario is not None and scenario.depth_range is not None:
            self._depth_min, self._depth_max = (
                float(scenario.depth_range[0]),
                float(scenario.depth_range[1]),
            )

        n_horiz = n_lat * n_lon
        n = n_horiz * n_depth

        # Bin observations to grid.
        sums = np.zeros(n)
        counts = np.zeros(n)
        for obs_i in range(lats.size):
            dep = float(depths[obs_i]) if depths is not None else None
            ijk = _coord_to_grid(
                lats[obs_i], lons[obs_i],
                lat_min, lat_max, lon_min, lon_max,
                n_lat, n_lon,
                depth=dep,
                depth_min=self._depth_min, depth_max=self._depth_max,
                n_depth=n_depth,
            )
            if self._use_3d:
                i, j, k = ijk
                flat_idx = k * n_horiz + i * n_lon + j
            else:
                i, j = ijk
                flat_idx = i * n_lon + j
            sums[flat_idx] += values[obs_i]
            counts[flat_idx] += 1

        observed = counts > 0
        obs_noise = self._cfg.obs_noise
        Q_prior = _build_laplacian_precision(
            n_lat, n_lon, self._cfg.prior_precision_scale, n_depth,
        )
        diag_obs = np.zeros(n)
        diag_obs[observed] = counts[observed] / obs_noise
        Q_post = Q_prior + sparse.diags(diag_obs, format="csr")

        rhs = np.zeros(n)
        rhs[observed] = sums[observed] / obs_noise

        mean_grid = splinalg.spsolve(Q_post, rhs)
        if np.any(~np.isfinite(mean_grid)):
            mean_grid = np.nan_to_num(mean_grid, nan=0.0, posinf=0.0, neginf=0.0)

        if self._use_3d:
            self._grid_mean = mean_grid.reshape(n_depth, n_lat, n_lon)
        else:
            self._grid_mean = mean_grid.reshape(n_lat, n_lon)

        approx_var = np.full(n, 1.0)
        approx_var[observed] = obs_noise / np.maximum(counts[observed], 1)
        approx_var[~observed] = 1.0

        if self._use_3d:
            self._grid_std = np.sqrt(approx_var).reshape(n_depth, n_lat, n_lon)
        else:
            self._grid_std = np.sqrt(approx_var).reshape(n_lat, n_lon)

        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "GMRFFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _interp_grid_2d(
        self, lats: ArrayLike, lons: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Bilinear interpolation of grid_mean and grid_std at (lats, lons)."""
        n_lat, n_lon = self._n_lat, self._n_lon
        lat_min, lat_max = self._lat_min, self._lat_max
        lon_min, lon_max = self._lon_min, self._lon_max

        lat_n = (np.asarray(lats) - lat_min) / (lat_max - lat_min) * n_lat - 0.5
        lon_n = (np.asarray(lons) - lon_min) / (lon_max - lon_min) * n_lon - 0.5
        lat_n = np.clip(lat_n, 0, n_lat - 1.001)
        lon_n = np.clip(lon_n, 0, n_lon - 1.001)

        i0 = np.floor(lat_n).astype(int)
        j0 = np.floor(lon_n).astype(int)
        i1 = np.minimum(i0 + 1, n_lat - 1)
        j1 = np.minimum(j0 + 1, n_lon - 1)
        si = lat_n - i0
        sj = lon_n - j0

        M = self._grid_mean
        S = self._grid_std
        mean = (
            (1 - si) * (1 - sj) * M[i0, j0]
            + (1 - si) * sj * M[i0, j1]
            + si * (1 - sj) * M[i1, j0]
            + si * sj * M[i1, j1]
        )
        std = (
            (1 - si) * (1 - sj) * S[i0, j0]
            + (1 - si) * sj * S[i0, j1]
            + si * (1 - sj) * S[i1, j0]
            + si * sj * S[i1, j1]
        )
        return mean, std

    def _interp_grid_3d(
        self,
        lats: ArrayLike,
        lons: ArrayLike,
        depths: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Trilinear interpolation of grid_mean and grid_std."""
        n_lat, n_lon, n_depth = self._n_lat, self._n_lon, self._n_depth
        lat_min, lat_max = self._lat_min, self._lat_max
        lon_min, lon_max = self._lon_min, self._lon_max
        dep_min, dep_max = self._depth_min, self._depth_max

        lat_n = (np.asarray(lats) - lat_min) / (lat_max - lat_min) * n_lat - 0.5
        lon_n = (np.asarray(lons) - lon_min) / (lon_max - lon_min) * n_lon - 0.5
        dep_n = (np.asarray(depths) - dep_min) / (dep_max - dep_min) * n_depth - 0.5

        lat_n = np.clip(lat_n, 0, n_lat - 1.001)
        lon_n = np.clip(lon_n, 0, n_lon - 1.001)
        dep_n = np.clip(dep_n, 0, n_depth - 1.001)

        i0 = np.floor(lat_n).astype(int)
        j0 = np.floor(lon_n).astype(int)
        k0 = np.floor(dep_n).astype(int)
        i1 = np.minimum(i0 + 1, n_lat - 1)
        j1 = np.minimum(j0 + 1, n_lon - 1)
        k1 = np.minimum(k0 + 1, n_depth - 1)
        si = lat_n - i0
        sj = lon_n - j0
        sk = dep_n - k0

        # Grid is indexed [depth, lat, lon].
        M = self._grid_mean
        S = self._grid_std

        def _trilinear(G: np.ndarray) -> np.ndarray:
            c000 = G[k0, i0, j0]
            c001 = G[k0, i0, j1]
            c010 = G[k0, i1, j0]
            c011 = G[k0, i1, j1]
            c100 = G[k1, i0, j0]
            c101 = G[k1, i0, j1]
            c110 = G[k1, i1, j0]
            c111 = G[k1, i1, j1]
            return (
                c000 * (1 - si) * (1 - sj) * (1 - sk)
                + c001 * (1 - si) * sj * (1 - sk)
                + c010 * si * (1 - sj) * (1 - sk)
                + c011 * si * sj * (1 - sk)
                + c100 * (1 - si) * (1 - sj) * sk
                + c101 * (1 - si) * sj * sk
                + c110 * si * (1 - sj) * sk
                + c111 * si * sj * sk
            )

        return _trilinear(M), _trilinear(S)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._grid_mean is None:
            raise RuntimeError("Model must be fitted before prediction.")
        lats = np.asarray(query_points.lats, dtype=float)
        lons = np.asarray(query_points.lons, dtype=float)

        if self._use_3d:
            if query_points.depths is not None:
                depths = np.asarray(query_points.depths, dtype=float)
            else:
                depths = np.full_like(lats, 0.5 * (self._depth_min + self._depth_max))
            mean, std = self._interp_grid_3d(lats, lons, depths)
        else:
            mean, std = self._interp_grid_2d(lats, lons)

        return FieldPrediction(mean=mean, std=std, metadata={})

    def reset(self) -> None:
        super().reset()
        self._grid_mean = None
        self._grid_std = None
        self._variable = None
