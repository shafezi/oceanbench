"""
Gaussian Markov Random Field (GMRF) for scalable spatial field estimation.

Discretizes the domain on a regular 2D grid and uses a sparse precision
matrix (from a Laplacian-style prior). Observations are binned to grid cells;
posterior mean is computed via sparse Cholesky. Prediction at arbitrary
points is by bilinear interpolation from the grid. Uncertainty is approximated
from the posterior marginal variances at grid nodes.
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
        Grid resolution (number of cells in lat and lon).
    obs_noise:
        Observation noise variance (same units as field values).
    prior_precision_scale:
        Scale of the prior precision (larger = smoother prior).
    include_time, include_depth:
        Not used for the grid (grid is 2D lat-lon). If observations have
        time/depth, we use only (lat, lon) for binning; one grid per
        time/depth can be added in a future extension.
    """

    n_lat: int = 30
    n_lon: int = 30
    obs_noise: float = 1e-2
    prior_precision_scale: float = 1.0
    include_time: bool = False
    include_depth: bool = False


def _build_2d_laplacian_precision(n_lat: int, n_lon: int, scale: float) -> sparse.csr_matrix:
    """Build sparse precision matrix for a 2D grid (5-point Laplacian stencil)."""
    n = n_lat * n_lon
    # Linear index: i = row * n_lon + col  -> row = i // n_lon, col = i % n_lon
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n):
        row, col = i // n_lon, i % n_lon
        count = 0
        if row > 0:
            j = (row - 1) * n_lon + col
            rows.append(i); cols.append(j); data.append(-scale); count += 1
        if row < n_lat - 1:
            j = (row + 1) * n_lon + col
            rows.append(i); cols.append(j); data.append(-scale); count += 1
        if col > 0:
            j = row * n_lon + (col - 1)
            rows.append(i); cols.append(j); data.append(-scale); count += 1
        if col < n_lon - 1:
            j = row * n_lon + (col + 1)
            rows.append(i); cols.append(j); data.append(-scale); count += 1
        rows.append(i); cols.append(i); data.append(scale * count + 1e-8)
    Q = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return Q


def _lat_lon_to_grid(
    lat: float, lon: float,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_lat: int, n_lon: int,
) -> Tuple[int, int]:
    """Map (lat, lon) to grid cell (i, j). Clamp to [0, n-1]."""
    if lat_max <= lat_min or lon_max <= lon_min:
        raise ValueError("Invalid domain: lat_max > lat_min and lon_max > lon_min")
    i = int((lat - lat_min) / (lat_max - lat_min) * n_lat)
    j = int((lon - lon_min) / (lon_max - lon_min) * n_lon)
    i = max(0, min(n_lat - 1, i))
    j = max(0, min(n_lon - 1, j))
    return i, j


def _grid_to_lat_lon(
    i: int, j: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    n_lat: int, n_lon: int,
) -> Tuple[float, float]:
    """Cell center (lat, lon)."""
    lat = lat_min + (i + 0.5) / n_lat * (lat_max - lat_min)
    lon = lon_min + (j + 0.5) / n_lon * (lon_max - lon_min)
    return lat, lon


class GMRFFieldModel(FieldBeliefModel):
    """
    GMRF on a regular 2D lat-lon grid with Laplacian prior.

    Observations are binned to the nearest grid cell (mean of values in that
    cell). The posterior is Gaussian with sparse precision Q_post = Q_prior + H'H/sigma^2;
    we solve for the mean and optionally approximate diagonal of the posterior
    covariance for uncertainty. Prediction at arbitrary points uses bilinear
    interpolation from the grid. Supports uncertainty (interpolated from grid
    marginal variances). Does not support online updates (refit for new data).
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
            obs_noise=float(cfg.get("obs_noise", 1e-2)),
            prior_precision_scale=float(cfg.get("prior_precision_scale", 1.0)),
            include_time=bool(cfg.get("include_time", False)),
            include_depth=bool(cfg.get("include_depth", False)),
        )
        self._n_lat = self._cfg.n_lat
        self._n_lon = self._cfg.n_lon
        self._lat_min = self._lat_max = self._lon_min = self._lon_max = 0.0
        self._grid_mean: Optional[ArrayLike] = None
        self._grid_std: Optional[ArrayLike] = None
        self._variable: Optional[str] = None

    @property
    def supports_uncertainty(self) -> bool:
        return True

    @property
    def supports_online_update(self) -> bool:
        return False

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
        # Slightly expand to avoid degenerate domains
        if lat_max - lat_min < 1e-6:
            lat_min -= 0.01
            lat_max += 0.01
        if lon_max - lon_min < 1e-6:
            lon_min -= 0.01
            lon_max += 0.01
        self._lat_min = lat_min
        self._lat_max = lat_max
        self._lon_min = lon_min
        self._lon_max = lon_max

        n_lat, n_lon = self._n_lat, self._n_lon
        n = n_lat * n_lon

        # Bin observations to grid: sum of values and count per cell
        sums = np.zeros(n)
        counts = np.zeros(n)
        for k in range(lats.size):
            i, j = _lat_lon_to_grid(
                lats[k], lons[k],
                lat_min, lat_max, lon_min, lon_max,
                n_lat, n_lon,
            )
            idx = i * n_lon + j
            sums[idx] += values[k]
            counts[idx] += 1

        observed = counts > 0
        obs_noise = self._cfg.obs_noise
        Q_prior = _build_2d_laplacian_precision(
            n_lat, n_lon, self._cfg.prior_precision_scale,
        )
        # Q_post = Q_prior + (1/sigma^2) * H^T H, where H selects observed cells.
        # H is (n_obs x n): one row per observed cell with 1 at that index.
        # So H^T H is diagonal: (H^T H)[i,i] = count of observations in cell i (or 0).
        diag_obs = np.zeros(n)
        diag_obs[observed] = counts[observed] / obs_noise
        Q_post = Q_prior + sparse.diags(diag_obs, format="csr")

        # rhs = (1/sigma^2) * H^T y = per-cell (sum of y in cell) / sigma^2
        rhs = np.zeros(n)
        rhs[observed] = sums[observed] / obs_noise

        # Solve Q_post @ mean = rhs
        mean_grid = splinalg.spsolve(Q_post, rhs)
        if np.any(~np.isfinite(mean_grid)):
            mean_grid = np.nan_to_num(mean_grid, nan=0.0, posinf=0.0, neginf=0.0)
        self._grid_mean = mean_grid.reshape(n_lat, n_lon)

        # Approximate posterior marginal std: 1/sqrt(diag(Q_post^{-1}))
        # We approximate diag(inv(Q_post)) by solving Q_post x = e_i for a sample of i
        # or use a diagonal approximation. For simplicity use a constant plus inverse count.
        approx_var = np.full(n, 1.0)
        approx_var[observed] = obs_noise / np.maximum(counts[observed], 1)
        approx_var[~observed] = 1.0  # unobserved cells get high variance
        self._grid_std = np.sqrt(approx_var).reshape(n_lat, n_lon)
        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        raise NotImplementedError(
            "GMRFFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    def _interp_grid(self, lats: ArrayLike, lons: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Bilinear interpolation of grid_mean and grid_std at (lats, lons)."""
        n_lat, n_lon = self._n_lat, self._n_lon
        lat_min, lat_max = self._lat_min, self._lat_max
        lon_min, lon_max = self._lon_min, self._lon_max
        # Normalize to [0, n_lat-1] and [0, n_lon-1]
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

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._grid_mean is None:
            raise RuntimeError("Model must be fitted before prediction.")
        lats = np.asarray(query_points.lats, dtype=float)
        lons = np.asarray(query_points.lons, dtype=float)
        mean, std = self._interp_grid(lats, lons)
        return FieldPrediction(mean=mean, std=std, metadata={})

    def reset(self) -> None:
        super().reset()
        self._grid_mean = None
        self._grid_std = None
        self._variable = None
