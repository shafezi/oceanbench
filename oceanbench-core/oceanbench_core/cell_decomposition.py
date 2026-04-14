"""Cell decomposition for AdaPP adaptive path planning.

Decomposes a survey region into a C1 x C2 grid of cells.  Each cell
tracks the planning-grid points it contains, its mean predictive
variance (Eq. 6 of Mishra et al. 2018), and its variance-weighted
centroid (Eq. 7).

Usage
-----
>>> grid = CellGrid.from_region(region, n_cell_lat=10, n_cell_lon=10,
...                             graph=waypoint_graph)
>>> grid.initialize(initial_variance=1.0)          # κ for t=0
>>> grid.update_from_model(field_model)             # after observations
>>> cell = grid.cell(3, 4)
>>> cell.variance, cell.centroid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .types import QueryPoints


# ---------------------------------------------------------------------------
# Cell dataclass
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """A single cell in the decomposed survey area."""

    row: int
    col: int

    # Spatial bounds (degrees).
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    # Optional depth bounds (meters, positive-down).
    depth_min: float | None = None
    depth_max: float | None = None
    layer: int = 0  # depth-layer index (0 when 2-D)

    # Planning-grid point indices that fall inside this cell.
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Point coordinates (cached for fast Eq. 7 computation).
    point_lats: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    point_lons: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    point_depths: np.ndarray | None = None

    # Per-point predictive variance (updated by model or initialised to κ).
    point_variances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @property
    def n_points(self) -> int:
        return len(self.point_indices)

    @property
    def variance(self) -> float:
        """Mean predictive variance over all points in cell (Eq. 6)."""
        if self.n_points == 0:
            return 0.0
        return float(np.mean(self.point_variances))

    @property
    def centroid(self) -> tuple[float, ...]:
        """Variance-weighted centroid (Eq. 7).

        Returns ``(lat, lon)`` in 2-D or ``(lat, lon, depth)`` in 3-D.
        Falls back to geometric centre when all variances are zero.
        """
        if self.n_points == 0:
            base = (
                0.5 * (self.lat_min + self.lat_max),
                0.5 * (self.lon_min + self.lon_max),
            )
            if self.depth_min is not None and self.depth_max is not None:
                return base + (0.5 * (self.depth_min + self.depth_max),)
            return base

        total_var = float(np.sum(self.point_variances))
        if total_var < 1e-30:
            lat = float(np.mean(self.point_lats))
            lon = float(np.mean(self.point_lons))
            if self.point_depths is not None:
                return (lat, lon, float(np.mean(self.point_depths)))
            return (lat, lon)

        lat = float(np.sum(self.point_variances * self.point_lats) / total_var)
        lon = float(np.sum(self.point_variances * self.point_lons) / total_var)
        if self.point_depths is not None:
            dep = float(np.sum(self.point_variances * self.point_depths) / total_var)
            return (lat, lon, dep)
        return (lat, lon)

    @property
    def geometric_centre(self) -> tuple[float, ...]:
        base = (
            0.5 * (self.lat_min + self.lat_max),
            0.5 * (self.lon_min + self.lon_max),
        )
        if self.depth_min is not None and self.depth_max is not None:
            return base + (0.5 * (self.depth_min + self.depth_max),)
        return base


# ---------------------------------------------------------------------------
# Cell grid
# ---------------------------------------------------------------------------


class CellGrid:
    """Grid of cells over a survey region, optionally with depth layers.

    Parameters
    ----------
    cells:
        2-D (or 3-D flattened to 2-D) array of :class:`Cell` objects.
    grid_lats, grid_lons:
        Full arrays of planning-grid point coordinates.
    grid_depths:
        Optional 1-D array of planning-grid depth coordinates.
    """

    def __init__(
        self,
        cells: list[list[Cell]],
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        grid_depths: np.ndarray | None = None,
    ) -> None:
        self._cells = cells
        self.grid_lats = grid_lats
        self.grid_lons = grid_lons
        self.grid_depths = grid_depths
        self.n_rows = len(cells)
        self.n_cols = len(cells[0]) if cells else 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_region(
        cls,
        region: Mapping[str, float],
        n_cell_lat: int,
        n_cell_lon: int,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        *,
        n_cell_depth: int = 1,
        grid_depths: np.ndarray | None = None,
    ) -> "CellGrid":
        """Build a cell grid from region bounds and planning-grid points.

        Parameters
        ----------
        region:
            Mapping with ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.
            May include ``depth_min``, ``depth_max`` for 3-D decomposition.
        n_cell_lat, n_cell_lon:
            Number of cells along each horizontal axis.
        grid_lats, grid_lons:
            1-D arrays of planning-grid point coordinates (same length).
        n_cell_depth:
            Number of depth layers (default 1 = 2-D only).
        grid_depths:
            Optional 1-D array of planning-grid depths (same length as
            *grid_lats*).
        """
        lat_min = float(region["lat_min"])
        lat_max = float(region["lat_max"])
        lon_min = float(region["lon_min"])
        lon_max = float(region["lon_max"])

        lat_edges = np.linspace(lat_min, lat_max, n_cell_lat + 1)
        lon_edges = np.linspace(lon_min, lon_max, n_cell_lon + 1)

        grid_lats = np.asarray(grid_lats, dtype=float)
        grid_lons = np.asarray(grid_lons, dtype=float)

        use_depth = (
            n_cell_depth > 1
            and grid_depths is not None
            and "depth_min" in region
            and "depth_max" in region
        )
        if use_depth:
            grid_depths_arr = np.asarray(grid_depths, dtype=float)
            depth_min = float(region["depth_min"])
            depth_max = float(region["depth_max"])
            depth_edges = np.linspace(depth_min, depth_max, n_cell_depth + 1)
        else:
            grid_depths_arr = None
            depth_edges = None

        n_depth_layers = n_cell_depth if use_depth else 1

        cells: list[list[Cell]] = []
        for k in range(n_depth_layers):
            for i in range(n_cell_lat):
                row: list[Cell] = []
                for j in range(n_cell_lon):
                    c_lat_min = lat_edges[i]
                    c_lat_max = lat_edges[i + 1]
                    c_lon_min = lon_edges[j]
                    c_lon_max = lon_edges[j + 1]

                    # Horizontal masking.
                    lat_mask = (grid_lats >= c_lat_min) & (
                        grid_lats <= c_lat_max if i == n_cell_lat - 1
                        else grid_lats < c_lat_max
                    )
                    lon_mask = (grid_lons >= c_lon_min) & (
                        grid_lons <= c_lon_max if j == n_cell_lon - 1
                        else grid_lons < c_lon_max
                    )
                    mask = lat_mask & lon_mask

                    # Depth masking when applicable.
                    c_dep_min: float | None = None
                    c_dep_max: float | None = None
                    if use_depth and grid_depths_arr is not None and depth_edges is not None:
                        c_dep_min = float(depth_edges[k])
                        c_dep_max = float(depth_edges[k + 1])
                        dep_mask = (grid_depths_arr >= c_dep_min) & (
                            grid_depths_arr <= c_dep_max if k == n_depth_layers - 1
                            else grid_depths_arr < c_dep_max
                        )
                        mask = mask & dep_mask

                    indices = np.where(mask)[0]

                    cell = Cell(
                        row=i,
                        col=j,
                        lat_min=c_lat_min,
                        lat_max=c_lat_max,
                        lon_min=c_lon_min,
                        lon_max=c_lon_max,
                        depth_min=c_dep_min,
                        depth_max=c_dep_max,
                        layer=k,
                        point_indices=indices,
                        point_lats=grid_lats[indices],
                        point_lons=grid_lons[indices],
                        point_depths=grid_depths_arr[indices] if grid_depths_arr is not None else None,
                        point_variances=np.zeros(len(indices)),
                    )
                    row.append(cell)
                cells.append(row)

        return cls(cells, grid_lats, grid_lons, grid_depths=grid_depths)

    @classmethod
    def from_graph(
        cls,
        region: Mapping[str, float],
        n_cell_lat: int,
        n_cell_lon: int,
        graph: Any,
    ) -> "CellGrid":
        """Build from a WaypointGraph, extracting node coordinates."""
        nodes = list(graph.graph.nodes)
        lats = np.array([float(graph.graph.nodes[n]["lat"]) for n in nodes])
        lons = np.array([float(graph.graph.nodes[n]["lon"]) for n in nodes])
        return cls.from_region(region, n_cell_lat, n_cell_lon, lats, lons)

    # -- access ------------------------------------------------------------

    def cell(self, row: int, col: int) -> Cell:
        return self._cells[row][col]

    def flat_cells(self) -> list[Cell]:
        """All cells in row-major order."""
        return [c for row in self._cells for c in row]

    def cell_index(self, row: int, col: int) -> int:
        """Flat index for (row, col)."""
        return row * self.n_cols + col

    def cell_from_flat(self, idx: int) -> Cell:
        r, c = divmod(idx, self.n_cols)
        return self._cells[r][c]

    def cell_containing_point(self, lat: float, lon: float) -> Optional[Cell]:
        """Find the cell containing a (lat, lon) point."""
        for row in self._cells:
            for c in row:
                if c.lat_min <= lat <= c.lat_max and c.lon_min <= lon <= c.lon_max:
                    return c
        return None

    def neighbors(self, row: int, col: int, connectivity: str = "4") -> list[Cell]:
        """Return adjacent cells (4-connected or 8-connected)."""
        deltas_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        deltas_8 = deltas_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        deltas = deltas_8 if connectivity == "8" else deltas_4

        result = []
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                result.append(self._cells[nr][nc])
        return result

    # -- variance management -----------------------------------------------

    def initialize(self, initial_variance: float) -> None:
        """Set all cell point variances to κ (for t=0, before any observations)."""
        for c in self.flat_cells():
            c.point_variances = np.full(c.n_points, initial_variance)

    def update_from_model(self, field_model: Any) -> None:
        """Recompute cell variances from the model's predictive variance.

        Calls ``field_model.predict(query_points)`` on all grid points,
        then distributes the per-point variance back into cells.
        """
        qp = QueryPoints(lats=self.grid_lats, lons=self.grid_lons)
        pred = field_model.predict(qp)
        if pred.std is not None:
            all_var = pred.std ** 2
        else:
            all_var = np.zeros(len(self.grid_lats))

        for c in self.flat_cells():
            if c.n_points > 0:
                c.point_variances = all_var[c.point_indices].copy()

    def set_cell_variance(self, row: int, col: int, variance: float) -> None:
        """Set all point variances in a cell to a single value.

        Used during θ-simulation to mark visited cells as σ².
        """
        c = self._cells[row][col]
        c.point_variances = np.full(c.n_points, variance)

    def variance_array(self) -> np.ndarray:
        """Return flat array of cell mean variances in row-major order."""
        return np.array([c.variance for c in self.flat_cells()])

    def centroid_array(self) -> np.ndarray:
        """Return (C, D) array of cell centroids — D is 2 or 3."""
        return np.array([c.centroid for c in self.flat_cells()])

    def copy_variances(self) -> list[np.ndarray]:
        """Snapshot all per-point variance arrays (for θ-simulation rollback)."""
        return [c.point_variances.copy() for c in self.flat_cells()]

    def restore_variances(self, snapshot: list[np.ndarray]) -> None:
        """Restore from a snapshot created by :meth:`copy_variances`."""
        for c, arr in zip(self.flat_cells(), snapshot):
            c.point_variances = arr.copy()

    # -- distances ---------------------------------------------------------

    @staticmethod
    def _haversine_m(
        lat1_deg: float,
        lon1_deg: float,
        lat2_deg: float,
        lon2_deg: float,
        depth1_m: float | None = None,
        depth2_m: float | None = None,
    ) -> float:
        """Distance in meters, optionally including depth via Pythagorean combination."""
        r_earth = 6371e3  # meters
        lat1 = np.deg2rad(lat1_deg)
        lon1 = np.deg2rad(lon1_deg)
        lat2 = np.deg2rad(lat2_deg)
        lon2 = np.deg2rad(lon2_deg)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        horiz = float(r_earth * c)
        if depth1_m is not None and depth2_m is not None:
            dz = float(depth2_m) - float(depth1_m)
            return float(np.sqrt(horiz**2 + dz**2))
        return horiz

    def cell_distance(self, c1: Cell, c2: Cell) -> float:
        """Great-circle distance between cell centroids (meters)."""
        lat1, lon1 = c1.centroid
        lat2, lon2 = c2.centroid
        return self._haversine_m(lat1, lon1, lat2, lon2)
