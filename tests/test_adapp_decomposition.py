"""Tests for the AdaPP cell decomposition module."""

import numpy as np
import pytest

from oceanbench_core.cell_decomposition import Cell, CellGrid


REGION = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0}


@pytest.fixture
def uniform_grid():
    """A 10x10 uniform point grid over [0,10] x [0,10]."""
    lats = np.linspace(0.5, 9.5, 10)
    lons = np.linspace(0.5, 9.5, 10)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
    return lat_mesh.ravel(), lon_mesh.ravel()


@pytest.fixture
def cell_grid(uniform_grid):
    lats, lons = uniform_grid
    return CellGrid.from_region(REGION, n_cell_lat=5, n_cell_lon=5,
                                grid_lats=lats, grid_lons=lons)


class TestCellGrid:
    def test_cell_count(self, cell_grid):
        assert cell_grid.n_rows == 5
        assert cell_grid.n_cols == 5
        assert len(cell_grid.flat_cells()) == 25

    def test_all_points_assigned(self, cell_grid, uniform_grid):
        """Every grid point should belong to exactly one cell."""
        total = sum(c.n_points for c in cell_grid.flat_cells())
        assert total == len(uniform_grid[0])

    def test_no_duplicate_assignments(self, cell_grid, uniform_grid):
        """No point index should appear in two cells."""
        seen = set()
        for c in cell_grid.flat_cells():
            for idx in c.point_indices:
                assert idx not in seen, f"Point {idx} assigned to multiple cells"
                seen.add(idx)

    def test_initialize_sets_kappa(self, cell_grid):
        cell_grid.initialize(initial_variance=2.5)
        for c in cell_grid.flat_cells():
            if c.n_points > 0:
                assert np.allclose(c.point_variances, 2.5)
                assert c.variance == pytest.approx(2.5)


class TestCellVariance:
    def test_eq6_mean_variance(self):
        """Eq. 6: cell variance = mean of point variances."""
        c = Cell(
            row=0, col=0, lat_min=0, lat_max=1, lon_min=0, lon_max=1,
            point_indices=np.arange(4),
            point_lats=np.array([0.1, 0.2, 0.3, 0.4]),
            point_lons=np.array([0.1, 0.2, 0.3, 0.4]),
            point_variances=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        assert c.variance == pytest.approx(2.5)

    def test_eq7_weighted_centroid(self):
        """Eq. 7: centroid = variance-weighted average of locations."""
        c = Cell(
            row=0, col=0, lat_min=0, lat_max=10, lon_min=0, lon_max=10,
            point_indices=np.arange(2),
            point_lats=np.array([2.0, 8.0]),
            point_lons=np.array([3.0, 7.0]),
            point_variances=np.array([1.0, 3.0]),
        )
        lat, lon = c.centroid
        # Weighted: lat = (1*2 + 3*8)/4 = 26/4 = 6.5
        assert lat == pytest.approx(6.5)
        # lon = (1*3 + 3*7)/4 = 24/4 = 6.0
        assert lon == pytest.approx(6.0)

    def test_zero_variance_centroid_fallback(self):
        """If all variances are zero, centroid should be geometric mean."""
        c = Cell(
            row=0, col=0, lat_min=0, lat_max=10, lon_min=0, lon_max=10,
            point_indices=np.arange(2),
            point_lats=np.array([2.0, 8.0]),
            point_lons=np.array([3.0, 7.0]),
            point_variances=np.array([0.0, 0.0]),
        )
        lat, lon = c.centroid
        assert lat == pytest.approx(5.0)
        assert lon == pytest.approx(5.0)


class TestCellGridOperations:
    def test_neighbors_4(self, cell_grid):
        """Corner cell should have 2 neighbors, edge 3, interior 4."""
        assert len(cell_grid.neighbors(0, 0, "4")) == 2
        assert len(cell_grid.neighbors(0, 2, "4")) == 3
        assert len(cell_grid.neighbors(2, 2, "4")) == 4

    def test_neighbors_8(self, cell_grid):
        assert len(cell_grid.neighbors(0, 0, "8")) == 3
        assert len(cell_grid.neighbors(2, 2, "8")) == 8

    def test_set_cell_variance(self, cell_grid):
        cell_grid.initialize(1.0)
        cell_grid.set_cell_variance(0, 0, 0.01)
        assert cell_grid.cell(0, 0).variance == pytest.approx(0.01)
        # Other cells unchanged.
        assert cell_grid.cell(1, 1).variance == pytest.approx(1.0)

    def test_copy_restore_variances(self, cell_grid):
        cell_grid.initialize(1.0)
        snapshot = cell_grid.copy_variances()
        cell_grid.set_cell_variance(0, 0, 999.0)
        assert cell_grid.cell(0, 0).variance == pytest.approx(999.0)
        cell_grid.restore_variances(snapshot)
        assert cell_grid.cell(0, 0).variance == pytest.approx(1.0)

    def test_variance_array_shape(self, cell_grid):
        cell_grid.initialize(1.0)
        arr = cell_grid.variance_array()
        assert arr.shape == (25,)

    def test_centroid_array_shape(self, cell_grid):
        cell_grid.initialize(1.0)
        arr = cell_grid.centroid_array()
        assert arr.shape == (25, 2)

    def test_cell_containing_point(self, cell_grid):
        c = cell_grid.cell_containing_point(1.0, 1.0)
        assert c is not None
        assert c.row == 0
        assert c.col == 0
