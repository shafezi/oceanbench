"""Tests for truth querying (OceanTruthField and interpolation)."""
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def tiny_dataset():
    """Minimal xarray Dataset with lat, lon (and optional time) for truth query tests."""
    lat = np.linspace(24, 36, 13)
    lon = np.linspace(-86, -74, 13)
    data = 20.0 + 0.1 * np.random.standard_normal((13, 13))
    ds = xr.Dataset(
        {"temp": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )
    return ds


def test_interpolate_dataset(tiny_dataset, small_query_points):
    from oceanbench_core.interpolation import interpolate_dataset
    da = interpolate_dataset(tiny_dataset, small_query_points, "temp", bounds_mode="clip")
    assert da.size == small_query_points.size
    assert np.all(np.isfinite(da.values))


def test_ocean_truth_field_query(tiny_dataset, small_query_points):
    from oceanbench_env import OceanTruthField
    truth = OceanTruthField(dataset=tiny_dataset, variable="temp")
    da = truth.query(small_query_points, bounds_mode="clip")
    assert da.size == small_query_points.size
    arr = truth.query_array(small_query_points, bounds_mode="clip")
    assert arr.shape == (small_query_points.size,)
    assert np.allclose(arr, da.values)
