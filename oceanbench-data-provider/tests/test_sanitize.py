"""Tests for sanitization pipeline."""

import numpy as np
import pytest
import xarray as xr

from oceanbench_data_provider.sanitize.qc import apply_qc
from oceanbench_data_provider.sanitize.pipeline import run_sanitization_pipeline, LON_CONVENTION


def test_qc_fill_value_to_nan():
    da = xr.DataArray([1.0, 2.0, -999.0], dims=["x"], attrs={"_FillValue": -999})
    ds = xr.Dataset({"v": da})
    out = apply_qc(ds)
    vals = out["v"].values
    assert np.isnan(vals[2])
    assert vals[0] == 1.0
    assert vals[1] == 2.0


def test_pipeline_normalizes_lon():
    ds = xr.Dataset(
        {"temp": (["time", "lat", "lon"], np.random.rand(2, 3, 4))},
        coords={
            "time": ["2020-01-01", "2020-01-02"],
            "lat": [30, 31, 32],
            "lon": [180, 181, 182, 183],  # 0-360 style
        },
    )
    req = {"product_id": "test", "variables": ["temp"], "_var_mapping": {"temp": "temp"}, "target_grid": {}}
    out = run_sanitization_pipeline(ds, req)
    assert out.lon.min().values >= -180 or out.lon.max().values <= 360
    assert out.attrs.get("oceanbench_lon_convention") == LON_CONVENTION
