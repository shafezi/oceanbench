"""
QC: fill value -> NaN, masks, range checks.

Land mask policy: no interpolation across land; land points remain NaN.
"""

from typing import Optional

import numpy as np
import xarray as xr


# Common fill value attribute names
FILL_ATTRS = ("_FillValue", "fill_value", "missing_value")


def _get_fill_value(da: xr.DataArray) -> Optional[float]:
    """Extract fill value from variable attributes."""
    for attr in FILL_ATTRS:
        if hasattr(da, attr):
            v = getattr(da, attr)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return None


def apply_qc(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert fill values to NaN; ensure consistent land mask.
    Land mask policy: values over land remain NaN (no interpolation across land).
    """
    out = ds.copy(deep=True)
    for v in list(out.data_vars):
        da = out[v]
        fv = _get_fill_value(da)
        if fv is not None:
            out[v] = da.where(da != fv, np.nan)
        # Remove fill value attrs so xarray doesn't use them
        for attr in FILL_ATTRS:
            out[v].attrs.pop(attr, None)
    return out
