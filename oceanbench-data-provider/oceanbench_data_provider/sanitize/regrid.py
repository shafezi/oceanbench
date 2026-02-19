"""
Regridding wrappers (e.g., xESMF integration).

Supports optional target lat-lon rectilinear grid.
Method: bilinear (default), nearest, conservative.
"""

from typing import Any, Optional, Tuple

import numpy as np
import xarray as xr


def regrid_to_target(
    ds: xr.Dataset,
    target_grid: dict[str, Any],
) -> Tuple[xr.Dataset, dict[str, Any]]:
    """
    Regrid dataset to target rectilinear lat-lon grid.

    target_grid: e.g. {"lat_res": 0.1, "lon_res": 0.1} or {"lat": [...], "lon": [...]}
    method: bilinear | nearest | conservative

    Returns (regridded ds, info dict).
    """
    method = target_grid.get("method", "bilinear")

    try:
        import xesmf
    except ImportError:
        # Fallback: simple nearest-neighbor regrid using scipy
        return _regrid_nearest_fallback(ds, target_grid, method)

    # Build target grid
    if "lat_res" in target_grid and "lon_res" in target_grid:
        lat = np.arange(
            float(ds.lat.min()),
            float(ds.lat.max()) + target_grid["lat_res"] / 2,
            target_grid["lat_res"],
        )
        lon = np.arange(
            float(ds.lon.min()),
            float(ds.lon.max()) + target_grid["lon_res"] / 2,
            target_grid["lon_res"],
        )
    elif "lat" in target_grid and "lon" in target_grid:
        lat = np.asarray(target_grid["lat"])
        lon = np.asarray(target_grid["lon"])
    else:
        return ds, {"regrid": False, "reason": "No target grid specified"}

    ds_out = xr.Dataset(
        {"lat": (["lat"], lat), "lon": (["lon"], lon)}
    )

    regridder = xesmf.Regridder(ds, ds_out, method, ignore_degenerate=True)
    out_vars = {}
    for v in ds.data_vars:
        if "lat" in ds[v].dims and "lon" in ds[v].dims:
            out_vars[v] = regridder(ds[v])
        else:
            out_vars[v] = ds[v]
    ds_regrid = xr.Dataset(out_vars)
    ds_regrid.attrs.update(ds.attrs)
    info = {"method": method, "target_lat_res": float(np.diff(lat).mean()) if len(lat) > 1 else None, "target_lon_res": float(np.diff(lon).mean()) if len(lon) > 1 else None}
    return ds_regrid, info


def _regrid_nearest_fallback(
    ds: xr.Dataset,
    target_grid: dict[str, Any],
    method: str,
) -> Tuple[xr.Dataset, dict[str, Any]]:
    """Simple nearest-neighbor regrid when xESMF not available."""
    from scipy.interpolate import NearestNDInterpolator

    if "lat_res" not in target_grid or "lon_res" not in target_grid:
        return ds, {"regrid": False, "reason": "xESMF not installed; need lat_res/lon_res for fallback"}

    lat_res = target_grid["lat_res"]
    lon_res = target_grid["lon_res"]
    lat_new = np.arange(float(ds.lat.min()), float(ds.lat.max()) + lat_res / 2, lat_res)
    lon_new = np.arange(float(ds.lon.min()), float(ds.lon.max()) + lon_res / 2, lon_res)
    LON, LAT = np.meshgrid(lon_new, lat_new)
    points_out = np.column_stack([LAT.ravel(), LON.ravel()])

    out_vars = {}
    for v in ds.data_vars:
        if "lat" in ds[v].dims and "lon" in ds[v].dims:
            da = ds[v]
            if "time" in da.dims:
                out_list = []
                for t in da.time:
                    pts = np.column_stack([da.lat.values.ravel(), da.lon.values.ravel()])
                    vals = da.sel(time=t).values.ravel()
                    valid = ~np.isnan(vals)
                    if valid.any():
                        nn = NearestNDInterpolator(pts[valid], vals[valid])
                        out_list.append(nn(points_out).reshape(LAT.shape))
                    else:
                        out_list.append(np.full(LAT.shape, np.nan))
                out_arr = np.stack(out_list)
                out_vars[v] = (["time", "lat", "lon"], out_arr)
            else:
                pts = np.column_stack([da.lat.values.ravel(), da.lon.values.ravel()])
                vals = da.values.ravel()
                valid = ~np.isnan(vals)
                if valid.any():
                    nn = NearestNDInterpolator(pts[valid], vals[valid])
                    out_vals = nn(points_out).reshape(LAT.shape)
                else:
                    out_vals = np.full(LAT.shape, np.nan)
                out_vars[v] = (["lat", "lon"], out_vals)
        else:
            out_vars[v] = ds[v]

    ds_out = xr.Dataset(
        out_vars,
        coords={"lat": lat_new, "lon": lon_new, **{k: v for k, v in ds.coords.items() if k not in ("lat", "lon")}},
    )
    return ds_out, {"method": "nearest_fallback", "lat_res": lat_res, "lon_res": lon_res}
