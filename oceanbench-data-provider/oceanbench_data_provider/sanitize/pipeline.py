"""
Canonical cleaning steps orchestrator.

Pipeline: open -> normalize coords -> fill/mask -> standardize vars/units -> optional regrid -> save.
"""

from typing import Any, Optional

import numpy as np
import xarray as xr

from oceanbench_data_provider.sanitize.qc import apply_qc
from oceanbench_data_provider.sanitize.units import normalize_units
from oceanbench_data_provider.sanitize.regrid import regrid_to_target


# Longitude convention: [-180, 180]
LON_CONVENTION = "[-180, 180]"


def _normalize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Normalize to canonical dimensions: time, lat, lon, depth."""
    renames = {}
    for d in ds.dims:
        dlower = str(d).lower()
        if dlower in ("latitude", "lat", "nav_lat"):
            renames[d] = "lat"
        elif dlower in ("longitude", "lon", "lont", "nav_lon"):
            renames[d] = "lon"
        elif dlower in ("depth", "deptht", "depthu", "depthv", "lev", "layer"):
            renames[d] = "depth"
        elif "time" in dlower or d == "time":
            renames[d] = "time"

    ds = ds.rename(renames)

    # Longitude: convert 0-360 to -180 to 180
    if "lon" in ds.coords:
        lon = ds.coords["lon"]
        if lon.max() > 180 or lon.min() < -180:
            # Assume 0-360
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        # Sort by lon (for consistency)
        ds = ds.sortby("lon")

    # Latitude: monotonically increasing
    if "lat" in ds.coords:
        ds = ds.sortby("lat")

    # Ensure time is datetime64 where possible
    if "time" in ds.coords:
        try:
            if ds.time.dtype.kind != "M":
                ds["time"] = ds["time"].astype("datetime64[ns]")
        except (TypeError, ValueError):
            pass  # Keep as-is if conversion fails (e.g. cftime)

    return ds


def _canonical_variable_names(ds: xr.Dataset, var_mapping: dict[str, str]) -> xr.Dataset:
    """Rename provider variables to canonical names."""
    renames = {}
    for prov, can in var_mapping.items():
        if prov in ds:
            renames[prov] = can
    return ds.rename(renames)


def run_sanitization_pipeline(
    ds: xr.Dataset,
    request: dict[str, Any],
    pipeline_version: str = "v1",
) -> xr.Dataset:
    """
    Run the full sanitization pipeline and return canonical dataset.

    Steps:
    1. Normalize coordinates/dimensions
    2. Convert fill values to NaN; apply masks
    3. Standardize variable names + units
    4. Optional regrid to target grid
    5. Return with provenance attrs
    """
    var_mapping = request.get("_var_mapping", {})
    if not var_mapping and "variables" in request:
        # Adapter should have set _var_mapping; use identity for requested vars
        for v in request["variables"]:
            var_mapping[v] = v

    ds = _normalize_coordinates(ds)
    ds = apply_qc(ds)
    ds = _canonical_variable_names(ds, var_mapping)
    ds = normalize_units(ds)

    target_grid = request.get("target_grid", {})
    if target_grid:
        ds, regrid_info = regrid_to_target(ds, target_grid)
    else:
        regrid_info = {}

    # Provenance attrs
    ds.attrs["oceanbench_pipeline_version"] = pipeline_version
    ds.attrs["oceanbench_lon_convention"] = LON_CONVENTION
    ds.attrs["oceanbench_product_id"] = request.get("product_id", "")
    if regrid_info:
        ds.attrs["oceanbench_regrid"] = str(regrid_info)

    return ds
