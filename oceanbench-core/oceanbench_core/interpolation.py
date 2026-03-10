from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import xarray as xr

from .types import QueryPoints


BoundsMode = Literal["nan", "clip", "error"]


def _require_coords(
    ds: xr.Dataset,
    *,
    require_time: bool,
    require_depth: bool,
) -> None:
    missing = []
    for name in ("lat", "lon"):
        if name not in ds.coords:
            missing.append(name)
    if require_time and "time" not in ds.coords:
        missing.append("time")
    if require_depth and "depth" not in ds.coords:
        missing.append("depth")

    if missing:
        raise ValueError(
            f"Dataset is missing required coordinate(s): {', '.join(missing)}"
        )


def interpolate_dataset(
    ds: xr.Dataset,
    query_points: QueryPoints,
    variable: str,
    *,
    method: str = "linear",
    bounds_mode: BoundsMode = "nan",
) -> xr.DataArray:
    """
    Safely interpolate a variable from a provider-returned Dataset.

    Parameters
    ----------
    ds:
        Canonical `xarray.Dataset` returned by the Data Provider.
    query_points:
        Locations (and optional times/depths) at which to interpolate.
    variable:
        Name of the data variable in `ds` to query.
    method:
        Interpolation method passed to `xarray.Dataset.interp`.
    bounds_mode:
        How to handle out-of-range coordinates:

        - "nan": allow interpolation to return NaNs outside the domain.
        - "clip": clip query coordinates to the dataset's min/max on each axis.
        - "error": raise a ValueError if any query is outside the domain.
    """

    if variable not in ds.data_vars:
        raise KeyError(f"Variable {variable!r} not present in dataset.")

    require_time = query_points.times is not None
    require_depth = query_points.depths is not None
    _require_coords(
        ds,
        require_time=require_time,
        require_depth=require_depth,
    )

    lat = np.asarray(query_points.lats, dtype=float)
    lon = np.asarray(query_points.lons, dtype=float)

    coords: dict[str, xr.DataArray] = {
        "lat": xr.DataArray(lat, dims=("points",)),
        "lon": xr.DataArray(lon, dims=("points",)),
    }

    if query_points.times is not None:
        times = np.asarray(query_points.times)
        coords["time"] = xr.DataArray(times, dims=("points",))

    if query_points.depths is not None:
        depths = np.asarray(query_points.depths, dtype=float)
        coords["depth"] = xr.DataArray(depths, dims=("points",))

    if bounds_mode == "clip":
        for name, arr in list(coords.items()):
            base = ds.coords[name]
            if np.issubdtype(base.dtype, np.datetime64):
                v_dt = np.asarray(arr.values).astype("datetime64[ns]")
                v_int = v_dt.astype("int64")
                bmin = base.min().values.astype("datetime64[ns]").astype("int64")
                bmax = base.max().values.astype("datetime64[ns]").astype("int64")
                v_int = np.clip(v_int, bmin, bmax)
                coords[name] = xr.DataArray(
                    v_int.astype("datetime64[ns]"),
                    dims=("points",),
                )
            else:
                v = arr.values.astype(float)
                v = np.clip(v, float(base.min()), float(base.max()))
                coords[name] = xr.DataArray(v, dims=("points",))
    elif bounds_mode == "error":
        for name, arr in coords.items():
            base = ds.coords[name]
            if np.issubdtype(base.dtype, np.datetime64):
                v_int = np.asarray(arr.values).astype("datetime64[ns]").astype("int64")
                bmin = base.min().values.astype("datetime64[ns]").astype("int64")
                bmax = base.max().values.astype("datetime64[ns]").astype("int64")
                if (v_int < bmin).any() or (v_int > bmax).any():
                    raise ValueError(
                        f"Query coordinate for {name!r} is outside dataset bounds."
                    )
            else:
                v = arr.values.astype(float)
                if (v < float(base.min())).any() or (v > float(base.max())).any():
                    raise ValueError(
                        f"Query coordinate for {name!r} is outside dataset bounds."
                    )
    elif bounds_mode != "nan":
        raise ValueError(
            f"Unknown bounds_mode {bounds_mode!r}; expected 'nan', 'clip', or 'error'."
        )

    da = ds[variable].interp(
        {k: v for k, v in coords.items()},
        method=method,
    )

    # Standardize the name of the point dimension.
    if "points" not in da.dims and "points" in coords["lat"].dims:
        # xarray may preserve an existing dimension name; in that case we
        # simply leave it as-is. For now we rely on the fact that the
        # returned DataArray is indexed in the same order as the input
        # `QueryPoints`.
        pass

    return da

