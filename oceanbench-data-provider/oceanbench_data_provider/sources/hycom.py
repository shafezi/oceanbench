"""
HYCOM adapter: OPeNDAP/THREDDS access.

- Reanalysis: GLBv0.08/expt_53.X (1994-2015)
- Analysis: 56.3, 57.2, 92.8, 57.7, 92.9, 93 with date-specific experiment paths
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

from oceanbench_data_provider.sources.base import SourceAdapter

HYCOM_OPENDAP_BASE = "https://tds.hycom.org/thredds/dodsC"
HYCOM_NCSS_BASE = "https://ncss.hycom.org/thredds/ncss/grid"

# HYCOM variable name -> canonical
HYCOM_VAR_MAP = {
    "temp": "water_temp",
    "thetao": "water_temp",
    "water_temperature": "water_temp",
    "water_temp": "water_temp",
    "sal": "salinity",
    "so": "salinity",
    "salinity": "salinity",
    "u": "water_u",
    "uo": "water_u",
    "water_u": "water_u",
    "v": "water_v",
    "vo": "water_v",
    "water_v": "water_v",
    "ssh": "surf_el",
    "zos": "surf_el",
    "surf_el": "surf_el",
}

# Reverse: canonical -> possible HYCOM names
CANONICAL_TO_HYCOM = {
    "temp": ["water_temp", "water_temperature"],
    "sal": ["salinity"],
    "u": ["water_u"],
    "v": ["water_v"],
    "ssh": ["surf_el"],
}

# HYCOM variables with non-CF time units ("hours since analysis") that break xarray decode
# Dropped on open to avoid ValueError
HYCOM_DROP_VARS = ["tau", "tau_u", "tau_v"]

# Experiment path by product
HYCOM_EXPERIMENTS = {
    "hycom_glbv0.08_reanalysis_53x": "GLBv0.08/expt_53.X",
    "hycom_glbv0.08_analysis_56_3": "GLBv0.08/expt_56.3",
    "hycom_glbv0.08_analysis_57_2": "GLBv0.08/expt_57.2",
    "hycom_glbv0.08_analysis_92_8": "GLBv0.08/expt_92.8",
    "hycom_glbv0.08_analysis_57_7": "GLBv0.08/expt_57.7",
    "hycom_glbv0.08_analysis_92_9": "GLBv0.08/expt_92.9",
    "hycom_glbv0.08_analysis_93": "GLBv0.08/expt_93.0",
}


def _get_experiment_path(product_id: str) -> str:
    if product_id in HYCOM_EXPERIMENTS:
        return HYCOM_EXPERIMENTS[product_id]
    raise ValueError(f"Unknown HYCOM product: {product_id}")


def _build_hycom_url(product_id: str, year: int) -> str:
    exp = _get_experiment_path(product_id)
    return f"{HYCOM_OPENDAP_BASE}/{exp}/data/{year}"


def _time_to_years(time: tuple[str, str]) -> list[int]:
    start, end = pd.Timestamp(time[0]), pd.Timestamp(time[1])
    return list(range(int(start.year), int(end.year) + 1))


class HycomAdapter(SourceAdapter):
    """HYCOM OPeNDAP/THREDDS adapter."""

    def __init__(self, product_id: str, config: dict[str, Any]):
        self.product_id = product_id
        self.config = config

    def fetch_subset(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
        depth_opts: dict[str, Any],
    ) -> Optional[xr.Dataset]:
        lon_b, lat_b = region.get("lon", [0, 360]), region.get("lat", [-90, 90])
        years = _time_to_years(time)
        datasets = []
        var_mapping = {}  # provider -> canonical

        for var in variables:
            hycom_names = CANONICAL_TO_HYCOM.get(var, [var])
            var_mapping[var] = hycom_names[0] if hycom_names else var

        for year in years:
            url = _build_hycom_url(self.product_id, year)
            try:
                # Drop vars with "hours since analysis" time units (non-CF, breaks xarray decode)
                ds = xr.open_dataset(url, drop_variables=HYCOM_DROP_VARS)
                # Select variables that exist
                sel_vars = [v for v in var_mapping.values() if v in ds]
                if not sel_vars:
                    continue
                ds = ds[sel_vars]
                # Subset by region (HYCOM may use Lat, Lon or lat, lon)
                lat_dim = "Lat" if "Lat" in ds.dims else "lat"
                lon_dim = "Lon" if "Lon" in ds.dims else "lon"
                if lat_dim in ds.dims and lon_dim in ds.dims:
                    ds = ds.sel(
                        {lat_dim: slice(lat_b[0], lat_b[1]), lon_dim: slice(lon_b[0], lon_b[1])}
                    )
                # Subset by time
                if "time" in ds.dims:
                    ds = ds.sel(time=slice(time[0], time[1]))
                datasets.append(ds)
            except Exception as e:
                raise RuntimeError(f"HYCOM fetch failed for {url}: {e}") from e

        if not datasets:
            return None

        combined = xr.concat(datasets, dim="time")
        combined.attrs["_var_mapping"] = {v: k for k, v in var_mapping.items()}
        return combined

    def estimate_size(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
    ) -> dict[str, Any]:
        lon_b, lat_b = region.get("lon", [0, 360]), region.get("lat", [-90, 90])
        years = _time_to_years(time)
        n_years = len(years)
        n_vars = len(variables)
        # Rough estimate: 1/12 deg ~ 9km, 3-hourly
        n_lat = int((lat_b[1] - lat_b[0]) / (1 / 12)) or 1
        n_lon = int((lon_b[1] - lon_b[0]) / (1 / 12)) or 1
        n_time = n_years * 365 * 8  # 8 steps per day
        n_depth = 41
        bytes_per_value = 4
        total = n_lat * n_lon * n_time * n_depth * n_vars * bytes_per_value
        return {
            "estimate_bytes": total,
            "estimate_mb": total / (1024 * 1024),
            "note": "Approximate; actual depends on available timesteps.",
        }
