"""
Copernicus Marine BGC adapter.

- Reanalysis: GLOBAL_MULTIYEAR_BGC_001_029 (1993-2025)
- Analysis: GLOBAL_ANALYSISFORECAST_BGC_001_028 (2021-now + 10-day forecast)

Requires Copernicus Marine credentials (copernicusmarine login).
"""

from __future__ import annotations

from typing import Any, Optional

import xarray as xr

from oceanbench_data_provider.sources.base import SourceAdapter

# Copernicus Marine open_dataset expects dataset_id (not product_id)
COPERNICUS_BGC_DATASETS = {
    "copernicus_bgc_reanalysis_001_029": "cmems_mod_glo_bgc_my_0.25deg_P1D-m",  # GLOBAL_MULTIYEAR_BGC_001_029 daily
    "copernicus_bgc_analysis_001_028": "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m",  # GLOBAL_ANALYSISFORECAST_BGC_001_028
}

# BGC reanalysis has chl, no3, o2, po4, si, nppv - NOT thetao/so
CANONICAL_TO_CMEMS_BGC = {
    "chl": "chl",
    "no3": "no3",
    "o2": "o2",
    "po4": "po4",
    "si": "si",
    "nppv": "nppv",
}


class CopernicusBgcAdapter(SourceAdapter):
    """Copernicus Marine BGC adapter using Copernicus Marine Toolbox."""

    def __init__(self, product_id: str, config: dict[str, Any]):
        self.product_id = product_id
        self.config = config
        self.dataset_id = COPERNICUS_BGC_DATASETS.get(
            product_id, "GLOBAL_MULTIYEAR_BGC_001_029"
        )

    def fetch_subset(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
        depth_opts: dict[str, Any],
    ) -> Optional[xr.Dataset]:
        try:
            import copernicusmarine
        except ImportError as e:
            raise ImportError(
                "Copernicus Marine BGC requires 'copernicusmarine' package. "
                "Install with: pip install copernicusmarine"
            ) from e

        lon_b, lat_b = region.get("lon", [0, 360]), region.get("lat", [-90, 90])
        var_mapping = {}
        cmems_vars = []
        for v in variables:
            c = CANONICAL_TO_CMEMS_BGC.get(v, v)
            cmems_vars.append(c)
            var_mapping[c] = v

        try:
            ds = copernicusmarine.open_dataset(
                dataset_id=self.dataset_id,
                minimum_longitude=float(lon_b[0]),
                maximum_longitude=float(lon_b[1]),
                minimum_latitude=float(lat_b[0]),
                maximum_latitude=float(lat_b[1]),
                start_datetime=time[0],
                end_datetime=time[1],
                variables=cmems_vars if cmems_vars else None,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "credential" in err_msg or "login" in err_msg or "auth" in err_msg:
                raise RuntimeError(
                    "Copernicus Marine credentials required. "
                    "Run: copernicusmarine login"
                ) from e
            raise RuntimeError(f"Copernicus BGC fetch failed: {e}") from e

        if ds is None:
            return None

        ds.attrs["_var_mapping"] = var_mapping
        return ds

    def estimate_size(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
    ) -> dict[str, Any]:
        lon_b, lat_b = region.get("lon", [0, 360]), region.get("lat", [-90, 90])
        n_lat = int((lat_b[1] - lat_b[0]) / 0.25) or 1
        n_lon = int((lon_b[1] - lon_b[0]) / 0.25) or 1
        from datetime import datetime
        start = datetime.fromisoformat(time[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(time[1].replace("Z", "+00:00"))
        n_days = (end - start).days or 1
        n_depth = 75
        total = n_lat * n_lon * n_days * n_depth * len(variables) * 4
        return {
            "estimate_bytes": total,
            "estimate_mb": total / (1024 * 1024),
            "note": "Approximate; uses 1/4° and daily resolution.",
        }
