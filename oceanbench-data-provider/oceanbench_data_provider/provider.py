"""
DataProvider: Public API providing a uniform interface across all ocean products.
"""

from __future__ import annotations

from typing import Any, Optional
import xarray as xr

from oceanbench_data_provider.catalog import describe, DatasetCard
from oceanbench_data_provider.cache.keys import make_cache_key
from oceanbench_data_provider.cache.store import CacheStore
from oceanbench_data_provider.sources.factory import get_adapter
from oceanbench_data_provider.sanitize.pipeline import run_sanitization_pipeline


class DataProvider:
    """
    Unified data provider for ocean products.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Args:
            config: Optional config dict. Keys: cache_dir, overwrite, etc.
                   If None, loads from default config path.
        """
        self.config = config or {}
        self._store = CacheStore(self.config.get("cache_dir"))

    def list_products(self) -> list[str]:
        """List available product IDs."""
        from oceanbench_data_provider.catalog import list_products
        return list_products()

    def describe(self, product_id: str) -> DatasetCard:
        """Return Dataset Card for the product."""
        return describe(product_id)

    def subset(
        self,
        product_id: str,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
        depth_opts: Optional[dict[str, Any]] = None,
        target_grid: Optional[dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> xr.Dataset:
        """
        Subset data: fetches (if needed), sanitizes, caches, and returns xarray.Dataset.

        Args:
            product_id: Product identifier (e.g., hycom_glbv0.08_reanalysis_53x).
            region: {"lon": [lon_min, lon_max], "lat": [lat_min, lat_max]}.
            time: (start, end) as ISO strings.
            variables: List of canonical variable names (e.g., temp, sal, u, v, ssh).
            depth_opts: Optional depth selection (e.g., {"min": 0, "max": 200}).
            target_grid: Optional regridding target (e.g., {"lat_res": 0.1, "lon_res": 0.1}).
            overwrite: If True, re-fetch even when cache exists.

        Returns:
            xarray.Dataset in canonical format.
        """
        request = {
            "product_id": product_id,
            "region": region,
            "time": time,
            "variables": variables,
            "depth_opts": depth_opts or {},
            "target_grid": target_grid or {},
        }
        cache_key = make_cache_key(request)

        # Check cache first (unless overwrite)
        if not overwrite:
            ds = self._store.load(cache_key)
            if ds is not None:
                return ds

        # Fetch via adapter and sanitize
        adapter = get_adapter(product_id, self.config)
        raw = adapter.fetch_subset(region, time, variables, depth_opts or {})

        if raw is None:
            raise RuntimeError(
                f"Adapter returned no data for product {product_id}. "
                "Check credentials (Copernicus) or availability."
            )

        # Merge var mapping from adapter output
        var_map = getattr(raw, "attrs", {}).get("_var_mapping", {})
        if var_map:
            request["_var_mapping"] = var_map  # provider_name -> canonical

        ds_sanitized = run_sanitization_pipeline(
            raw, request, self._store.pipeline_version
        )

        # Save to cache
        self._store.save(cache_key, ds_sanitized, request)

        return ds_sanitized

    def estimate_size(
        self,
        product_id: str,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
    ) -> dict[str, Any]:
        """
        Estimate download and cache size before fetching.
        """
        adapter = get_adapter(product_id, self.config)
        return adapter.estimate_size(region, time, variables)

    def cache_status(self, product_id: Optional[str] = None) -> dict[str, Any]:
        """
        Report cache status. If product_id given, filter to that product.
        """
        return self._store.status(product_id)
