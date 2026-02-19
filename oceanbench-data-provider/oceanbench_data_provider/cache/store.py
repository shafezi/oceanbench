"""
Cache read/write with metadata bundle (request.json, dataset_card.json, processing.json, log).
Uses Zarr for chunked storage suitable for large data and partial reads.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import xarray as xr

from oceanbench_data_provider.cache.keys import make_cache_key
from oceanbench_data_provider.catalog import describe

PIPELINE_VERSION = "v1"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.oceanbench/cache")
LOGGER = logging.getLogger("oceanbench.cache")


def _get_cache_dir(config: dict) -> Path:
    d = config.get("cache_dir", os.environ.get("OCEANBENCH_CACHE_DIR", DEFAULT_CACHE_DIR))
    return Path(os.path.expanduser(str(d)))


class CacheStore:
    """Manages cache directory, read/write of Zarr datasets, and metadata bundle."""

    def __init__(self, cache_dir: Optional[str] = None):
        cfg = {"cache_dir": cache_dir} if cache_dir else {}
        self.root = _get_cache_dir(cfg)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pipeline_version = PIPELINE_VERSION

    def _path(self, cache_key: str, suffix: str = "") -> Path:
        return self.root / cache_key / f"{cache_key}{suffix}"

    def _dir(self, cache_key: str) -> Path:
        return self.root / cache_key

    def load(self, cache_key: str) -> Optional[xr.Dataset]:
        """Load dataset from cache if it exists. Returns None if not found."""
        zarr_path = self._dir(cache_key) / f"{cache_key}.zarr"
        if not zarr_path.exists():
            return None
        try:
            ds = xr.open_zarr(zarr_path)
            return ds.load()
        except Exception as e:
            LOGGER.warning("Failed to load cache %s: %s", cache_key, e)
            return None

    def save(
        self,
        cache_key: str,
        ds: xr.Dataset,
        request: dict[str, Any],
        processing_summary: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save dataset as Zarr and write metadata bundle."""
        d = self._dir(cache_key)
        d.mkdir(parents=True, exist_ok=True)

        zarr_path = d / f"{cache_key}.zarr"
        # Load into memory and clear encoding (avoids Blosc/Zarr v3 incompatibility)
        ds = ds.load()
        for v in list(ds.data_vars) + list(ds.coords):
            if v in ds and hasattr(ds[v], "encoding"):
                ds[v].encoding = {}
        try:
            ds.to_zarr(zarr_path, mode="w", consolidated=True)
        except TypeError as e:
            if "Blosc" in str(e) or "BytesBytesCodec" in str(e):
                nc_path = d / f"{cache_key}.nc"
                ds.to_netcdf(nc_path, mode="w")
                LOGGER.info("Saved to NetCDF (Zarr fallback): %s", nc_path)
            else:
                raise

        # Metadata bundle
        with open(d / "request.json", "w") as f:
            json.dump(request, f, indent=2)

        product_id = request.get("product_id", "")
        try:
            card = describe(product_id)
            with open(d / "dataset_card.json", "w") as f:
                json.dump(card.to_dict(), f, indent=2)
        except KeyError:
            pass

        proc = processing_summary or {}
        with open(d / "processing.json", "w") as f:
            json.dump(proc, f, indent=2)

        # Simple log
        with open(d / "log.txt", "w") as f:
            f.write(f"Cache key: {cache_key}\n")
            f.write(f"Product: {product_id}\n")
            f.write(f"Pipeline: {self.pipeline_version}\n")

        LOGGER.info("Cached to %s", d)

    def status(self, product_id: Optional[str] = None) -> dict[str, Any]:
        """Report cache status. Optionally filter by product_id."""
        entries = []
        for p in self.root.iterdir():
            if not p.is_dir():
                continue
            req_file = p / "request.json"
            if not req_file.exists():
                continue
            try:
                with open(req_file) as f:
                    req = json.load(f)
                if product_id and req.get("product_id") != product_id:
                    continue
                entries.append({"cache_key": p.name, "request": req})
            except Exception:
                pass
        return {"cache_dir": str(self.root), "entries": entries}
