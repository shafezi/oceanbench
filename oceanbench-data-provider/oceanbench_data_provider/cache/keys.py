"""
Deterministic cache keys/hashes from request spec.
"""

import hashlib
import json
from typing import Any


def _canonicalize(obj: Any) -> str:
    """Produce deterministic JSON-like string for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def make_cache_key(request: dict[str, Any], pipeline_version: str = "v1") -> str:
    """
    Compute a deterministic cache key from the request and pipeline version.

    Components: product_id, region, time, variables, depth_opts, target_grid, pipeline_version.
    """
    components = {
        "product_id": request.get("product_id", ""),
        "region": request.get("region", {}),
        "time": request.get("time", ()),
        "variables": sorted(request.get("variables", [])),
        "depth_opts": request.get("depth_opts", {}),
        "target_grid": request.get("target_grid", {}),
        "pipeline_version": pipeline_version,
    }
    canonical = _canonicalize(components)
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return f"{request.get('product_id', 'unknown')}_{h[:16]}"
