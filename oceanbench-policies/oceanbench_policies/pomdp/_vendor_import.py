"""Stable importlib-based loader for vendored AdaptiveSamplingPOMCP modules.

All adapter code should import upstream objects through this module rather
than manipulating ``sys.path``.  This keeps the path surgery in one place
and avoids polluting the global module namespace.

Usage
-----
>>> from oceanbench_policies.pomdp._vendor_import import upstream
>>> BuildTree = upstream("pomcp.auxilliary", "BuildTree")
>>> POMCP = upstream("pomcp.pomcp", "POMCP")
"""

from __future__ import annotations

import importlib.util
import logging
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Root of the vendored repo (third_party/AdaptiveSamplingPOMCP/).
_VENDOR_ROOT = (
    Path(__file__).resolve().parents[4]  # oceanbench repo root
    / "third_party"
    / "AdaptiveSamplingPOMCP"
)

# Cache for already-loaded modules keyed by dotted name.
_module_cache: dict[str, types.ModuleType] = {}


def _resolve_path(dotted_name: str) -> Path:
    """Convert ``"pomcp.auxilliary"`` to ``<vendor_root>/pomcp/auxilliary.py``."""
    parts = dotted_name.split(".")
    candidate = _VENDOR_ROOT / Path(*parts)
    # Try as a file first, then as package __init__.
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    init = candidate / "__init__.py"
    if init.is_file():
        return init
    raise FileNotFoundError(
        f"Cannot locate vendored module {dotted_name!r} under {_VENDOR_ROOT}"
    )


def _load_module(dotted_name: str) -> types.ModuleType:
    """Load a vendored module by dotted name using importlib."""
    if dotted_name in _module_cache:
        return _module_cache[dotted_name]

    # Ensure parent packages are loaded first so relative references work.
    parts = dotted_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in _module_cache:
            try:
                _load_module(parent)
            except FileNotFoundError:
                pass  # Some intermediate dirs may not be real packages.

    path = _resolve_path(dotted_name)
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {dotted_name} at {path}")

    module = importlib.util.module_from_spec(spec)

    # Register in sys.modules under the dotted name so that internal
    # relative imports within the vendored code resolve correctly.
    import sys
    sys.modules[dotted_name] = module
    _module_cache[dotted_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # Roll back on failure.
        sys.modules.pop(dotted_name, None)
        _module_cache.pop(dotted_name, None)
        raise

    return module


def upstream(dotted_module: str, attr: str | None = None) -> Any:
    """Import an attribute (or whole module) from the vendored upstream repo.

    Parameters
    ----------
    dotted_module:
        Dotted module path relative to the vendor root, e.g.
        ``"pomcp.pomcp"`` or ``"sample_sim.planning.pomcp_rollout_allocation.fixed"``.
    attr:
        Optional attribute name to extract from the module.  If *None* the
        module object itself is returned.

    Returns
    -------
    The requested module or attribute.
    """
    mod = _load_module(dotted_module)
    if attr is None:
        return mod
    try:
        return getattr(mod, attr)
    except AttributeError:
        raise ImportError(
            f"Module {dotted_module!r} has no attribute {attr!r}"
        ) from None


def vendor_available() -> bool:
    """Return True if the vendored upstream repo is present on disk."""
    return (_VENDOR_ROOT / "pomcp" / "pomcp.py").is_file()
