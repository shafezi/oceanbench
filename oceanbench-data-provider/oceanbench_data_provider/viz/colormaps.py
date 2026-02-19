"""
Colormaps for ocean variables (paper-style conventions).

Uses cmocean when available (Thyng et al., Oceanography 2016); otherwise
falls back to matplotlib colormaps that match common oceanography usage.
"""

from typing import Any, Union

# Matplotlib fallbacks (no extra dependency): match common paper conventions
# - temp: thermal (cmocean when available; else RdBu_r = blue cold → red warm)
# - sal: white → blue (haline)
# - u/v: signed velocity (diverging)
# - ssh: diverging (anomalies)
# - chl: green (algae)
# - BGC: sensible defaults
VAR_CMAP_MATPLOTLIB: dict[str, str] = {
    "temp": "plasma",       # thermal-like fallback
    "sal": "YlGnBu",        # haline-like: light to blue
    "u": "RdBu_r",         # signed velocity
    "v": "RdBu_r",
    "ssh": "RdBu_r",       # diverging for anomalies
    "chl": "YlGn",         # chlorophyll green
    "no3": "YlOrBr",       # nitrate
    "o2": "RdYlBu_r",      # oxygen (red = low)
    "po4": "YlOrBr",
    "si": "PuBu",
    "nppv": "Greens",
}


def get_cmap(var: str) -> Union[str, Any]:
    """
    Return the recommended colormap for an ocean variable.

    Uses cmocean (thermal, haline, balance, algae, etc.) if installed;
    otherwise uses matplotlib colormaps that match oceanography conventions.

    Args:
        var: Variable name (e.g. 'temp', 'sal', 'ssh', 'chl').

    Returns:
        Colormap name (str) or cmocean colormap object for use with
        matplotlib (e.g. plot(..., cmap=get_cmap('temp'))).
    """
    var_lower = var.lower().strip()
    try:
        import cmocean
    except ImportError:
        return VAR_CMAP_MATPLOTLIB.get(var_lower, "viridis")

    # cmocean maps (paper-standard when available)
    cmocean_map = {
        "temp": getattr(cmocean.cm, "thermal", None),
        "sal": getattr(cmocean.cm, "haline", None),
        "u": getattr(cmocean.cm, "balance", None),
        "v": getattr(cmocean.cm, "balance", None),
        "ssh": getattr(cmocean.cm, "balance", None),
        "chl": getattr(cmocean.cm, "algae", None),
        "density": getattr(cmocean.cm, "dense", None),
        "o2": getattr(cmocean.cm, "oxy", None),
    }
    if var_lower in cmocean_map and cmocean_map[var_lower] is not None:
        return cmocean_map[var_lower]
    return VAR_CMAP_MATPLOTLIB.get(var_lower, "viridis")
