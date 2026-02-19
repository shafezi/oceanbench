"""
Units normalization utilities.

Document conversions; preserve units in variable attributes.
"""

import xarray as xr


# Canonical units per variable (if conversion needed)
CANONICAL_UNITS = {
    "temp": "degC",
    "sal": "g/kg",
    "u": "m/s",
    "v": "m/s",
    "ssh": "m",
    "chl": "mg/m3",
    "no3": "mmol/m3",
    "o2": "mmol/m3",
    "po4": "mmol/m3",
}


def normalize_units(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure canonical units where applicable.
    Document any conversions in attributes.
    """
    out = ds.copy(deep=True)
    for v in out.data_vars:
        if v in CANONICAL_UNITS:
            target = CANONICAL_UNITS[v]
            if "units" not in out[v].attrs:
                out[v].attrs["units"] = target
            # Optionally convert if needed (e.g., Kelvin -> degC)
            # For now we just set the attribute; actual conversion would need cf-units
    return out
