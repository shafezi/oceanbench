"""
Predefined geographic regions (lon/lat bounds) for common oceanographic areas.

Users can select a region and use it with any product, time window, and variables.
"""

from dataclasses import dataclass


@dataclass
class Region:
    """Geographic region with longitude and latitude bounds."""
    
    id: str
    name: str
    region: dict[str, list[float]]  # {"lon": [min, max], "lat": [min, max]}
    description: str = ""


_REGIONS: dict[str, Region] = {
    "monterey_bay": Region(
        id="monterey_bay",
        name="Monterey Bay",
        region={"lon": [-122.5, -121.5], "lat": [36.4, 37.0]},
        description="Monterey Bay, California - Central California coast",
    ),
    "gulf_of_mexico": Region(
        id="gulf_of_mexico",
        name="Gulf of Mexico",
        region={"lon": [-98, -80], "lat": [18, 31]},
        description="Gulf of Mexico - Western Atlantic basin",
    ),
    "juan_de_fuca_ridge": Region(
        id="juan_de_fuca_ridge",
        name="Juan de Fuca Ridge",
        region={"lon": [-130, -125], "lat": [44, 49]},
        description="Juan de Fuca Ridge - Pacific spreading center off Pacific Northwest",
    ),
    "gulf_of_maine": Region(
        id="gulf_of_maine",
        name="Gulf of Maine",
        region={"lon": [-71, -66], "lat": [41, 45]},
        description="Gulf of Maine - Northwest Atlantic, between Cape Cod and Nova Scotia",
    ),
    "gulf_of_alaska": Region(
        id="gulf_of_alaska",
        name="Gulf of Alaska",
        region={"lon": [-160, -140], "lat": [54, 60]},
        description="Gulf of Alaska - Northeast Pacific, south of Alaska",
    ),
    "strait_of_juan_de_fuca": Region(
        id="strait_of_juan_de_fuca",
        name="Strait of Juan de Fuca",
        region={"lon": [-125, -123], "lat": [48, 49]},
        description="Strait of Juan de Fuca - Between Vancouver Island and Washington State",
    ),
    "gulf_of_california": Region(
        id="gulf_of_california",
        name="Gulf of California",
        region={"lon": [-115, -107], "lat": [23, 32]},
        description="Gulf of California (Sea of Cortez) - Between Baja California and mainland Mexico",
    ),
    "chesapeake_bay": Region(
        id="chesapeake_bay",
        name="Chesapeake Bay",
        region={"lon": [-77, -75.5], "lat": [36.5, 39.5]},
        description="Chesapeake Bay - Largest estuary in the United States, Mid-Atlantic",
    ),
    "north_atlantic_ocean": Region(
        id="north_atlantic_ocean",
        name="North Atlantic Ocean",
        region={"lon": [-80, -20], "lat": [30, 60]},
        description="North Atlantic Ocean - Broad region from US East Coast to Europe",
    ),
}


def list_regions() -> list[str]:
    """List available region IDs."""
    return sorted(_REGIONS.keys())


def get_region(region_id: str) -> Region:
    """Get region by ID."""
    if region_id not in _REGIONS:
        raise KeyError(f"Unknown region: {region_id}. Available: {list_regions()}")
    return _REGIONS[region_id]


def get_region_bounds(region_id: str) -> dict[str, list[float]]:
    """Get just the lon/lat bounds for a region."""
    return get_region(region_id).region
