"""
Scenario registry: presets for regions, time windows, and variables.

Enables reproducible benchmark scenarios. Scenarios can use predefined regions
(from regions.py) via get_region_bounds() so bounds are not duplicated.
"""

from dataclasses import dataclass
from typing import Any

from oceanbench_data_provider.scenarios.regions import get_region_bounds


@dataclass
class Scenario:
    """Scenario preset."""

    id: str
    name: str
    region: dict[str, list[float]]
    time: tuple[str, str]
    variables: list[str]
    product_id: str
    description: str = ""


_SCENARIOS: dict[str, Scenario] = {
    "gulf_stream_phy": Scenario(
        id="gulf_stream_phy",
        name="Gulf Stream (Physical)",
        region={"lon": [-80, -60], "lat": [30, 45]},
        time=("2014-01-01", "2014-01-31"),  # HYCOM reanalysis 53.X: 1994-2015 only
        variables=["temp", "sal", "u", "v", "ssh"],
        product_id="hycom_glbv0.08_reanalysis_53x",
        description="Gulf Stream region, 1 month, physical vars",
    ),
    "mediterranean_bgc": Scenario(
        id="mediterranean_bgc",
        name="Mediterranean (BGC)",
        region={"lon": [-6, 37], "lat": [30, 46]},
        time=("2020-06-01", "2020-06-30"),
        variables=["chl", "temp", "sal"],
        product_id="copernicus_bgc_reanalysis_001_029",
        description="Mediterranean, June 2020, BGC vars",
    ),
    "california_current": Scenario(
        id="california_current",
        name="California Current",
        region={"lon": [-130, -115], "lat": [30, 45]},
        time=("2015-07-01", "2015-07-14"),
        variables=["temp", "sal", "u", "v"],
        product_id="hycom_glbv0.08_reanalysis_53x",
        description="California Current, 2 weeks",
    ),
    # Scenarios using predefined regions (no duplicate bounds)
    "monterey_bay_phy": Scenario(
        id="monterey_bay_phy",
        name="Monterey Bay (Physical)",
        region=get_region_bounds("monterey_bay"),
        time=("2014-06-01", "2014-06-14"),
        variables=["temp", "sal", "u", "v"],
        product_id="hycom_glbv0.08_reanalysis_53x",
        description="Monterey Bay, 2 weeks, physical vars",
    ),
    "gulf_of_mexico_phy": Scenario(
        id="gulf_of_mexico_phy",
        name="Gulf of Mexico (Physical)",
        region=get_region_bounds("gulf_of_mexico"),
        time=("2014-01-01", "2014-01-31"),
        variables=["temp", "sal", "u", "v", "ssh"],
        product_id="hycom_glbv0.08_reanalysis_53x",
        description="Gulf of Mexico, 1 month, physical vars",
    ),
    "gulf_of_maine_phy": Scenario(
        id="gulf_of_maine_phy",
        name="Gulf of Maine (Physical)",
        region=get_region_bounds("gulf_of_maine"),
        time=("2014-07-01", "2014-07-31"),
        variables=["temp", "sal", "u", "v"],
        product_id="hycom_glbv0.08_reanalysis_53x",
        description="Gulf of Maine, 1 month, physical vars",
    ),
}


def list_scenarios() -> list[str]:
    """List available scenario IDs."""
    return sorted(_SCENARIOS.keys())


def get_scenario(scenario_id: str) -> Scenario:
    """Get scenario by ID."""
    if scenario_id not in _SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_id}. Available: {list_scenarios()}")
    return _SCENARIOS[scenario_id]
