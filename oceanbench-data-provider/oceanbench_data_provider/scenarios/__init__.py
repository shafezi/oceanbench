"""Scenario presets: regions, time windows, variables."""

from oceanbench_data_provider.scenarios.registry import list_scenarios, get_scenario
from oceanbench_data_provider.scenarios.regions import list_regions, get_region, get_region_bounds

__all__ = [
    "list_scenarios",
    "get_scenario",
    "list_regions",
    "get_region",
    "get_region_bounds",
]
