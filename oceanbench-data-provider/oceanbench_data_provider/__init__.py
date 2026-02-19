"""
OceanBench: Ocean data provider for Informative Path Planning (IPP) benchmark.

Provides sanitized, subsettable, and reproducible ocean model data from
HYCOM and Copernicus Marine Service.
"""

from oceanbench_data_provider.provider import DataProvider
from oceanbench_data_provider.catalog import list_products, describe
from oceanbench_data_provider.thermo import add_pressure_and_sound_speed

__version__ = "0.1.0"
__all__ = ["DataProvider", "list_products", "describe", "add_pressure_and_sound_speed", "__version__"]
