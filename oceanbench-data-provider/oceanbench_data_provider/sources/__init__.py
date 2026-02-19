"""Source adapters for HYCOM and Copernicus Marine."""

from oceanbench_data_provider.sources.factory import get_adapter
from oceanbench_data_provider.sources.base import SourceAdapter

__all__ = ["get_adapter", "SourceAdapter"]
