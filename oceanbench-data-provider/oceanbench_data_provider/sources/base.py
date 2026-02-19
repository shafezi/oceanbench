"""Base class for source adapters."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import xarray as xr


class SourceAdapter(ABC):
    """Abstract adapter for ocean data sources."""

    @abstractmethod
    def fetch_subset(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
        depth_opts: dict[str, Any],
    ) -> Optional[xr.Dataset]:
        """Fetch subset and return raw xarray.Dataset. Sets request['_var_mapping']."""
        pass

    @abstractmethod
    def estimate_size(
        self,
        region: dict[str, list[float]],
        time: tuple[str, str],
        variables: list[str],
    ) -> dict[str, Any]:
        """Estimate download and cache size."""
        pass
