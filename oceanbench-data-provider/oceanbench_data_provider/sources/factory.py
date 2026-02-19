"""Factory to get the appropriate source adapter for a product_id."""

from typing import Any

from oceanbench_data_provider.sources.base import SourceAdapter
from oceanbench_data_provider.sources.hycom import HycomAdapter
from oceanbench_data_provider.sources.copernicus_phy import CopernicusPhyAdapter
from oceanbench_data_provider.sources.copernicus_bgc import CopernicusBgcAdapter


def get_adapter(product_id: str, config: dict[str, Any]) -> SourceAdapter:
    """Return the adapter for the given product_id."""
    if product_id.startswith("hycom_"):
        return HycomAdapter(product_id, config)
    if product_id.startswith("copernicus_phy_"):
        return CopernicusPhyAdapter(product_id, config)
    if product_id.startswith("copernicus_bgc_"):
        return CopernicusBgcAdapter(product_id, config)
    raise ValueError(f"No adapter for product: {product_id}")
