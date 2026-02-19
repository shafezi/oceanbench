"""Tests for DataProvider (mocked)."""

import pytest
from unittest.mock import Mock, patch

from oceanbench_data_provider.provider import DataProvider
from oceanbench_data_provider.catalog import list_products


def test_provider_list_products():
    p = DataProvider()
    prods = p.list_products()
    assert "hycom_glbv0.08_reanalysis_53x" in prods


def test_provider_describe():
    p = DataProvider()
    card = p.describe("hycom_glbv0.08_reanalysis_53x")
    assert card.product_id == "hycom_glbv0.08_reanalysis_53x"


def test_provider_estimate_size():
    p = DataProvider()
    region = {"lon": [-80, -70], "lat": [25, 35]}
    time_range = ("2020-01-01", "2020-01-07")
    variables = ["temp", "sal"]
    est = p.estimate_size("hycom_glbv0.08_reanalysis_53x", region, time_range, variables)
    assert "estimate_bytes" in est or "estimate_mb" in est
