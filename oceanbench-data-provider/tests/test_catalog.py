"""Tests for catalog and Dataset Cards."""

import pytest
from oceanbench_data_provider.catalog import list_products, describe, DatasetCard


def test_list_products():
    products = list_products()
    assert len(products) > 0
    assert "hycom_glbv0.08_reanalysis_53x" in products
    assert "copernicus_phy_reanalysis_001_030" in products


def test_describe_hycom():
    card = describe("hycom_glbv0.08_reanalysis_53x")
    assert isinstance(card, DatasetCard)
    assert card.product_id == "hycom_glbv0.08_reanalysis_53x"
    assert card.provider_name == "HYCOM"
    assert "temp" in [v.canonical_name for v in card.variables]


def test_describe_unknown_raises():
    with pytest.raises(KeyError):
        describe("unknown_product_xyz")


def test_dataset_card_to_dict():
    card = describe("hycom_glbv0.08_reanalysis_53x")
    d = card.to_dict()
    assert "product_id" in d
    assert "variables" in d
    card2 = DatasetCard.from_dict(d)
    assert card2.product_id == card.product_id
