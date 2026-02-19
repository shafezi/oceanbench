"""Tests for cache keys."""

from oceanbench_data_provider.cache.keys import make_cache_key


def test_cache_key_deterministic():
    req = {
        "product_id": "hycom_glbv0.08_reanalysis_53x",
        "region": {"lon": [-80, -70], "lat": [25, 35]},
        "time": ("2020-01-01", "2020-01-07"),
        "variables": ["temp", "sal"],
        "depth_opts": {},
        "target_grid": {},
    }
    k1 = make_cache_key(req)
    k2 = make_cache_key(req)
    assert k1 == k2


def test_cache_key_different_requests_differ():
    req1 = {"product_id": "a", "region": {"lon": [0, 1], "lat": [0, 1]}, "time": ("2020-01-01", "2020-01-02"), "variables": ["temp"], "depth_opts": {}, "target_grid": {}}
    req2 = {"product_id": "a", "region": {"lon": [0, 2], "lat": [0, 1]}, "time": ("2020-01-01", "2020-01-02"), "variables": ["temp"], "depth_opts": {}, "target_grid": {}}
    assert make_cache_key(req1) != make_cache_key(req2)


def test_cache_key_includes_product_prefix():
    req = {"product_id": "hycom_glbv0.08_reanalysis_53x", "region": {}, "time": (), "variables": [], "depth_opts": {}, "target_grid": {}}
    k = make_cache_key(req)
    assert k.startswith("hycom_")
