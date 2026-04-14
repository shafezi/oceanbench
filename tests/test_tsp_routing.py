"""Tests for routing/tsp backends and modes."""

from __future__ import annotations

import numpy as np

from oceanbench_policies.routing import solve_tsp_route


def _points(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    lats = rng.uniform(24.0, 36.0, n)
    lons = rng.uniform(-86.0, -74.0, n)
    return np.column_stack([lats, lons])


def test_open_route_visits_all_once():
    pts = _points(9)
    route = solve_tsp_route(
        pts,
        start_index=0,
        backend="networkx",
        mode="open_end_anywhere",
        metric="haversine",
    )
    idx = np.asarray(route.indices, dtype=int)
    assert idx[0] == 0
    assert np.unique(idx).shape[0] == pts.shape[0]
    assert set(idx.tolist()) == set(range(pts.shape[0]))


def test_open_fixed_end_and_closed_modes():
    pts = _points(7)
    route_open_end = solve_tsp_route(
        pts,
        start_index=0,
        end_index=6,
        backend="networkx",
        mode="open_fixed_end",
        metric="haversine",
    )
    idx_open = np.asarray(route_open_end.indices, dtype=int)
    assert idx_open[0] == 0
    assert idx_open[-1] == 6

    route_closed = solve_tsp_route(
        pts,
        start_index=0,
        backend="networkx",
        mode="closed",
        metric="haversine",
    )
    idx_closed = np.asarray(route_closed.indices, dtype=int)
    assert idx_closed[0] == 0
    assert idx_closed[-1] == 0
    assert set(idx_closed[:-1].tolist()) == set(range(pts.shape[0]))


def test_routing_deterministic_for_networkx_backend():
    pts = _points(10)
    r1 = solve_tsp_route(
        pts,
        start_index=0,
        backend="networkx",
        mode="open_end_anywhere",
        metric="haversine",
    )
    r2 = solve_tsp_route(
        pts,
        start_index=0,
        backend="networkx",
        mode="open_end_anywhere",
        metric="haversine",
    )
    assert np.array_equal(r1.indices, r2.indices)
