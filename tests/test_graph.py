from __future__ import annotations

import numpy as np

from oceanbench_core import WaypointGraph


def test_grid_graph_deterministic():
    region = {"lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0}
    g1 = WaypointGraph.grid(region, n_lat=3, n_lon=4, speed_mps=1.0, seed=0)
    g2 = WaypointGraph.grid(region, n_lat=3, n_lon=4, speed_mps=1.0, seed=999)

    assert g1.n_nodes == g2.n_nodes == 12
    assert g1.n_edges == g2.n_edges

    coords1 = [g1.node_coords(i) for i in range(g1.n_nodes)]
    coords2 = [g2.node_coords(i) for i in range(g2.n_nodes)]
    assert np.allclose(coords1, coords2)


def test_shortest_path_and_feasibility():
    region = {"lat_min": 0.0, "lat_max": 1.0, "lon_min": 0.0, "lon_max": 1.0}
    g = WaypointGraph.grid(region, n_lat=2, n_lon=2, speed_mps=1.0, seed=0)
    path = g.shortest_path(0, 3)
    assert path[0] == 0 and path[-1] == 3
    T = g.shortest_time(0, 3)
    assert T > 0.0
    assert g.is_feasible(0, 3, budget_seconds=T + 1e-6)
    assert not g.is_feasible(0, 3, budget_seconds=T - 1e-3)

