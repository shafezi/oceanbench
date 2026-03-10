from __future__ import annotations

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.sampling import h, arrival_time


def test_h_edge_sampling_counts_and_times():
    region = {"lat_min": 0.0, "lat_max": 0.1, "lon_min": 0.0, "lon_max": 0.1}
    g = WaypointGraph.grid(region, n_lat=2, n_lon=2, speed_mps=1.0, seed=0)
    # Path across two edges: 0 -> 1 -> 3
    path = [0, 1, 3]
    tau = np.datetime64("2014-01-01T00:00:00", "ns")

    # Use a relatively small spacing to guarantee at least one interior edge sample.
    edge = g.edge_attributes(0, 1)
    d = float(edge["length_m"])
    spacing = d / 2.0

    items = h(
        path=path,
        tau=tau,
        graph=g,
        sampling_cfg={"edge_spacing_m": spacing, "include_nodes": True},
    )

    # There should be 3 node samples plus at least one edge sample.
    node_items = [it for it in items if it.source == "node"]
    edge_items = [it for it in items if it.source == "edge"]
    assert len(node_items) == len(path)
    assert len(edge_items) >= 1

    # Arrival time at final node should match arrival_time helper.
    t_arrival = arrival_time(path, tau, g)
    assert node_items[-1].time == t_arrival

