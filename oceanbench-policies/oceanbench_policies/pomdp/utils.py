"""Shared helpers for the POMCP adapter layer."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core.types import QueryPoints
from oceanbench_core import WaypointGraph


# ---------------------------------------------------------------------------
# Action-space builders
# ---------------------------------------------------------------------------


def graph_neighbor_actions(
    graph: WaypointGraph,
    node_id: int,
    *,
    include_stay: bool = True,
) -> list[int]:
    """Return the list of reachable node ids from *node_id* in the graph.

    If *include_stay* is True the current node is prepended (STAY action).
    """
    neighbors = list(graph.graph.neighbors(node_id))
    if include_stay and node_id not in neighbors:
        neighbors = [node_id] + neighbors
    return neighbors


def knn_candidate_actions(
    graph: WaypointGraph,
    node_id: int,
    k: int = 8,
) -> list[int]:
    """Return up to *k* nearest nodes by graph shortest-path distance."""
    nodes = list(graph.graph.nodes)
    if len(nodes) <= k + 1:
        return [n for n in nodes if n != node_id]

    lat0 = float(graph.graph.nodes[node_id]["lat"])
    lon0 = float(graph.graph.nodes[node_id]["lon"])

    dists = []
    for n in nodes:
        if n == node_id:
            continue
        lat_n = float(graph.graph.nodes[n]["lat"])
        lon_n = float(graph.graph.nodes[n]["lon"])
        d = np.hypot(lat_n - lat0, lon_n - lon0)
        dists.append((d, n))
    dists.sort()
    return [n for _, n in dists[:k]]


def node_coords(graph: WaypointGraph, node_id: int) -> tuple[float, float]:
    """Return (lat, lon) for a graph node."""
    data = graph.graph.nodes[node_id]
    return float(data["lat"]), float(data["lon"])


# ---------------------------------------------------------------------------
# Transition helpers
# ---------------------------------------------------------------------------


def build_transition_table(
    graph: WaypointGraph,
    action_space: str = "graph_neighbors",
    knn_k: int = 8,
    include_stay: bool = True,
) -> dict[int, list[int]]:
    """Build ``{node_id: [reachable_node_ids]}`` for every node in the graph.

    Parameters
    ----------
    graph:
        The WaypointGraph.
    action_space:
        ``"graph_neighbors"`` uses adjacency, ``"knn"`` uses k-nearest.
    knn_k:
        Number of nearest neighbours when ``action_space="knn"``.
    include_stay:
        Whether the current node appears as an available action.
    """
    table: dict[int, list[int]] = {}
    for n in graph.graph.nodes:
        if action_space == "knn":
            table[n] = knn_candidate_actions(graph, n, k=knn_k)
        else:
            table[n] = graph_neighbor_actions(graph, n, include_stay=include_stay)
    return table


# ---------------------------------------------------------------------------
# Discretised observation helper
# ---------------------------------------------------------------------------


def discretize_observation(value: float, n_bins: int = 20, lo: float = -5.0, hi: float = 5.0) -> int:
    """Map a continuous scalar observation to an integer bin index.

    POMCP requires discrete observations for tree indexing.  We use a simple
    uniform binning of the value range.
    """
    clamped = max(lo, min(hi, value))
    bin_width = (hi - lo) / n_bins
    idx = int((clamped - lo) / bin_width)
    return min(idx, n_bins - 1)


# ---------------------------------------------------------------------------
# Timing / budget helpers
# ---------------------------------------------------------------------------


def travel_time_seconds(
    graph: WaypointGraph, src: int, dst: int,
) -> float:
    """Return edge travel time in seconds between two adjacent nodes."""
    if src == dst:
        return 0.0
    try:
        edge = graph.edge_attributes(src, dst)
        return float(edge["time_s"])
    except Exception:
        return float("inf")


def advance_time(
    current: np.datetime64,
    seconds: float,
) -> np.datetime64:
    """Advance a numpy datetime64 by *seconds*."""
    return current + np.timedelta64(int(seconds), "s")
