from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .graph import WaypointGraph


@dataclass(frozen=True)
class MeasurementItem:
    """
    A single measurement item collected along a path.

    Fields are chosen to be hashable so that sets of MeasurementItem can be
    used directly as the ground set for objectives.
    """

    lat: float
    lon: float
    time: Optional[np.datetime64]
    depth: Optional[float] = None

    source: str = "node"  # "node" or "edge"
    node_index: Optional[int] = None  # for nodes: index in path; for edges: edge index in path
    edge: Optional[Tuple[int, int]] = None
    alpha: Optional[float] = None  # fractional position along edge in [0, 1]


def arrival_time(
    path: Sequence[int],
    tau: np.datetime64,
    graph: WaypointGraph,
) -> np.datetime64:
    """
    Compute the arrival time at the final node in `path`.

    Parameters
    ----------
    path:
        Sequence of node ids, at least length 1.
    tau:
        Start time at the first node as np.datetime64.
    graph:
        WaypointGraph providing edge travel times.
    """
    if not path:
        raise ValueError("Path must be non-empty.")

    t = np.datetime64(tau)
    for u, v in zip(path[:-1], path[1:]):
        edge = graph.edge_attributes(u, v)
        dt_s = float(edge["time_s"])
        # Convert seconds to nanoseconds for datetime64[ns].
        dt_ns = int(round(dt_s * 1e9))
        t = t + np.timedelta64(dt_ns, "ns")
    return t


def h(
    path: Sequence[int],
    tau: np.datetime64,
    graph: WaypointGraph,
    sampling_cfg: Mapping[str, object],
) -> List[MeasurementItem]:
    """
    Deterministic node + edge sampling h(P, tau).

    Parameters
    ----------
    path:
        Ordered list of node ids [v0, v1, ..., vm].
    tau:
        Start time at node v0 as np.datetime64.
    graph:
        WaypointGraph providing node coordinates and edge travel times.
    sampling_cfg:
        Mapping with at least:
            - ``edge_spacing_m`` (float)
            - ``include_nodes`` (bool)

    Returns
    -------
    List[MeasurementItem]
        Measurement items collected along the path, including node samples
        (if enabled) and fixed-spacing edge samples with interpolated times.
    """
    if not path:
        raise ValueError("Path must be non-empty.")

    include_nodes = bool(sampling_cfg.get("include_nodes", True))
    edge_spacing_m = float(sampling_cfg.get("edge_spacing_m", 0.0))

    t = np.datetime64(tau)
    items: List[MeasurementItem] = []

    # Node samples — compute arrival times incrementally (O(n) instead of
    # O(n²) from repeated prefix traversals).
    if include_nodes:
        t_node = np.datetime64(tau)
        for idx, node in enumerate(path):
            lat, lon = graph.node_coords(node)
            node_dep = graph.node_depth(node)
            if idx > 0:
                edge = graph.edge_attributes(path[idx - 1], node)
                dt_ns = int(round(float(edge["time_s"]) * 1e9))
                t_node = t_node + np.timedelta64(dt_ns, "ns")
            items.append(
                MeasurementItem(
                    lat=lat,
                    lon=lon,
                    time=t_node,
                    depth=node_dep,
                    source="node",
                    node_index=idx,
                    edge=None,
                    alpha=None,
                )
            )

    # Edge samples.
    t_i = np.datetime64(tau)
    for idx, (u, v) in enumerate(zip(path[:-1], path[1:])):
        edge = graph.edge_attributes(u, v)
        d = float(edge["length_m"])
        c = float(edge["time_s"])

        lat_u, lon_u = graph.node_coords(u)
        lat_v, lon_v = graph.node_coords(v)
        dep_u = graph.node_depth(u)
        dep_v = graph.node_depth(v)

        if edge_spacing_m > 0.0 and d > 0.0:
            k = int(d // edge_spacing_m)
            for j in range(1, k + 1):
                alpha = (j * edge_spacing_m) / d
                if alpha >= 1.0:
                    break  # avoid duplicating the destination node
                lat = (1.0 - alpha) * lat_u + alpha * lat_v
                lon = (1.0 - alpha) * lon_u + alpha * lon_v
                # Interpolate depth along the edge when available.
                if dep_u is not None and dep_v is not None:
                    dep_alpha: Optional[float] = float(
                        (1.0 - alpha) * dep_u + alpha * dep_v
                    )
                else:
                    dep_alpha = None
                dt_s = alpha * c
                dt_ns = int(round(dt_s * 1e9))
                t_alpha = t_i + np.timedelta64(dt_ns, "ns")
                items.append(
                    MeasurementItem(
                        lat=float(lat),
                        lon=float(lon),
                        time=t_alpha,
                        depth=dep_alpha,
                        source="edge",
                        node_index=idx,
                        edge=(int(u), int(v)),
                        alpha=float(alpha),
                    )
                )

        # Advance base time to the next node.
        dt_edge_ns = int(round(c * 1e9))
        t_i = t_i + np.timedelta64(dt_edge_ns, "ns")

    return items


def snap_times_to_available(
    times: np.ndarray,
    available_times: np.ndarray,
    *,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Snap an array of times to the nearest available provider time stamps.

    This utility is kept generic so that higher-level code can pass in the
    provider's time coordinate (e.g., from an xarray.Dataset). It does not
    depend on any provider internals.
    """
    if times.size == 0 or available_times.size == 0:
        return times

    t_arr = np.asarray(times).astype("datetime64[ns]")
    avail = np.asarray(available_times).astype("datetime64[ns]")

    if mode != "nearest":
        raise ValueError(f"Unsupported snapping mode: {mode!r}")

    # Convert to integer nanoseconds for fast nearest-neighbour search.
    t_int = t_arr.astype("int64")
    avail_int = avail.astype("int64")

    # For each time, find index of closest available time.
    # This is O(N*M) but N is typically modest for measurement sets; can be
    # replaced by a more efficient search if needed later.
    snapped = []
    for ti in t_int:
        idx = int(np.argmin(np.abs(avail_int - ti)))
        snapped.append(avail[idx])
    return np.array(snapped, dtype="datetime64[ns]")


def features_from_items(items: Sequence[MeasurementItem]) -> np.ndarray:
    """
    Convert measurement items to a feature matrix for covariance backends
    and objectives.

    Returns an array with columns [lat, lon] or [lat, lon, depth] or
    [lat, lon, time_seconds] or [lat, lon, depth, time_seconds] depending
    on which optional fields are present.  Time is represented as seconds
    since the Unix epoch (float64).  Depth is in meters (positive down).

    Column order: lat, lon, [depth], [time].

    This is the single canonical implementation; all planners, baselines,
    and tasks should use this rather than maintaining their own copies.
    """
    if not items:
        return np.zeros((0, 2), dtype=float)

    lats = np.array([it.lat for it in items], dtype=float)
    lons = np.array([it.lon for it in items], dtype=float)
    feats: list[np.ndarray] = [lats, lons]

    if any(it.depth is not None for it in items):
        depths = np.array(
            [it.depth if it.depth is not None else 0.0 for it in items],
            dtype=float,
        )
        feats.append(depths)

    if any(it.time is not None for it in items):
        times = []
        for it in items:
            if it.time is None:
                times.append(np.datetime64("NaT", "ns"))
            else:
                times.append(np.datetime64(it.time, "ns"))
        t_arr = np.array(times, dtype="datetime64[ns]").astype("int64") / 1e9
        feats.append(t_arr.astype(float))

    return np.column_stack(feats)

