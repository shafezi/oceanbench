from __future__ import annotations

# Stop messages:
# - "[lawnmower] stop: no path to next waypoint"
# - "[lawnmower] stop: invalid shortest path to next waypoint"
# - "[lawnmower] stop: budget exhausted"
# - "[lawnmower] stop: completed waypoint order"

from dataclasses import dataclass
from math import isclose
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class LawnmowerConfig:
    """Configuration for the lawnmower baseline."""

    max_nodes: int = 200


@dataclass
class LawnmowerBinneyPlanner:
    """
    Lawnmower baseline over a grid-like WaypointGraph.

    For graphs constructed via :func:`WaypointGraph.grid`, node ids follow a
    row-major ordering and this planner constructs a serpentine path that
    sweeps the grid while respecting a travel-time budget.
    """

    graph: WaypointGraph
    objective: BinneyObjective
    sampling_config: Mapping[str, object]
    config: LawnmowerConfig

    def _infer_grid_shape(self) -> Tuple[int, int]:
        n = self.graph.n_nodes
        candidates: list[Tuple[int, int]] = []
        for a in range(1, int(np.sqrt(n)) + 1):
            if n % a == 0:
                b = n // a
                candidates.append((a, b))
        if not candidates:
            return n, 1
        # Prefer shapes that are closer to square.
        a, b = min(candidates, key=lambda ab: abs(ab[0] - ab[1]))
        return a, b

    def _lawnmower_order(self) -> List[int]:
        n_lat, n_lon = self._infer_grid_shape()
        order: list[int] = []
        for i in range(n_lat):
            if i % 2 == 0:
                row = [i * n_lon + j for j in range(n_lon)]
            else:
                row = [i * n_lon + j for j in reversed(range(n_lon))]
            order.extend(row)
        return order

    def plan(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
    ) -> Tuple[List[int], List[MeasurementItem], float]:
        order = self._lawnmower_order()
        if len(order) > self.config.max_nodes:
            order = order[: self.config.max_nodes]

        # Start from s, then follow the lawnmower order.
        waypoints = [s] + [n for n in order if n != s]

        path: list[int] = [s]
        budget = float(B)

        stop_reason: Optional[str] = None
        for nxt in waypoints[1:]:
            # Connect current path end to nxt via shortest path.
            current = path[-1]
            try:
                local = self.graph.shortest_path(current, int(nxt))
            except Exception:
                stop_reason = "no path to next waypoint"
                print(f"[lawnmower] stop: {stop_reason}")
                continue
            if not local or local[0] != current:
                stop_reason = "invalid shortest path to next waypoint"
                print(f"[lawnmower] stop: {stop_reason}")
                continue
            # Compute travel time for this segment.
            seg_time = self.graph.path_travel_time(local)
            if seg_time > budget:
                stop_reason = "budget exhausted"
                print(f"[lawnmower] stop: {stop_reason}")
                break
            # Append without duplicating the first node.
            path.extend(local[1:])
            budget -= seg_time

        if stop_reason is None:
            print("[lawnmower] stop: completed waypoint order")

        samples = sampling_fn(
            path=path,
            tau=tau,
            graph=self.graph,
            sampling_cfg=self.sampling_config,
        )
        feats = self._features_from_items(samples)
        gain = float(self.objective.value(feats))
        return path, samples, gain

    @staticmethod
    def _features_from_items(items: Sequence[MeasurementItem]) -> np.ndarray:
        if not items:
            return np.zeros((0, 2), dtype=float)
        lats = np.array([it.lat for it in items], dtype=float)
        lons = np.array([it.lon for it in items], dtype=float)
        feats = [lats, lons]
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
