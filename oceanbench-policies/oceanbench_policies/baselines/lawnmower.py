from __future__ import annotations

import logging

from dataclasses import dataclass
from math import isclose
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from oceanbench_core import WaypointGraph, features_from_items
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

    def _infer_grid_shape(self) -> Tuple[int, ...]:
        """Infer (n_lat, n_lon) or (n_lat, n_lon, n_depth) from node count.

        For 3-D grids (``graph.has_depth``), the total node count is
        ``n_lat * n_lon * n_depth``.  We attempt to factorise into three
        factors; if no clean factorisation exists we fall back to a 2-D
        shape with a single depth layer.
        """
        n = self.graph.n_nodes
        use_3d = self.graph.has_depth

        if use_3d:
            # Try to find three factors close to cube root.
            best: Tuple[int, ...] = (n, 1, 1)
            best_score = float("inf")
            for a in range(1, int(round(n ** (1 / 3))) + 2):
                if n % a != 0:
                    continue
                rem = n // a
                for b in range(a, int(np.sqrt(rem)) + 2):
                    if rem % b != 0:
                        continue
                    c = rem // b
                    score = abs(a - b) + abs(b - c)
                    if score < best_score:
                        best_score = score
                        best = (a, b, c)
            return best

        # 2-D factorisation.
        candidates: list[Tuple[int, int]] = []
        for a in range(1, int(np.sqrt(n)) + 1):
            if n % a == 0:
                b = n // a
                candidates.append((a, b))
        if not candidates:
            return (n, 1)
        a, b = min(candidates, key=lambda ab: abs(ab[0] - ab[1]))
        return (a, b)

    def _lawnmower_order(self) -> List[int]:
        shape = self._infer_grid_shape()

        if len(shape) == 3:
            n_lat, n_lon, n_depth = shape
            return self._lawnmower_order_3d(n_lat, n_lon, n_depth)

        n_lat, n_lon = shape
        return self._lawnmower_order_2d(n_lat, n_lon)

    @staticmethod
    def _lawnmower_order_2d(n_lat: int, n_lon: int, base: int = 0) -> List[int]:
        order: list[int] = []
        for i in range(n_lat):
            if i % 2 == 0:
                row = [base + i * n_lon + j for j in range(n_lon)]
            else:
                row = [base + i * n_lon + j for j in reversed(range(n_lon))]
            order.extend(row)
        return order

    @classmethod
    def _lawnmower_order_3d(
        cls, n_lat: int, n_lon: int, n_depth: int,
    ) -> List[int]:
        """Layer-by-layer serpentine: sweep each depth layer, alternate direction."""
        n_horiz = n_lat * n_lon
        order: list[int] = []
        for k in range(n_depth):
            base = k * n_horiz
            layer = cls._lawnmower_order_2d(n_lat, n_lon, base)
            if k % 2 == 1:
                layer = list(reversed(layer))
            order.extend(layer)
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
                logger.debug("[lawnmower] stop: %s", stop_reason)
                break
            if not local or local[0] != current:
                stop_reason = "invalid shortest path to next waypoint"
                logger.debug("[lawnmower] stop: %s", stop_reason)
                break
            # Compute travel time for this segment.
            seg_time = self.graph.path_travel_time(local)
            if seg_time > budget:
                stop_reason = "budget exhausted"
                logger.debug("[lawnmower] stop: %s", stop_reason)
                break
            # Append without duplicating the first node.
            path.extend(local[1:])
            budget -= seg_time

        if stop_reason is None:
            logger.debug("[lawnmower] stop: completed waypoint order")

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
        return features_from_items(items)
