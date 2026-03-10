from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph, arrival_time
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class BruteforceConfig:
    """
    Configuration for the brute-force finite-horizon baseline.
    """

    max_depth: int = 4
    max_nodes: int = 50
    max_paths: int = 10_000


@dataclass
class BruteforceFiniteHorizonPlanner:
    """
    Brute-force planner enumerating simple paths up to a small depth.

    Intended only for toy graphs; callers should guard its use via config.
    """

    graph: WaypointGraph
    objective: BinneyObjective
    sampling_config: Mapping[str, object]
    config: BruteforceConfig

    def plan(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
    ) -> Tuple[Optional[List[int]], List[MeasurementItem], float]:
        nodes = list(self.graph.nodes())
        if len(nodes) > self.config.max_nodes:
            # Too large; decline and let caller skip this baseline.
            return None, [], float("nan")

        best_gain = float("-inf")
        best_path: Optional[List[int]] = None
        best_samples: List[MeasurementItem] = []

        paths_explored = 0

        def dfs(path: List[int], remaining: float) -> None:
            nonlocal best_gain, best_path, best_samples, paths_explored
            if paths_explored >= self.config.max_paths:
                return

            current = path[-1]
            if current == t:
                samples = sampling_fn(
                    path=path,
                    tau=tau,
                    graph=self.graph,
                    sampling_cfg=self.sampling_config,
                )
                feats = self._features_from_items(samples)
                gain = float(self.objective.value(feats))
                if gain > best_gain:
                    best_gain = gain
                    best_path = list(path)
                    best_samples = samples
                paths_explored += 1
                return

            if len(path) >= self.config.max_depth:
                return

            for nb in self.graph.graph.neighbors(current):
                nb = int(nb)
                if nb in path:
                    continue
                edge = self.graph.edge_attributes(current, nb)
                dt_edge = float(edge["time_s"])
                if dt_edge > remaining:
                    continue
                path.append(nb)
                dfs(path, remaining - dt_edge)
                path.pop()

        dfs([s], float(B))
        return best_path, best_samples, best_gain

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

