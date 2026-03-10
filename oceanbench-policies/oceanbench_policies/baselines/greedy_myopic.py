from __future__ import annotations

# Stop messages:
# - "[greedy_myopic] stop: reached goal node"
# - "[greedy_myopic] stop: budget exhausted"
# - "[greedy_myopic] stop: no neighbors"
# - "[greedy_myopic] stop: no feasible neighbor within budget"

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph, arrival_time
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class GreedyMyopicConfig:
    """Configuration for the greedy myopic baseline."""

    max_steps: int = 1_000


@dataclass
class GreedyMyopicBinneyPlanner:
    """
    Greedy myopic baseline: at each decision point, pick the neighbour that
    maximizes immediate marginal gain per unit travel time.
    """

    graph: WaypointGraph
    objective: BinneyObjective
    sampling_config: Mapping[str, object]
    config: GreedyMyopicConfig

    def plan(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
    ) -> Tuple[List[int], List[MeasurementItem], float]:
        current = s
        path = [current]
        remaining = float(B)
        X_items: list[MeasurementItem] = []
        X_feats = self._features_from_items(X_items)

        for _ in range(self.config.max_steps):
            if current == t or remaining <= 0.0:
                reason = "reached goal node" if current == t else "budget exhausted"
                print(f"[greedy_myopic] stop: {reason}")
                break

            neighbors = list(self.graph.graph.neighbors(current))
            if not neighbors:
                print("[greedy_myopic] stop: no neighbors")
                break

            best_score = float("-inf")
            best_next: Optional[int] = None
            best_samples: Optional[List[MeasurementItem]] = None

            for nb in neighbors:
                edge = self.graph.edge_attributes(current, nb)
                dt_edge = float(edge["time_s"])
                if dt_edge <= 0.0 or dt_edge > remaining:
                    continue

                # Approximate short local path [current, nb].
                local_path = [current, int(nb)]
                tau_local = arrival_time(path, tau, self.graph)
                S = sampling_fn(
                    path=local_path,
                    tau=tau_local,
                    graph=self.graph,
                    sampling_cfg=self.sampling_config,
                )
                S_feats = self._features_from_items(S)
                gain = float(self.objective.marginal_gain(X_feats, S_feats))
                score = gain / dt_edge if dt_edge > 0 else 0.0

                if score > best_score:
                    best_score = score
                    best_next = int(nb)
                    best_samples = S

            if best_next is None or best_samples is None:
                print("[greedy_myopic] stop: no feasible neighbor within budget")
                break

            # Commit to best_next.
            path.append(best_next)
            X_items.extend(best_samples)
            X_feats = self._features_from_items(X_items)
            remaining -= float(self.graph.edge_attributes(current, best_next)["time_s"])
            current = best_next

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
