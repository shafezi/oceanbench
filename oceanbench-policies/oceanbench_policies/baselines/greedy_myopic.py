from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from oceanbench_core import WaypointGraph, arrival_time, features_from_items
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
        tau_current = np.datetime64(tau)  # cached arrival time at current node
        X_items: list[MeasurementItem] = []
        X_feats = self._features_from_items(X_items)

        for _ in range(self.config.max_steps):
            if current == t or remaining <= 0.0:
                reason = "reached goal node" if current == t else "budget exhausted"
                logger.debug("[greedy_myopic] stop: %s", reason)
                break

            neighbors = list(self.graph.graph.neighbors(current))
            if not neighbors:
                logger.debug("[greedy_myopic] stop: no neighbors")
                break

            best_score = float("-inf")
            best_next: Optional[int] = None
            best_samples: Optional[List[MeasurementItem]] = None

            for nb in neighbors:
                edge = self.graph.edge_attributes(current, nb)
                dt_edge = float(edge["time_s"])
                if dt_edge <= 0.0 or dt_edge > remaining:
                    continue
                # Must still be able to reach goal t after moving to nb.
                remaining_after = remaining - dt_edge
                if not self.graph.is_feasible(int(nb), t, remaining_after):
                    continue

                # Approximate short local path [current, nb].
                local_path = [current, int(nb)]
                S = sampling_fn(
                    path=local_path,
                    tau=tau_current,
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
                logger.debug("[greedy_myopic] stop: no feasible neighbor within budget")
                break

            # Commit to best_next and advance cached arrival time.
            edge = self.graph.edge_attributes(current, best_next)
            dt_ns = int(round(float(edge["time_s"]) * 1e9))
            tau_current = tau_current + np.timedelta64(dt_ns, "ns")
            path.append(best_next)
            X_items.extend(best_samples)
            X_feats = self._features_from_items(X_items)
            remaining -= float(edge["time_s"])
            current = best_next

        samples = sampling_fn(
            path=path,
            tau=tau,
            graph=self.graph,
            sampling_cfg=self.sampling_config,
        )
        feats = self._features_from_items(samples)
        # Use value() which equals marginal_gain(empty, feats) since this
        # planner starts fresh with no prior context.
        gain = float(self.objective.value(feats))
        return path, samples, gain

    @staticmethod
    def _features_from_items(items: Sequence[MeasurementItem]) -> np.ndarray:
        return features_from_items(items)
