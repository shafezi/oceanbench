from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from oceanbench_core import WaypointGraph, features_from_items
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class RandomWalkConfig:
    """Configuration for the random-walk baseline under a travel-time budget."""

    max_steps: int = 1_000
    seed: Optional[int] = None


@dataclass
class RandomWalkBinneyPlanner:
    """
    Random walk baseline: at each step, choose a random feasible neighbour
    until the budget is exhausted or the goal is reached.
    """

    graph: WaypointGraph
    objective: BinneyObjective
    sampling_config: Mapping[str, object]
    config: RandomWalkConfig

    def plan(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
    ) -> Tuple[List[int], List[MeasurementItem], float]:
        rng = np.random.default_rng(self.config.seed)
        current = s
        path = [current]
        remaining = float(B)

        for _ in range(self.config.max_steps):
            # If direct path to goal might still be feasible, optionally stop.
            if not self.graph.is_feasible(current, t, remaining):
                logger.debug("[random_walk] stop: cannot reach goal within remaining budget")
                break

            neighbors = list(self.graph.graph.neighbors(current))
            if not neighbors:
                logger.debug("[random_walk] stop: no neighbors")
                break

            rng.shuffle(neighbors)
            moved = False
            for nb in neighbors:
                # Check if we can move to nb and still reach goal.
                # Approximate cost: time to nb plus shortest path from nb to t.
                edge = self.graph.edge_attributes(current, nb)
                dt_edge = float(edge["time_s"])
                if dt_edge > remaining:
                    continue
                remaining_after = remaining - dt_edge
                if not self.graph.is_feasible(nb, t, remaining_after):
                    continue
                path.append(int(nb))
                current = int(nb)
                remaining = remaining_after
                moved = True
                break

            if not moved:
                logger.debug("[random_walk] stop: no feasible neighbor within budget")
                break

            if current == t:
                logger.debug("[random_walk] stop: reached goal node")
                break

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
