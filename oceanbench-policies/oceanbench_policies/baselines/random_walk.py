from __future__ import annotations

# Stop messages:
# - "[random_walk] stop: cannot reach goal within remaining budget"
# - "[random_walk] stop: no neighbors"
# - "[random_walk] stop: no feasible neighbor within budget"
# - "[random_walk] stop: reached goal node"

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class RandomWalkConfig:
    """Configuration for the random-walk baseline under a travel-time budget."""

    max_steps: int = 1_000


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
        rng = np.random.default_rng()
        current = s
        path = [current]
        remaining = float(B)

        for _ in range(self.config.max_steps):
            # If direct path to goal might still be feasible, optionally stop.
            if not self.graph.is_feasible(current, t, remaining):
                print("[random_walk] stop: cannot reach goal within remaining budget")
                break

            neighbors = list(self.graph.graph.neighbors(current))
            if not neighbors:
                print("[random_walk] stop: no neighbors")
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
                print("[random_walk] stop: no feasible neighbor within budget")
                break

            if current == t:
                print("[random_walk] stop: reached goal node")
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
