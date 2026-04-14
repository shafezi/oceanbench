from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph, arrival_time, features_from_items
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
    Brute-force finite-horizon planner (Binney et al. 2013, Sec 7.1).

    Enumerates all simple paths up to ``max_depth`` edges from s, scores
    every partial path (not only those reaching the goal t), and returns
    the best one.  The paper's receding-horizon behaviour — take the first
    edge, then replan — is handled by the runner, not by this planner.

    With max_depth=1, this reduces to a purely greedy (myopic) planner.
    Intended only for toy graphs; callers should guard via config.
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
        X_items: Sequence[MeasurementItem] | None = None,
    ) -> Tuple[Optional[List[int]], List[MeasurementItem], float]:
        nodes = list(self.graph.nodes())
        if len(nodes) > self.config.max_nodes:
            # Too large; decline and let caller skip this baseline.
            return None, [], float("nan")

        X_items = list(X_items or [])
        X_feats = self._features_from_items(X_items)

        best_gain = float("-inf")
        best_path: Optional[List[int]] = None
        best_samples: List[MeasurementItem] = []

        paths_explored = 0

        def _score_path(path: List[int]) -> None:
            """Score a (possibly partial) path and update best if improved."""
            nonlocal best_gain, best_path, best_samples, paths_explored
            if len(path) < 2:
                return  # need at least one edge to have any samples
            samples = sampling_fn(
                path=path,
                tau=tau,
                graph=self.graph,
                sampling_cfg=self.sampling_config,
            )
            feats = self._features_from_items(samples)
            # Use marginal gain relative to prior context X so the planner
            # works correctly in receding-horizon mode.
            gain = float(self.objective.marginal_gain(X_feats, feats))
            if gain > best_gain:
                best_gain = gain
                best_path = list(path)
                best_samples = samples
            paths_explored += 1

        def dfs(path: List[int], remaining: float) -> None:
            nonlocal paths_explored
            if paths_explored >= self.config.max_paths:
                return

            # Score every partial path (the key fix: the paper's brute-force
            # planner evaluates all paths up to the horizon, not only those
            # that reach the goal node t).
            _score_path(path)

            # Stop expanding if we've reached the goal or the depth limit.
            if path[-1] == t:
                return
            if len(path) - 1 >= self.config.max_depth:
                return

            for nb in self.graph.graph.neighbors(path[-1]):
                nb = int(nb)
                if nb in path:
                    continue
                edge = self.graph.edge_attributes(path[-1], nb)
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
        return features_from_items(items)

