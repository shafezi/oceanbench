from __future__ import annotations

# Stop messages:
# - "[grg] stop: shortest path time exceeds budget (shortest_time=..., budget=...)"
# - "[grg] stop: no feasible path found"
# - "[grg] stop: recursion completed (path_len=..., depth=...)"

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import WaypointGraph, arrival_time
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_tasks.mapping.binney_objectives import BinneyObjective


@dataclass
class GRGConfig:
    """
    Configuration for the Generalized Recursive Greedy (GRG) planner.
    """

    depth: int = 2
    n_splits: int = 5
    split_strategy: str = "uniform"  # or "geometric"


@dataclass
class GRGPlanner:
    """
    Generalized Recursive Greedy (GRG) planner for Binney et al. (2013).

    This planner operates on:
      - a WaypointGraph with travel-time edge costs,
      - a BinneyObjective instance over measurement-item features,
      - a sampling function h(P, tau) producing MeasurementItems.
    """

    graph: WaypointGraph
    objective: BinneyObjective
    sampling_config: Mapping[str, object]
    config: GRGConfig
    _dist_cache: Dict[Tuple[int, int], float] = field(default_factory=dict, repr=False)
    _path_cache: Dict[Tuple[int, int], List[int]] = field(default_factory=dict, repr=False)

    def _shortest_time_cached(self, s: int, t: int) -> float:
        key = (int(s), int(t))
        if key not in self._dist_cache:
            self._dist_cache[key] = self.graph.shortest_time(s, t)
        return self._dist_cache[key]

    def _shortest_path_cached(self, s: int, t: int) -> List[int]:
        key = (int(s), int(t))
        if key not in self._path_cache:
            self._path_cache[key] = self.graph.shortest_path(s, t)
        return list(self._path_cache[key])

    def _budget_split_candidates(self, B: float) -> np.ndarray:
        if self.config.n_splits <= 1:
            return np.array([B / 2.0], dtype=float)
        if self.config.split_strategy == "uniform":
            return np.linspace(0.0, B, self.config.n_splits + 2)[1:-1]
        if self.config.split_strategy == "geometric":
            # Geometric sequence biased towards equal split.
            xs = np.linspace(0.1, 0.9, self.config.n_splits)
            return xs * B
        raise ValueError(f"Unknown split_strategy: {self.config.split_strategy!r}")

    def plan(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
        X_items: Sequence[MeasurementItem] | None = None,
    ) -> Tuple[Optional[List[int]], List[MeasurementItem], float]:
        """
        Entry point for GRG.

        Returns (best_path, best_samples, best_gain).
        """
        if self._shortest_time_cached(s, t) > float(B):
            print(
                f"[grg] stop: shortest path time exceeds budget "
                f"(shortest_time={self._shortest_time_cached(s, t):.3f}, budget={float(B):.3f})"
            )
            return None, [], float("-inf")
        X_items = list(X_items or [])
        X_feats = self._features_from_items(X_items)
        path, samples, gain = self._grg_recursive(
            s=s,
            t=t,
            B=float(B),
            tau=tau,
            X_items=X_items,
            X_feats=X_feats,
            depth=self.config.depth,
        )
        if path is None:
            print("[grg] stop: no feasible path found")
        else:
            print(
                f"[grg] stop: recursion completed (path_len={len(path)}, depth={self.config.depth})"
            )
        return path, samples, gain

    # ------------------------------------------------------------------
    # Core recursion
    # ------------------------------------------------------------------

    def _grg_recursive(
        self,
        s: int,
        t: int,
        B: float,
        tau: np.datetime64,
        X_items: List[MeasurementItem],
        X_feats: np.ndarray,
        depth: int,
    ) -> Tuple[Optional[List[int]], List[MeasurementItem], float]:
        # Feasibility check.
        if self._shortest_time_cached(s, t) > B:
            return None, [], float("-inf")

        # Base case: depth == 0 -> shortest path with no midpoints.
        if depth == 0:
            P0 = self._shortest_path_cached(s, t)
            S0 = sampling_fn(
                path=P0,
                tau=tau,
                graph=self.graph,
                sampling_cfg=self.sampling_config,
            )
            S0_feats = self._features_from_items(S0)
            gain = self.objective.marginal_gain(X_feats, S0_feats)
            return P0, S0, float(gain)

        best_gain = float("-inf")
        best_path: Optional[List[int]] = None
        best_samples: List[MeasurementItem] = []

        nodes = list(self.graph.nodes())
        splits = self._budget_split_candidates(B)

        for v in nodes:
            if self._shortest_time_cached(s, v) + self._shortest_time_cached(v, t) > B:
                continue

            for B1 in splits:
                B2 = B - float(B1)
                if self._shortest_time_cached(s, v) > B1:
                    continue
                if self._shortest_time_cached(v, t) > B2:
                    continue

                # First half recursion.
                P1, S1, gain1 = self._grg_recursive(
                    s=s,
                    t=v,
                    B=float(B1),
                    tau=tau,
                    X_items=X_items,
                    X_feats=X_feats,
                    depth=depth - 1,
                )
                if P1 is None:
                    continue

                tau_v = arrival_time(P1, tau, self.graph)
                X1_items = X_items + S1
                X1_feats = self._features_from_items(X1_items)

                # Second half recursion.
                P2, S2, gain2 = self._grg_recursive(
                    s=v,
                    t=t,
                    B=float(B2),
                    tau=tau_v,
                    X_items=X1_items,
                    X_feats=X1_feats,
                    depth=depth - 1,
                )
                if P2 is None:
                    continue

                P = self._concat_paths(P1, P2)
                S = S1 + S2
                S_feats = self._features_from_items(S)
                total_gain = float(self.objective.marginal_gain(X_feats, S_feats))

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_path = P
                    best_samples = S

        return best_path, best_samples, best_gain

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_paths(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
        if not p1:
            return list(p2)
        if not p2:
            return list(p1)
        if p1[-1] == p2[0]:
            return list(p1) + list(p2[1:])
        return list(p1) + list(p2)

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

