from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from oceanbench_core.types import QueryPoints

from .mi_dp_planner import (
    MIPlannerResult,
    _safe_precision,
    compute_mutual_information_from_covariance,
    conditional_variance,
    predictive_covariance_from_model,
    select_mi_universe_points,
    subset_query_points,
)


@dataclass
class MIGreedyConfig:
    batch_size_n: int = 10
    x_set: str = "candidate_grid"
    x_subsample_max_points: int = 500
    jitter: float = 1e-8
    max_candidates_for_greedy: int = 1024


class MIGreedyPlanner:
    """
    Greedy MI baseline: add one point at a time by max marginal gain.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        cfg = dict(config or {})
        root = dict(cfg.get("planner", cfg))
        mi_cfg = dict(cfg.get("mi", {}))
        self.config = MIGreedyConfig(
            batch_size_n=max(1, int(root.get("batch_size_n", 10))),
            x_set=str(mi_cfg.get("X_set", root.get("X_set", "candidate_grid"))).lower(),
            x_subsample_max_points=max(1, int(mi_cfg.get("X_subsample", {}).get("max_points", mi_cfg.get("x_subsample_max_points", 500)))),
            jitter=float(mi_cfg.get("jitter", 1e-8)),
            max_candidates_for_greedy=max(4, int(root.get("max_candidates_for_greedy", 1024))),
        )
        self._rng = np.random.default_rng(seed)

    def plan(
        self,
        model: Any,
        candidate_points: QueryPoints,
        *,
        eval_points: Optional[QueryPoints] = None,
    ) -> MIPlannerResult:
        universe = select_mi_universe_points(
            candidate_points=candidate_points,
            eval_points=eval_points,
            x_set=self.config.x_set,
            subsample_max_points=self.config.x_subsample_max_points,
            rng=self._rng,
        )
        n_u = int(universe.size)
        if n_u == 0:
            raise ValueError("Universe points are empty.")
        n_select = min(self.config.batch_size_n, n_u)

        cov_full = predictive_covariance_from_model(model, universe, jitter=self.config.jitter)
        pool_idx = np.arange(n_u, dtype=int)
        if n_u > self.config.max_candidates_for_greedy:
            diag_var = np.diag(cov_full)
            keep = np.argsort(diag_var)[-self.config.max_candidates_for_greedy :]
            keep = np.sort(keep)
            pool_idx = keep.astype(int)
            cov = cov_full[np.ix_(pool_idx, pool_idx)]
        else:
            cov = cov_full

        selected_local, gains = _greedy_select(
            cov,
            n_select=n_select,
            jitter=self.config.jitter,
        )
        selected_global = pool_idx[np.asarray(selected_local, dtype=int)]
        selected_points = subset_query_points(universe, selected_global)
        obj = compute_mutual_information_from_covariance(
            cov_full,
            selected_indices=selected_global,
            jitter=self.config.jitter,
        )
        return MIPlannerResult(
            selected_indices=np.asarray(selected_global, dtype=int),
            selected_points=selected_points,
            objective_value=float(obj),
            marginal_gains=np.asarray(gains, dtype=float),
            debug={
                "universe_size": int(n_u),
                "pool_size": int(pool_idx.size),
                "x_set": self.config.x_set,
            },
        )


def _greedy_select(
    covariance: np.ndarray,
    *,
    n_select: int,
    jitter: float,
) -> tuple[list[int], list[float]]:
    n = int(covariance.shape[0])
    if n_select <= 0:
        return [], []
    precision = _safe_precision(covariance, jitter=jitter)
    marginal_var = np.diag(covariance)

    # The conditional variance given *all other* points.  Clip to at
    # least a small fraction of the marginal variance to avoid extreme
    # gain ratios when the covariance matrix is ill-conditioned.
    prec_diag = np.diag(precision)
    cond_all_var = np.where(
        prec_diag > jitter,
        1.0 / prec_diag,
        marginal_var,  # fallback: treat as unconditioned
    )
    # Floor: at least 1% of marginal variance (prevents log-ratio blowup).
    cond_all_var = np.maximum(cond_all_var, 0.01 * np.maximum(marginal_var, jitter))

    selected: list[int] = []
    gains: list[float] = []
    available = set(range(n))

    for _ in range(n_select):
        best_idx = None
        best_gain = -np.inf
        for i in available:
            var_cond_prev = conditional_variance(
                covariance,
                index=int(i),
                conditioned_on=selected,
                jitter=jitter,
            )
            denom = max(float(cond_all_var[i]), jitter)
            g = float(0.5 * np.log(max(var_cond_prev, jitter) / denom))
            # Clamp gain to prevent extreme values from numerical issues.
            g = min(g, 30.0)
            if g > best_gain:
                best_gain = g
                best_idx = int(i)
        if best_idx is None:
            break
        selected.append(best_idx)
        gains.append(float(best_gain))
        available.remove(best_idx)

    return selected, gains

