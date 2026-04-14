from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core.types import QueryPoints


@dataclass
class MIPlannerConfig:
    batch_size_n: int = 10
    x_set: str = "candidate_grid"  # candidate_grid | eval_grid | subsample
    x_subsample_max_points: int = 500
    jitter: float = 1e-8
    dp_beam_width: int = 64
    max_candidates_for_dp: int = 256
    dp_exact_threshold: int = 64  # use exact DP when pool_size <= this


@dataclass
class MIPlannerResult:
    selected_indices: np.ndarray
    selected_points: QueryPoints
    objective_value: float
    marginal_gains: np.ndarray
    debug: dict[str, Any]


class MIDPPlanner:
    """
    DP-style MI planner (paper-style approximation in Eq.26 recursion form).
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
        self.config = MIPlannerConfig(
            batch_size_n=max(1, int(root.get("batch_size_n", 10))),
            x_set=str(mi_cfg.get("X_set", root.get("X_set", "candidate_grid"))).lower(),
            x_subsample_max_points=max(1, int(mi_cfg.get("X_subsample", {}).get("max_points", mi_cfg.get("x_subsample_max_points", 500)))),
            jitter=float(mi_cfg.get("jitter", 1e-8)),
            dp_beam_width=max(1, int(root.get("dp_beam_width", 64))),
            max_candidates_for_dp=max(4, int(root.get("max_candidates_for_dp", 256))),
            dp_exact_threshold=max(0, int(root.get("dp_exact_threshold", 64))),
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
        if n_u > self.config.max_candidates_for_dp:
            # Keep most uncertain points for tractable DP.
            diag_var = np.diag(cov_full)
            keep = np.argsort(diag_var)[-self.config.max_candidates_for_dp :]
            keep = np.sort(keep)
            pool_idx = keep.astype(int)
            cov = cov_full[np.ix_(pool_idx, pool_idx)]
        else:
            cov = cov_full

        # Use exact DP (unlimited beam width) for small candidate pools,
        # matching the paper's Algorithm 1.  Fall back to beam search for
        # larger pools where exact DP is intractable.
        pool_size = int(pool_idx.size)
        if pool_size <= self.config.dp_exact_threshold:
            effective_beam = pool_size ** 2  # effectively unlimited
        else:
            effective_beam = self.config.dp_beam_width

        seq_local, gains = _dp_select_sequence(
            cov,
            n_select=n_select,
            beam_width=effective_beam,
            jitter=self.config.jitter,
        )
        seq_global = pool_idx[np.asarray(seq_local, dtype=int)]
        selected = subset_query_points(universe, seq_global)
        obj = compute_mutual_information_from_covariance(
            cov_full,
            selected_indices=seq_global,
            jitter=self.config.jitter,
        )
        return MIPlannerResult(
            selected_indices=np.asarray(seq_global, dtype=int),
            selected_points=selected,
            objective_value=float(obj),
            marginal_gains=np.asarray(gains, dtype=float),
            debug={
                "universe_size": int(n_u),
                "pool_size": int(pool_idx.size),
                "x_set": self.config.x_set,
                "beam_width": int(self.config.dp_beam_width),
            },
        )


def select_mi_universe_points(
    *,
    candidate_points: QueryPoints,
    eval_points: Optional[QueryPoints],
    x_set: str,
    subsample_max_points: int,
    rng: np.random.Generator,
) -> QueryPoints:
    x_set = str(x_set).lower()
    if x_set == "candidate_grid":
        return candidate_points
    if x_set == "eval_grid":
        return eval_points if eval_points is not None else candidate_points
    if x_set == "subsample":
        base = candidate_points
        n = int(base.size)
        m = min(max(1, int(subsample_max_points)), n)
        idx = np.sort(rng.choice(n, size=m, replace=False))
        return subset_query_points(base, idx)
    raise ValueError("x_set must be one of {'candidate_grid','eval_grid','subsample'}.")


def subset_query_points(points: QueryPoints, indices: Sequence[int]) -> QueryPoints:
    idx = np.asarray(indices, dtype=int)
    lats = np.asarray(points.lats, dtype=float)[idx]
    lons = np.asarray(points.lons, dtype=float)[idx]
    times = None
    depths = None
    if points.times is not None:
        times = np.asarray(points.times)[idx]
    if points.depths is not None:
        depths = np.asarray(points.depths, dtype=float)[idx]
    return QueryPoints(
        lats=lats,
        lons=lons,
        times=times,
        depths=depths,
        metadata=dict(points.metadata),
    )


def predictive_covariance_from_model(model: Any, points: QueryPoints, *, jitter: float) -> np.ndarray:
    """
    Obtain a covariance matrix for MI computation from model outputs.
    """
    if hasattr(model, "predictive_covariance"):
        cov = np.asarray(model.predictive_covariance(points), dtype=float)
    else:
        pred = model.predict(points)
        std = np.asarray(pred.std if pred.std is not None else np.ones(points.size), dtype=float)
        std = np.maximum(std.ravel(), np.sqrt(jitter))
        coords = np.column_stack([np.asarray(points.lats, dtype=float), np.asarray(points.lons, dtype=float)])
        # Estimate a reasonable lengthscale from the spatial extent of the
        # query points (~10% of the domain diameter) instead of a hardcoded
        # 1.0 degree which is meaningless for most regions.
        diam = max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])), 0.01)
        ls_auto = max(0.1 * diam, 1e-4)
        cov = _rbf_from_std(coords, std, lengthscale=ls_auto)
    cov = _ensure_spd(cov, jitter=jitter)
    return cov


def stable_cholesky(K: np.ndarray, *, jitter: float = 1e-8, max_tries: int = 7) -> tuple[np.ndarray, float]:
    K = np.asarray(K, dtype=float)
    eye = np.eye(K.shape[0], dtype=float)
    j = float(max(jitter, 1e-12))
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(K + j * eye)
            return L, j
        except np.linalg.LinAlgError:
            j *= 10.0
    # Final robust fallback: project to PSD and retry with escalating jitter.
    K_psd = _ensure_spd(K, jitter=j)
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(K_psd + j * eye)
            return L, j
        except np.linalg.LinAlgError:
            j *= 10.0

    # Absolute fallback to diagonal SPD approximation (never fail numerically).
    d = np.diag(K_psd).copy()
    d = np.nan_to_num(d, nan=j, posinf=1e6, neginf=j)
    d = np.maximum(d, j)
    L = np.diag(np.sqrt(d))
    return L, j


def stable_logdet(K: np.ndarray, *, jitter: float = 1e-8) -> float:
    if K.size == 0:
        return 0.0
    L, _ = stable_cholesky(K, jitter=jitter)
    return float(2.0 * np.sum(np.log(np.diag(L))))


def compute_mutual_information_from_covariance(
    covariance: np.ndarray,
    *,
    selected_indices: Sequence[int],
    jitter: float = 1e-8,
) -> float:
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    S = np.asarray(sorted(set(int(i) for i in selected_indices)), dtype=int)
    if S.size == 0 or S.size >= n:
        return 0.0
    R = np.asarray([i for i in range(n) if i not in set(S.tolist())], dtype=int)
    if R.size == 0:
        return 0.0

    SS = cov[np.ix_(S, S)]
    SR = cov[np.ix_(S, R)]
    RR = cov[np.ix_(R, R)]
    L_RR, _ = stable_cholesky(RR, jitter=jitter)
    A = np.linalg.solve(L_RR.T, np.linalg.solve(L_RR, SR.T))
    S_cond_R = SS - SR @ A
    S_cond_R = 0.5 * (S_cond_R + S_cond_R.T)
    mi = 0.5 * (stable_logdet(SS, jitter=jitter) - stable_logdet(S_cond_R, jitter=jitter))
    return float(max(mi, 0.0))


def conditional_variance(
    covariance: np.ndarray,
    *,
    index: int,
    conditioned_on: Sequence[int],
    jitter: float = 1e-8,
) -> float:
    cov = np.asarray(covariance, dtype=float)
    i = int(index)
    S = np.asarray(sorted(set(int(x) for x in conditioned_on if int(x) != i)), dtype=int)
    v_ii = float(cov[i, i])
    if S.size == 0:
        return max(v_ii, jitter)
    K_SS = cov[np.ix_(S, S)]
    k_iS = cov[np.ix_([i], S)].reshape(-1)
    L, _ = stable_cholesky(K_SS, jitter=jitter)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, k_iS))
    var = float(v_ii - k_iS @ alpha)
    return max(var, jitter)


def _dp_select_sequence(
    covariance: np.ndarray,
    *,
    n_select: int,
    beam_width: int,
    jitter: float,
) -> tuple[list[int], list[float]]:
    n = int(covariance.shape[0])
    if n_select <= 0:
        return [], []

    precision = _safe_precision(covariance, jitter=jitter)
    marginal_var = np.diag(covariance)

    # Clip cond_all_var to at least 1% of marginal variance to prevent
    # extreme log-ratio gains when the covariance is ill-conditioned.
    prec_diag = np.diag(precision)
    cond_all_var = np.where(
        prec_diag > jitter,
        1.0 / prec_diag,
        marginal_var,
    )
    cond_all_var = np.maximum(cond_all_var, 0.01 * np.maximum(marginal_var, jitter))

    def gain(idx: int, S_prev: Sequence[int]) -> float:
        var_cond_prev = conditional_variance(
            covariance,
            index=idx,
            conditioned_on=S_prev,
            jitter=jitter,
        )
        denom = max(float(cond_all_var[idx]), jitter)
        g = float(0.5 * np.log(max(var_cond_prev, jitter) / denom))
        return min(g, 30.0)  # clamp to prevent numerical blowup

    # Stage 1 initialization: V1(x) = I(Zx ; Z_{X\\{x}})
    states: list[tuple[tuple[int, ...], float, tuple[float, ...]]] = []
    for x in range(n):
        g = gain(x, [])
        states.append(((x,), g, (g,)))
    states.sort(key=lambda t: t[1], reverse=True)
    states = states[:beam_width]

    for _stage in range(2, n_select + 1):
        expanded: list[tuple[tuple[int, ...], float, tuple[float, ...]]] = []
        for seq, score, gains in states:
            used = set(seq)
            for x in range(n):
                if x in used:
                    continue
                g = gain(x, seq)
                expanded.append((seq + (x,), score + g, gains + (g,)))
        if not expanded:
            break
        expanded.sort(key=lambda t: t[1], reverse=True)
        states = expanded[:beam_width]

    best_seq, _best_score, best_gains = states[0]
    return list(best_seq), list(best_gains)


def _rbf_from_std(coords: np.ndarray, std: np.ndarray, *, lengthscale: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    std = np.asarray(std, dtype=float).ravel()
    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    corr = np.exp(-0.5 * d2 / max(lengthscale**2, 1e-12))
    return corr * np.outer(std, std)


def _ensure_spd(K: np.ndarray, *, jitter: float) -> np.ndarray:
    K = np.array(K, dtype=float, copy=True)  # copy to avoid mutating caller's array
    K = np.nan_to_num(K, nan=0.0, posinf=1e12, neginf=-1e12)
    K = np.clip(K, -1e12, 1e12)
    K = 0.5 * (K + K.T)
    if K.size == 0:
        return K
    # Clip diagonal to positive floor first.
    d = np.diag(K).copy()
    d = np.maximum(d, jitter)
    np.fill_diagonal(K, d)
    try:
        # Cheap path.
        np.linalg.cholesky(K + jitter * np.eye(K.shape[0], dtype=float))
        return K + jitter * np.eye(K.shape[0], dtype=float)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(K)
        w = np.nan_to_num(w, nan=jitter, posinf=1e12, neginf=jitter)
        w = np.maximum(w, jitter)
        K_psd = V @ np.diag(w) @ V.T
        K_psd = 0.5 * (K_psd + K_psd.T)
        K_psd = np.nan_to_num(K_psd, nan=0.0, posinf=1e12, neginf=-1e12)
        K_psd = np.clip(K_psd, -1e12, 1e12)
        K_psd = K_psd + jitter * np.eye(K_psd.shape[0], dtype=float)
        return K_psd


def _safe_precision(covariance: np.ndarray, *, jitter: float) -> np.ndarray:
    cov = _ensure_spd(covariance, jitter=jitter)
    eye = np.eye(cov.shape[0], dtype=float)
    j = float(max(jitter, 1e-12))
    for _ in range(6):
        try:
            return np.linalg.inv(cov + j * eye)
        except np.linalg.LinAlgError:
            j *= 10.0
    # Pseudo-inverse fallback.
    return np.linalg.pinv(cov + j * eye)

