from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core.types import QueryPoints, Scenario


@dataclass
class GridSpec:
    n_lat: int
    n_lon: int
    max_points: int
    subsample_strategy: str


def build_candidate_points(
    scenario: Scenario,
    candidates_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> QueryPoints:
    """
    Build candidate planning points from config.
    """
    ctype = str(candidates_cfg.get("type", "grid")).lower()
    rng = np.random.default_rng(int(seed))
    region = scenario.region
    if not region:
        raise ValueError("Scenario.region is required for candidate construction.")

    if ctype == "grid":
        n_lat = int(candidates_cfg.get("grid", {}).get("n_lat", 50))
        n_lon = int(candidates_cfg.get("grid", {}).get("n_lon", 50))
        q = _regular_grid_points(
            lat_min=float(region["lat_min"]),
            lat_max=float(region["lat_max"]),
            lon_min=float(region["lon_min"]),
            lon_max=float(region["lon_max"]),
            n_lat=n_lat,
            n_lon=n_lon,
        )
    elif ctype == "random":
        n_points = int(candidates_cfg.get("max_points", 2_500))
        lats = rng.uniform(float(region["lat_min"]), float(region["lat_max"]), n_points)
        lons = rng.uniform(float(region["lon_min"]), float(region["lon_max"]), n_points)
        q = QueryPoints(lats=lats, lons=lons)
    elif ctype == "graph":
        # Graph candidates can be passed directly in config (from graph nodes).
        pts = np.asarray(candidates_cfg.get("points", []), dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] > 0:
            q = QueryPoints(lats=pts[:, 0], lons=pts[:, 1])
        else:
            # Safe fallback to grid if graph points are unavailable.
            n_lat = int(candidates_cfg.get("grid", {}).get("n_lat", 50))
            n_lon = int(candidates_cfg.get("grid", {}).get("n_lon", 50))
            q = _regular_grid_points(
                lat_min=float(region["lat_min"]),
                lat_max=float(region["lat_max"]),
                lon_min=float(region["lon_min"]),
                lon_max=float(region["lon_max"]),
                n_lat=n_lat,
                n_lon=n_lon,
            )
    else:
        raise ValueError("candidates.type must be one of {'grid','random','graph'}.")

    max_points = int(candidates_cfg.get("max_points", q.size))
    if q.size > max_points:
        q = cap_query_points(
            q,
            max_points=max_points,
            strategy=str(candidates_cfg.get("subsample_strategy", "random")),
            seed=int(candidates_cfg.get("seed", seed)),
        )
    return q


def build_eval_spatial_grid(
    scenario: Scenario,
    eval_cfg: Mapping[str, Any],
    *,
    seed: int,
) -> QueryPoints:
    """
    Build a fixed-in-space dense spatial grid for evaluation.
    """
    region = scenario.region
    if not region:
        raise ValueError("Scenario.region is required for eval grid.")

    grid_cfg = dict(eval_cfg.get("grid", eval_cfg))
    fixed = bool(grid_cfg.get("fixed", True))
    if not fixed:
        raise ValueError("Persistent sampling evaluation requires eval.grid.fixed=true.")

    n_lat = grid_cfg.get("n_lat")
    n_lon = grid_cfg.get("n_lon")
    if n_lat is None or n_lon is None:
        res_km = float(grid_cfg.get("resolution_km", 25.0))
        dlat = float(region["lat_max"]) - float(region["lat_min"])
        dlon = float(region["lon_max"]) - float(region["lon_min"])
        n_lat = max(2, int(round((dlat * 111.0) / max(res_km, 1e-6))))
        n_lon = max(2, int(round((dlon * 111.0) / max(res_km, 1e-6))))
    q = _regular_grid_points(
        lat_min=float(region["lat_min"]),
        lat_max=float(region["lat_max"]),
        lon_min=float(region["lon_min"]),
        lon_max=float(region["lon_max"]),
        n_lat=int(n_lat),
        n_lon=int(n_lon),
    )
    max_points = int(eval_cfg.get("max_points", 10_000))
    if q.size > max_points:
        q = cap_query_points(
            q,
            max_points=max_points,
            strategy=str(eval_cfg.get("subsample_strategy", "random")),
            seed=int(seed),
        )
    return q


def resolve_eval_times(
    eval_cfg: Mapping[str, Any],
    *,
    scenario: Scenario,
    provider_times: Optional[np.ndarray] = None,
    current_time: Optional[np.datetime64] = None,
) -> np.ndarray:
    """
    Resolve evaluation time stamps according to eval.times mode.
    """
    times_mode = str(eval_cfg.get("times", "single")).lower()
    if times_mode == "single":
        if current_time is not None:
            return np.array([np.datetime64(current_time, "ns")], dtype="datetime64[ns]")
        if scenario.time_range is not None:
            return np.array([np.datetime64(scenario.time_range[0], "ns")], dtype="datetime64[ns]")
        if provider_times is not None and provider_times.size > 0:
            return np.array([np.datetime64(provider_times[0], "ns")], dtype="datetime64[ns]")
        raise ValueError("Cannot resolve eval single time: no current/scenario/provider time available.")

    if times_mode == "sequence":
        seq = eval_cfg.get("time_sequence", [])
        if seq:
            return np.asarray(seq, dtype="datetime64[ns]")
        if provider_times is not None and provider_times.size > 0:
            k = int(eval_cfg.get("sequence_length", 4))
            idx = np.linspace(0, provider_times.size - 1, min(k, provider_times.size)).astype(int)
            return np.asarray(provider_times[idx], dtype="datetime64[ns]")
        raise ValueError("eval.times='sequence' requires time_sequence or provider_times.")

    if times_mode == "all_in_window":
        if provider_times is not None and provider_times.size > 0:
            if scenario.time_range is None:
                return np.asarray(provider_times, dtype="datetime64[ns]")
            t0 = np.datetime64(scenario.time_range[0], "ns")
            t1 = np.datetime64(scenario.time_range[1], "ns")
            arr = np.asarray(provider_times, dtype="datetime64[ns]")
            mask = (arr >= t0) & (arr <= t1)
            return arr[mask]
        raise ValueError("eval.times='all_in_window' requires provider_times.")

    raise ValueError("eval.times must be one of {'single','sequence','all_in_window'}.")


def score_persistent_sampling(
    *,
    model: Any,
    truth_field: Any,
    eval_spatial_points: QueryPoints,
    eval_times: np.ndarray,
    time_mode: str,
    available_provider_times: Optional[np.ndarray] = None,
    route_points: Optional[np.ndarray] = None,
    route_metric: str = "haversine",
    speed_mps: float = 1.0,
    planning_time_s: float = 0.0,
    update_time_s: float = 0.0,
    n_replans: int = 0,
    bounds_mode: str = "clip",
    return_maps: bool = False,
) -> dict[str, Any]:
    """
    Evaluate model RMSE/MAE + uncertainty on fixed spatial grid across times.
    """
    lats = np.asarray(eval_spatial_points.lats, dtype=float)
    lons = np.asarray(eval_spatial_points.lons, dtype=float)
    depths = np.asarray(eval_spatial_points.depths, dtype=float) if eval_spatial_points.depths is not None else None
    times = np.asarray(eval_times, dtype="datetime64[ns]")
    if times.size == 0:
        raise ValueError("eval_times cannot be empty.")

    per_time: list[dict[str, Any]] = []
    map_frames: list[dict[str, Any]] = []
    rmses: list[float] = []
    maes: list[float] = []
    mean_stds: list[float] = []

    for t in times:
        t_eval = np.datetime64(t, "ns")
        if str(time_mode).lower() == "snap_to_provider" and available_provider_times is not None and available_provider_times.size > 0:
            t_eval = _snap_time(t_eval, np.asarray(available_provider_times, dtype="datetime64[ns]"))
            interp_method = "nearest"
        else:
            interp_method = "linear"

        q_t = QueryPoints(
            lats=lats,
            lons=lons,
            times=np.full(lats.shape[0], t_eval, dtype="datetime64[ns]"),
            depths=depths,
        )
        pred = model.predict(q_t)
        y_pred = np.asarray(pred.mean, dtype=float).ravel()
        y_std = np.asarray(pred.std, dtype=float).ravel() if pred.std is not None else np.full_like(y_pred, np.nan)
        y_true = np.asarray(
            truth_field.query_array(q_t, method=interp_method, bounds_mode=bounds_mode),
            dtype=float,
        ).ravel()
        if y_true.shape != y_pred.shape:
            # For datasets with depth dimension, use surface values.
            if y_true.size % y_pred.size == 0:
                y_true = y_true.reshape(y_pred.size, -1)[:, 0]
            else:
                raise ValueError("Truth and prediction shapes are incompatible during scoring.")

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask):
            err = y_pred[mask] - y_true[mask]
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            ms = float(np.nanmean(y_std[mask])) if y_std.size else float("nan")
        else:
            rmse = float("nan")
            mae = float("nan")
            ms = float("nan")

        rmses.append(rmse)
        maes.append(mae)
        mean_stds.append(ms)
        per_time.append(
            {
                "time": str(t_eval),
                "rmse": rmse,
                "mae": mae,
                "mean_std": ms,
            }
        )
        if return_maps:
            map_frames.append(
                {
                    "time": str(t_eval),
                    "truth": y_true.tolist(),
                    "pred": y_pred.tolist(),
                    "std": y_std.tolist() if y_std is not None else None,
                }
            )

    distance = 0.0
    travel_time = 0.0
    if route_points is not None and np.asarray(route_points).shape[0] >= 2:
        distance = route_distance(route_points, metric=route_metric)
        travel_time = float(distance / max(speed_mps, 1e-6))

    out = {
        "rmse": float(np.nanmean(rmses)),
        "mae": float(np.nanmean(maes)),
        "uncertainty_mean_std": float(np.nanmean(mean_stds)),
        "distance": float(distance),
        "travel_time": float(travel_time),
        "planning_time": float(planning_time_s),
        "update_time": float(update_time_s),
        "runtime": float(planning_time_s + update_time_s),
        "replans": int(n_replans),
        "eval_times_count": int(times.size),
        "eval_points_count": int(eval_spatial_points.size),
        "per_time": per_time,
    }
    if return_maps:
        out["map_frames"] = map_frames
        out["eval_spatial_lats"] = np.asarray(eval_spatial_points.lats, dtype=float).tolist()
        out["eval_spatial_lons"] = np.asarray(eval_spatial_points.lons, dtype=float).tolist()
    return out


def cap_query_points(
    points: QueryPoints,
    *,
    max_points: int,
    strategy: str,
    seed: int,
) -> QueryPoints:
    n = int(points.size)
    m = min(max(1, int(max_points)), n)
    if n <= m:
        return points
    rng = np.random.default_rng(int(seed))
    strategy = str(strategy).lower()
    if strategy == "random":
        idx = np.sort(rng.choice(n, size=m, replace=False))
    elif strategy == "stratified":
        # Deterministic, full-range coverage over flattened order.
        # Using a simple stride can bias toward the beginning when n/m < 2
        # (e.g., 14_400 -> 10_000 would truncate the tail of the region).
        idx = np.linspace(0, n - 1, num=m, dtype=int)
        idx = np.unique(idx)
        if idx.size < m:
            # Fill any rounding-induced gaps while preserving determinism.
            missing = np.setdiff1d(np.arange(n, dtype=int), idx, assume_unique=False)
            need = int(m - idx.size)
            fill = missing[:need]
            idx = np.sort(np.concatenate([idx, fill]))
        elif idx.size > m:
            idx = idx[:m]
    else:
        raise ValueError("subsample strategy must be one of {'random','stratified'}.")
    return QueryPoints(
        lats=np.asarray(points.lats, dtype=float)[idx],
        lons=np.asarray(points.lons, dtype=float)[idx],
        times=(np.asarray(points.times)[idx] if points.times is not None else None),
        depths=(np.asarray(points.depths, dtype=float)[idx] if points.depths is not None else None),
        metadata=dict(points.metadata),
    )


def route_distance(points: np.ndarray, *, metric: str = "haversine") -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("route points must have shape (n, 2).")
    if pts.shape[0] < 2:
        return 0.0
    metric = str(metric).lower()
    total = 0.0
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        if metric in {"haversine", "travel_time"}:
            total += _haversine_distance_m(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
        elif metric == "euclidean":
            total += float(np.linalg.norm(a - b))
        else:
            raise ValueError("route metric must be one of {'haversine','euclidean','travel_time'}.")
    return float(total)


def _regular_grid_points(
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
) -> QueryPoints:
    lats = np.linspace(lat_min, lat_max, int(n_lat))
    lons = np.linspace(lon_min, lon_max, int(n_lon))
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    return QueryPoints(lats=lat_grid.ravel(), lons=lon_grid.ravel())


def _snap_time(t: np.datetime64, available: np.ndarray) -> np.datetime64:
    arr = np.asarray(available, dtype="datetime64[ns]")
    if arr.size == 0:
        return np.datetime64(t, "ns")
    t_int = np.datetime64(t, "ns").astype("int64")
    a_int = arr.astype("int64")
    idx = int(np.argmin(np.abs(a_int - t_int)))
    return np.datetime64(arr[idx], "ns")


def _haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    r_earth = 6_371_000.0
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(max(1.0 - a, 0.0)))
    return float(r_earth * c)

