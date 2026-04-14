from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .types import QueryPoints, Scenario


@dataclass
class EvalGrid:
    """
    Dense evaluation grid over a region, represented as QueryPoints.

    This wrapper carries minimal metadata so that downstream components can
    log grid construction parameters alongside the raw coordinates.
    """

    query_points: QueryPoints
    scenario: Optional[Scenario] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.query_points.size


def _region_from_scenario(scenario: Scenario) -> Mapping[str, float]:
    if not scenario.region:
        raise ValueError("Scenario.region must be set for EvalGrid construction.")
    return scenario.region


def _eval_time_from_config(
    scenario: Scenario,
    eval_cfg: Mapping[str, Any],
) -> Optional[np.datetime64]:
    if "eval_time" in eval_cfg:
        return np.datetime64(eval_cfg["eval_time"])
    if scenario.time_range is not None:
        return scenario.time_range[0]
    return None


def build_eval_grid(
    scenario: Scenario,
    eval_cfg: Mapping[str, Any],
    *,
    rng: Optional[np.random.Generator] = None,
) -> EvalGrid:
    """
    Build a rectilinear evaluation grid over the scenario region.

    Configuration
    -------------
    Expects keys under ``eval_cfg``:
      - ``grid_n_lat``, ``grid_n_lon`` (ints), or
      - ``grid_resolution_km`` (float) to derive a resolution.
      - optional ``max_points`` (int, default 10_000).
      - optional ``subsample_strategy``: ``"random"`` or ``"stratified"``.
    """
    rng = rng or np.random.default_rng()
    region = _region_from_scenario(scenario)
    lat_min = float(region["lat_min"])
    lat_max = float(region["lat_max"])
    lon_min = float(region["lon_min"])
    lon_max = float(region["lon_max"])

    n_lat = eval_cfg.get("grid_n_lat")
    n_lon = eval_cfg.get("grid_n_lon")
    if n_lat is None or n_lon is None:
        # Approximate grid size from target spatial resolution in km.
        res_km = float(eval_cfg.get("grid_resolution_km", 50.0))
        # Rough conversion: 1 degree ~ 111 km.
        dlat = lat_max - lat_min
        dlon = lon_max - lon_min
        n_lat = max(2, int(round((dlat * 111.0) / res_km)))
        n_lon = max(2, int(round((dlon * 111.0) / res_km)))
    else:
        n_lat = int(n_lat)
        n_lon = int(n_lon)

    # Depth grid (optional).
    n_depth = eval_cfg.get("grid_n_depth")
    use_depth = False
    if n_depth is not None and int(n_depth) > 1 and scenario.depth_range is not None:
        use_depth = True
        n_depth = int(n_depth)
        depth_min, depth_max = scenario.depth_range
        depth_arr = np.linspace(float(depth_min), float(depth_max), n_depth)
    elif scenario.depth_range is not None and n_depth is None:
        # Single depth layer at the midpoint when range is given but no grid size.
        use_depth = False

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    if use_depth:
        lat_grid, lon_grid, dep_grid = np.meshgrid(
            lats, lons, depth_arr, indexing="ij"
        )
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        dep_flat = dep_grid.ravel()
    else:
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        dep_flat = None

    eval_time = _eval_time_from_config(scenario, eval_cfg)
    if eval_time is not None:
        times = np.full(lat_flat.shape[0], np.datetime64(eval_time), dtype="datetime64[ns]")
    else:
        times = None

    qps = QueryPoints(lats=lat_flat, lons=lon_flat, times=times, depths=dep_flat)
    meta: dict[str, Any] = {
        "grid_n_lat": n_lat,
        "grid_n_lon": n_lon,
        "region": dict(region),
    }
    if use_depth:
        meta["grid_n_depth"] = n_depth
    grid = EvalGrid(
        query_points=qps,
        scenario=scenario,
        metadata=meta,
    )

    max_points = int(eval_cfg.get("max_points", 10_000))
    subsample_strategy = str(eval_cfg.get("subsample_strategy", "random"))
    if grid.size > max_points:
        grid = subsample_eval_grid(
            grid,
            max_points=max_points,
            strategy=subsample_strategy,
            rng=rng,
        )

    return grid


def subsample_eval_grid(
    grid: EvalGrid,
    *,
    max_points: int,
    strategy: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> EvalGrid:
    """
    Subsample an evaluation grid down to at most ``max_points`` locations.

    Parameters
    ----------
    strategy:
        - ``"random"``: uniform random subsampling.
        - ``"stratified"``: regular striding in the existing grid ordering.
    """
    n = grid.size
    if n <= max_points:
        return grid

    rng = rng or np.random.default_rng()
    strategy = strategy.lower()

    if strategy == "random":
        idx = rng.permutation(n)[:max_points]
    elif strategy == "stratified":
        stride = max(1, int(np.ceil(n / max_points)))
        idx = np.arange(0, n, stride)[:max_points]
    else:
        raise ValueError(f"Unknown subsample strategy: {strategy!r}")

    qp = grid.query_points
    lat_sub = np.asarray(qp.lats)[idx]
    lon_sub = np.asarray(qp.lons)[idx]
    times_sub = None
    depths_sub = None

    if qp.times is not None:
        times_sub = np.asarray(qp.times)[idx]
    if qp.depths is not None:
        depths_sub = np.asarray(qp.depths)[idx]

    qps_sub = QueryPoints(
        lats=lat_sub,
        lons=lon_sub,
        times=times_sub,
        depths=depths_sub,
        metadata=dict(qp.metadata),
    )

    new_meta = dict(grid.metadata)
    new_meta.update(
        {
            "subsampled": True,
            "subsample_strategy": strategy,
            "max_points": int(max_points),
            "original_size": n,
            "subsampled_size": qps_sub.size,
        }
    )

    return EvalGrid(query_points=qps_sub, scenario=grid.scenario, metadata=new_meta)

