from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from oceanbench_core import (
    EvalGrid,
    Scenario,
    WaypointGraph,
    build_eval_grid,
)
from oceanbench_core.sampling import MeasurementItem, h as sampling_fn
from oceanbench_models.belief.field.covariance_backends import (
    CovarianceBackend,
    build_covariance_backend,
)
from oceanbench_tasks.mapping.binney_objectives import (
    BinneyObjective,
    build_binney_objective,
)


@dataclass
class BinneyMappingTask:
    """
    Container wiring together graph, sampling, covariance backend, and objective.

    This object is designed to be lightweight and easily consumable by
    planners and runners. It does not implement the GRG algorithm itself.
    """

    scenario: Scenario
    graph: WaypointGraph
    eval_grid: EvalGrid
    covariance: CovarianceBackend
    objective: BinneyObjective
    sampling_config: Mapping[str, Any]
    time_config: Mapping[str, Any]

    @classmethod
    def from_configs(
        cls,
        scenario_cfg: Mapping[str, Any],
        task_cfg: Mapping[str, Any],
        *,
        provider: Any,
        rng: Optional[np.random.Generator] = None,
    ) -> "BinneyMappingTask":
        """
        Build a BinneyMappingTask from YAML-style configuration mappings.

        Parameters
        ----------
        scenario_cfg:
            Scenario configuration (region, time window, variable, etc.).
        task_cfg:
            Task configuration including graph, sampling, objective, covariance,
            and evaluation grid options.
        provider:
            DataProvider or compatible object used by empirical covariance
            backends.
        """
        rng = rng or np.random.default_rng(int(task_cfg.get("seed", 42)))

        # Construct Scenario.
        region_cfg = scenario_cfg.get("region", {})
        scenario = Scenario(
            name=scenario_cfg.get("name"),
            variable=scenario_cfg.get("variable", "temp"),
            region={
                "lat_min": float(region_cfg["lat_min"]),
                "lat_max": float(region_cfg["lat_max"]),
                "lon_min": float(region_cfg["lon_min"]),
                "lon_max": float(region_cfg["lon_max"]),
            },
            time_range=scenario_cfg.get("time_range"),
            depth_range=scenario_cfg.get("depth_range"),
            metadata=scenario_cfg.get("metadata", {}),
        )

        # Graph construction.
        graph_cfg = task_cfg.get("graph", {})
        graph_type = str(graph_cfg.get("type", "grid"))
        speed_mps = float(task_cfg.get("robot", {}).get("speed_mps", 1.0))
        seed = int(graph_cfg.get("seed", 0))

        if graph_type == "grid":
            n_lat = int(graph_cfg.get("grid", {}).get("n_lat", 10))
            n_lon = int(graph_cfg.get("grid", {}).get("n_lon", 10))
            graph = WaypointGraph.grid(
                region=scenario.region,
                n_lat=n_lat,
                n_lon=n_lon,
                speed_mps=speed_mps,
                seed=seed,
            )
        elif graph_type == "random_geometric":
            random_cfg = graph_cfg.get("random", {})
            n_nodes = int(random_cfg.get("n_nodes", 50))
            k = random_cfg.get("k")
            radius = random_cfg.get("radius")
            graph = WaypointGraph.random_geometric(
                region=scenario.region,
                n_nodes=n_nodes,
                k=int(k) if k is not None else None,
                radius=float(radius) if radius is not None else None,
                speed_mps=speed_mps,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown graph.type: {graph_type!r}")

        # Evaluation grid.
        eval_cfg = task_cfg.get("eval", {})
        eval_grid = build_eval_grid(scenario, eval_cfg, rng=rng)

        # Covariance backend.
        covariance = build_covariance_backend(
            task_cfg,
            eval_grid=eval_grid,
            provider=provider,
        )

        # Objective.
        # Features for Y are taken from the evaluation grid as [lat, lon(, time)].
        Y = np.column_stack(
            [
                np.asarray(eval_grid.query_points.lats, dtype=float),
                np.asarray(eval_grid.query_points.lons, dtype=float),
            ]
        )
        if eval_grid.query_points.times is not None:
            # Represent time as seconds since epoch for kernel backends.
            t = np.asarray(eval_grid.query_points.times, dtype="datetime64[ns]").astype("int64") / 1e9
            Y = np.column_stack([Y, t.astype(float)])

        noise_var = float(
            task_cfg.get("sampling", {}).get("measurement_noise_var", 1e-2)
        )
        obj_cfg = dict(task_cfg)
        obj_cfg.setdefault("objective", {})
        obj_cfg["objective"].setdefault("noise_var", noise_var)
        objective = build_binney_objective(
            obj_cfg,
            covariance=covariance,
            eval_features=Y,
        )

        sampling_config = task_cfg.get("sampling", {})
        time_config = task_cfg.get("time", {})

        return cls(
            scenario=scenario,
            graph=graph,
            eval_grid=eval_grid,
            covariance=covariance,
            objective=objective,
            sampling_config=sampling_config,
            time_config=time_config,
        )

    # ------------------------------------------------------------------
    # Helper methods for planners / runners
    # ------------------------------------------------------------------

    def sample_path(
        self,
        path: Sequence[int],
        tau: np.datetime64,
    ) -> list[MeasurementItem]:
        """
        Apply h(P, tau) using this task's sampling configuration.
        """
        return sampling_fn(
            path=path,
            tau=tau,
            graph=self.graph,
            sampling_cfg=self.sampling_config,
        )

    def features_from_measurements(
        self,
        items: Sequence[MeasurementItem],
    ) -> np.ndarray:
        """
        Convert a sequence of MeasurementItem to a feature matrix for objectives.

        The convention matches the covariance backends: columns are at least
        [lat, lon] and optionally a time column (seconds since epoch).
        """
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

