"""AdaPP — Adaptive Path Planning using Sparse Gaussian Processes.

Implements Algorithm 1 from Mishra, Chitre & Swarup (2018):

  "Online Informative Path Planning using Sparse Gaussian Processes"

The planner operates on a :class:`CellGrid` decomposition of the survey
area and uses the variance from any :class:`FieldBeliefModel` (typically
``PseudoInputGPFieldModel``) to drive waypoint selection.

Key components
--------------
- **Single-robot DP** (Eqs. 8–9): value iteration on cells with reward
  R(c, a) = σ²_{c'} / ||c - c'||.
- **θ-simulation** (Eq. 10): for each candidate action, simulate future
  planning using DP to estimate the potential of reducing uncertainty
  with remaining time.
- **Lawnmower baseline**: serpentine coverage path within time budget.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.cell_decomposition import Cell, CellGrid
from oceanbench_core.types import ObservationBatch, QueryPoints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaPPConfig:
    """Configuration for the AdaPP planner."""

    # Cell decomposition
    n_cell_lat: int = 10
    n_cell_lon: int = 10
    initial_variance: float = 1.0       # κ at t=0

    # DP (Eqs. 8–9)
    gamma: float = 0.9                  # discount factor
    dp_max_iters: int = 100             # value iteration convergence
    dp_tol: float = 1e-4                # convergence tolerance

    # Time-constrained planning (Eq. 10)
    eta: float = 0.5                    # weight on θ_{T-t}
    noise_variance: float = 0.01        # σ² after visiting a cell

    # Action space
    connectivity: str = "4"             # "4" or "8"

    # Robot
    speed_mps: float = 1.0

    # Hyperparameter fitting
    hyperparams_mode: str = "initial_only"  # none | initial_only | periodic
    hyperparams_periodic_interval: int = 5

    # SPGP / field model
    variance_backend: str = "spgp_fitc"  # spgp_fitc (paper) | svgp_gpytorch
    n_pseudo: int = 50
    refit_interval: int = 1             # refit model every N steps

    seed: int = 42

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "AdaPPConfig":
        adapp = dict(cfg.get("adapp", cfg))
        hp = dict(adapp.get("hyperparams", {}))
        return cls(
            n_cell_lat=int(adapp.get("cell_resolution", [10, 10])[0]
                           if isinstance(adapp.get("cell_resolution"), list)
                           else adapp.get("n_cell_lat", 10)),
            n_cell_lon=int(adapp.get("cell_resolution", [10, 10])[1]
                           if isinstance(adapp.get("cell_resolution"), list)
                           else adapp.get("n_cell_lon", 10)),
            initial_variance=float(adapp.get("initial_variance", 1.0)),
            gamma=float(adapp.get("gamma", 0.9)),
            dp_max_iters=int(adapp.get("dp_max_iters", 100)),
            dp_tol=float(adapp.get("dp_tol", 1e-4)),
            eta=float(adapp.get("eta", 0.5)),
            noise_variance=float(adapp.get("noise_variance", 0.01)),
            connectivity=str(adapp.get("connectivity", adapp.get("action_space", "4"))),
            speed_mps=float(adapp.get("speed_mps", 1.0)),
            hyperparams_mode=str(hp.get("mode", adapp.get("hyperparams_mode", "initial_only"))),
            hyperparams_periodic_interval=int(hp.get("periodic_interval", 5)),
            variance_backend=str(adapp.get("variance_backend", "spgp_fitc")),
            n_pseudo=int(adapp.get("n_pseudo", 50)),
            refit_interval=int(adapp.get("refit_interval", 1)),
            seed=int(cfg.get("seed", adapp.get("seed", 42))),
        )


def build_field_model(config: AdaPPConfig, seed: int = 42) -> Any:
    """Factory to create the field model based on variance_backend config.

    Returns a FieldBeliefModel instance (either SPGPFITCFieldModel or
    PseudoInputGPFieldModel).
    """
    model_cfg = {
        "n_pseudo": config.n_pseudo,
        "noise": config.noise_variance,
        "include_time": False,
        "include_depth": False,
        "use_scaling": True,
    }
    if config.variance_backend == "spgp_fitc":
        from oceanbench_models.belief.field.spgp_fitc import SPGPFITCFieldModel
        return SPGPFITCFieldModel(model_cfg, seed=seed)
    elif config.variance_backend in ("svgp_gpytorch", "pseudo_input_gp"):
        from oceanbench_models.belief.field.pseudo_input_gp import PseudoInputGPFieldModel
        return PseudoInputGPFieldModel(model_cfg, seed=seed)
    else:
        raise ValueError(f"Unknown variance_backend: {config.variance_backend!r}")


# ---------------------------------------------------------------------------
# Single-robot DP  (Eqs. 8–9)
# ---------------------------------------------------------------------------


def _dp_value_iteration(
    cell_grid: CellGrid,
    gamma: float,
    connectivity: str,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    """Run value iteration on the cell grid.

    Returns
    -------
    V : 1-D array of shape (n_cells,) — value for each cell (flat index).
    """
    n = cell_grid.n_rows * cell_grid.n_cols
    V = np.zeros(n)

    for _ in range(max_iters):
        V_new = np.zeros(n)
        for idx in range(n):
            cell = cell_grid.cell_from_flat(idx)
            neighbors = cell_grid.neighbors(cell.row, cell.col, connectivity)
            if not neighbors:
                V_new[idx] = 0.0
                continue
            best = -np.inf
            for nb in neighbors:
                nb_idx = cell_grid.cell_index(nb.row, nb.col)
                dist = cell_grid.cell_distance(cell, nb)
                if dist < 1e-12:
                    dist = 1e-12
                reward = nb.variance / dist
                val = reward + gamma * V[nb_idx]
                if val > best:
                    best = val
            V_new[idx] = best
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def _dp_policy_action(
    cell_grid: CellGrid,
    V: np.ndarray,
    cell: Cell,
    gamma: float,
    connectivity: str,
) -> Optional[Cell]:
    """Return the best neighbor cell according to DP policy (Eq. 9)."""
    neighbors = cell_grid.neighbors(cell.row, cell.col, connectivity)
    if not neighbors:
        return None
    best_val = -np.inf
    best_nb = None
    for nb in neighbors:
        nb_idx = cell_grid.cell_index(nb.row, nb.col)
        dist = cell_grid.cell_distance(cell, nb)
        if dist < 1e-12:
            dist = 1e-12
        reward = nb.variance / dist
        val = reward + gamma * V[nb_idx]
        if val > best_val:
            best_val = val
            best_nb = nb
    return best_nb


# ---------------------------------------------------------------------------
# θ-simulation  (Eq. 10 inner loop)
# ---------------------------------------------------------------------------


def _simulate_theta(
    cell_grid: CellGrid,
    start_cell: Cell,
    remaining_time: float,
    speed_mps: float,
    noise_variance: float,
    gamma: float,
    connectivity: str,
    dp_max_iters: int,
    dp_tol: float,
) -> float:
    """Estimate θ_{T-t}(a⁺) by simulating future planning from *start_cell*.

    1. Mark start_cell variance → noise_variance.
    2. Run DP value iteration.
    3. Greedily follow DP policy until time runs out.
    4. Accumulate total variance reduction.

    Returns the total accumulated variance reduction.
    """
    # Mark the action-destination cell as visited.
    old_var = start_cell.variance
    cell_grid.set_cell_variance(start_cell.row, start_cell.col, noise_variance)

    total_reduction = max(old_var - noise_variance, 0.0)
    current = start_cell
    time_left = remaining_time

    while time_left > 0:
        V = _dp_value_iteration(cell_grid, gamma, connectivity, dp_max_iters, dp_tol)
        next_cell = _dp_policy_action(cell_grid, V, current, gamma, connectivity)
        if next_cell is None:
            break

        travel_dist = cell_grid.cell_distance(current, next_cell)
        travel_time = travel_dist / max(speed_mps, 1e-12)
        if travel_time > time_left:
            break

        time_left -= travel_time
        var_before = next_cell.variance
        cell_grid.set_cell_variance(next_cell.row, next_cell.col, noise_variance)
        total_reduction += max(var_before - noise_variance, 0.0)
        current = next_cell

    return total_reduction


# ---------------------------------------------------------------------------
# AdaPP Planner
# ---------------------------------------------------------------------------


class AdaPPPlanner:
    """Adaptive Path Planner (Mishra et al. 2018, Algorithm 1).

    Parameters
    ----------
    graph:
        WaypointGraph defining the survey area grid.
    region:
        Region dict with lat_min/max, lon_min/max.
    field_model:
        Any FieldBeliefModel (typically PseudoInputGPFieldModel).
    config:
        AdaPPConfig or raw mapping.
    truth_field:
        Optional OceanTruthField for observation simulation.
    """

    def __init__(
        self,
        graph: WaypointGraph,
        region: Mapping[str, float],
        field_model: Any,
        config: AdaPPConfig | Mapping[str, Any],
        *,
        truth_field: Any = None,
    ) -> None:
        if isinstance(config, Mapping):
            config = AdaPPConfig.from_mapping(config)
        self.graph = graph
        self.region = dict(region)
        self.field_model = field_model
        self.config = config
        self.truth_field = truth_field
        self._rng = np.random.default_rng(config.seed)

        # Build cell grid from graph nodes.
        nodes = list(graph.graph.nodes)
        self._node_lats = np.array([float(graph.graph.nodes[n]["lat"]) for n in nodes])
        self._node_lons = np.array([float(graph.graph.nodes[n]["lon"]) for n in nodes])
        self._nodes = nodes

        self.cell_grid = CellGrid.from_region(
            region, config.n_cell_lat, config.n_cell_lon,
            self._node_lats, self._node_lons,
        )

        # Observation accumulator (for refitting).
        self._obs_lats: list[float] = []
        self._obs_lons: list[float] = []
        self._obs_values: list[float] = []
        self._variable: str = "temp"

        # Trajectory log.
        self._trajectory: list[dict[str, Any]] = []
        self._step = 0
        self._hyperparams_fitted = False

    # -- main entry point --------------------------------------------------

    def run_episode(
        self,
        start_lat: float,
        start_lon: float,
        time_budget: float,
        variable: str = "temp",
        prior_obs: Any = None,
    ) -> dict[str, Any]:
        """Run a full AdaPP episode (Algorithm 1).

        Parameters
        ----------
        start_lat, start_lon:
            Starting coordinates.
        time_budget:
            Total mission time T in seconds.
        variable:
            Variable name for observations.
        prior_obs:
            Optional ObservationBatch of shared prior data to warm-start the
            field model before planning.

        Returns
        -------
        Dict with keys: trajectory, observations, n_steps, time_used.
        """
        self._variable = variable
        cfg = self.config
        t = 0.0
        T = time_budget

        # Initialization (Alg. 1, line 1).
        self.cell_grid.initialize(cfg.initial_variance)

        # Warm-start with shared prior observations if provided.
        if prior_obs is not None and prior_obs.size >= 3:
            self._obs_lats = list(np.asarray(prior_obs.lats, dtype=float))
            self._obs_lons = list(np.asarray(prior_obs.lons, dtype=float))
            self._obs_values = list(np.asarray(prior_obs.values, dtype=float))
            self._refit_model()

        # Find starting cell.
        current_cell = self.cell_grid.cell_containing_point(start_lat, start_lon)
        if current_cell is None:
            current_cell = self.cell_grid.cell(0, 0)

        self._step = 0
        self._trajectory = []

        while t < T:
            # Construct action set (Alg. 1, line 3).
            neighbors = self.cell_grid.neighbors(
                current_cell.row, current_cell.col, cfg.connectivity,
            )
            if not neighbors:
                break

            # Evaluate each candidate action (Alg. 1, lines 4–6).
            best_score = -np.inf
            best_nb = neighbors[0]

            for nb in neighbors:
                # U(a⁺) = variance of destination cell.
                U = nb.variance

                # Travel time for this action.
                travel_dist = self.cell_grid.cell_distance(current_cell, nb)
                travel_time = travel_dist / max(cfg.speed_mps, 1e-12)
                remaining_after = T - t - travel_time
                if remaining_after < 0:
                    continue  # Can't reach this cell in time.

                # θ_{T-t}(a⁺): simulate future planning (Alg. 1, line 5).
                snapshot = self.cell_grid.copy_variances()
                theta = _simulate_theta(
                    self.cell_grid,
                    nb,
                    remaining_after,
                    cfg.speed_mps,
                    cfg.noise_variance,
                    cfg.gamma,
                    cfg.connectivity,
                    cfg.dp_max_iters,
                    cfg.dp_tol,
                )
                self.cell_grid.restore_variances(snapshot)

                score = U + cfg.eta * theta
                if score > best_score:
                    best_score = score
                    best_nb = nb

            # Take action (Alg. 1, lines 7–9).
            travel_dist = self.cell_grid.cell_distance(current_cell, best_nb)
            travel_time = travel_dist / max(cfg.speed_mps, 1e-12)
            if t + travel_time > T:
                break

            t += travel_time
            obs_lat, obs_lon = best_nb.centroid

            # Collect observation.
            obs_value = self._observe(obs_lat, obs_lon)
            self._obs_lats.append(obs_lat)
            self._obs_lons.append(obs_lon)
            self._obs_values.append(obs_value)

            self._trajectory.append({
                "step": self._step,
                "lat": obs_lat,
                "lon": obs_lon,
                "value": obs_value,
                "cell_row": best_nb.row,
                "cell_col": best_nb.col,
                "time": t,
                "cell_variance_before": best_nb.variance,
            })
            self._step += 1

            # Refit model (Alg. 1, lines 10–12).
            if self._step % cfg.refit_interval == 0 and len(self._obs_values) >= 3:
                self._refit_model()
                # Re-decompose (Alg. 1, line 13).
                self.cell_grid.update_from_model(self.field_model)
            else:
                # At minimum, mark visited cell variance reduced.
                self.cell_grid.set_cell_variance(
                    best_nb.row, best_nb.col, cfg.noise_variance,
                )

            current_cell = best_nb

        return {
            "trajectory": list(self._trajectory),
            "observations": ObservationBatch(
                lats=np.array(self._obs_lats),
                lons=np.array(self._obs_lons),
                values=np.array(self._obs_values),
                variable=self._variable,
            ) if self._obs_values else None,
            "n_steps": self._step,
            "time_used": t,
        }

    # -- internal ----------------------------------------------------------

    def _observe(self, lat: float, lon: float) -> float:
        """Get an observation at (lat, lon) from truth or synthetic."""
        if self.truth_field is not None:
            qp = QueryPoints(lats=np.array([lat]), lons=np.array([lon]))
            try:
                val = np.asarray(
                    self.truth_field.query_array(qp, method="linear", bounds_mode="nan"),
                    dtype=float,
                ).ravel()
                # Take first finite value (surface).
                finite = val[np.isfinite(val)]
                if finite.size > 0:
                    noise = self._rng.normal(0, np.sqrt(self.config.noise_variance))
                    return float(finite[0]) + noise
            except Exception:
                pass
        # Synthetic fallback: predict from model + noise.
        if self.field_model.is_fitted:
            qp = QueryPoints(lats=np.array([lat]), lons=np.array([lon]))
            pred = self.field_model.predict(qp)
            noise = self._rng.normal(0, np.sqrt(self.config.noise_variance))
            return float(pred.mean[0]) + noise
        return self._rng.normal(0, 1)

    def _refit_model(self) -> None:
        """Refit the field model with all accumulated observations."""
        cfg = self.config
        obs = ObservationBatch(
            lats=np.array(self._obs_lats),
            lons=np.array(self._obs_lons),
            values=np.array(self._obs_values),
            variable=self._variable,
        )

        should_fit_hyperparams = False
        if cfg.hyperparams_mode == "initial_only" and not self._hyperparams_fitted:
            should_fit_hyperparams = True
            self._hyperparams_fitted = True
        elif cfg.hyperparams_mode == "periodic":
            if self._step % cfg.hyperparams_periodic_interval == 0:
                should_fit_hyperparams = True
        # mode == "none": never fit hyperparams, use config defaults.

        if should_fit_hyperparams:
            # Full refit (hyperparams + model).
            self.field_model.reset()
            self.field_model.fit(obs)
        else:
            # Refit model with fixed hyperparams by resetting and refitting.
            # The model's config retains the hyperparams from last fit.
            self.field_model.reset()
            self.field_model.fit(obs)

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        return list(self._trajectory)

    def reset(self) -> None:
        self._obs_lats.clear()
        self._obs_lons.clear()
        self._obs_values.clear()
        self._trajectory.clear()
        self._step = 0
        self._hyperparams_fitted = False


# ---------------------------------------------------------------------------
# Lawnmower baseline (clean wrapper, no Binney dependency)
# ---------------------------------------------------------------------------


class LawnmowerBaseline:
    """Serpentine lawnmower coverage path within a time budget.

    Reuses the serpentine ordering logic but does not depend on
    ``BinneyObjective`` or the Binney planner interface.
    """

    def __init__(
        self,
        graph: WaypointGraph,
        speed_mps: float = 1.0,
    ) -> None:
        self.graph = graph
        self.speed_mps = speed_mps

    def _infer_grid_shape(self) -> tuple[int, int]:
        n = self.graph.graph.number_of_nodes()
        candidates = []
        for a in range(1, int(np.sqrt(n)) + 1):
            if n % a == 0:
                candidates.append((a, n // a))
        if not candidates:
            return n, 1
        return min(candidates, key=lambda ab: abs(ab[0] - ab[1]))

    def _lawnmower_order(self) -> list[int]:
        n_lat, n_lon = self._infer_grid_shape()
        order: list[int] = []
        for i in range(n_lat):
            if i % 2 == 0:
                row = [i * n_lon + j for j in range(n_lon)]
            else:
                row = [i * n_lon + j for j in reversed(range(n_lon))]
            order.extend(row)
        return order

    def plan(
        self,
        start_node: int,
        time_budget: float,
    ) -> list[int]:
        """Return a list of node ids forming a lawnmower path within budget."""
        order = self._lawnmower_order()
        # Start from start_node.
        waypoints = [start_node] + [n for n in order if n != start_node]

        path = [start_node]
        budget = time_budget
        for nxt in waypoints[1:]:
            current = path[-1]
            try:
                local = self.graph.shortest_path(current, int(nxt))
            except Exception:
                break
            if not local or local[0] != current:
                break
            seg_time = self.graph.path_travel_time(local)
            if seg_time > budget:
                break
            path.extend(local[1:])
            budget -= seg_time
        return path

    def run_episode(
        self,
        start_lat: float,
        start_lon: float,
        time_budget: float,
        truth_field: Any = None,
        field_model: Any = None,
        variable: str = "temp",
        noise_variance: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> dict[str, Any]:
        """Run a full lawnmower episode, collecting observations along the path."""
        if rng is None:
            rng = np.random.default_rng(42)

        # Find nearest start node.
        nodes = list(self.graph.graph.nodes)
        node_lats = np.array([float(self.graph.graph.nodes[n]["lat"]) for n in nodes])
        node_lons = np.array([float(self.graph.graph.nodes[n]["lon"]) for n in nodes])
        dists = np.hypot(node_lats - start_lat, node_lons - start_lon)
        start_node = nodes[int(np.argmin(dists))]

        path = self.plan(start_node, time_budget)

        # Collect observations at each node.
        obs_lats, obs_lons, obs_values = [], [], []
        trajectory = []
        t = 0.0

        for i, node_id in enumerate(path):
            lat = float(self.graph.graph.nodes[node_id]["lat"])
            lon = float(self.graph.graph.nodes[node_id]["lon"])

            # Travel time.
            if i > 0:
                prev = path[i - 1]
                try:
                    edge = self.graph.edge_attributes(prev, node_id)
                    t += float(edge["time_s"])
                except Exception:
                    pass

            if t > time_budget:
                break

            # Observe.
            if truth_field is not None:
                qp = QueryPoints(lats=np.array([lat]), lons=np.array([lon]))
                try:
                    val = float(truth_field.query_array(qp)[0])
                except Exception:
                    val = rng.normal(0, 1)
            else:
                val = rng.normal(20, 1)

            obs_lats.append(lat)
            obs_lons.append(lon)
            obs_values.append(val)
            trajectory.append({"step": i, "lat": lat, "lon": lon, "value": val, "time": t})

        observations = None
        if obs_values:
            observations = ObservationBatch(
                lats=np.array(obs_lats), lons=np.array(obs_lons),
                values=np.array(obs_values), variable=variable,
            )

        return {
            "trajectory": trajectory,
            "path": path,
            "observations": observations,
            "n_steps": len(trajectory),
            "time_used": t,
        }
