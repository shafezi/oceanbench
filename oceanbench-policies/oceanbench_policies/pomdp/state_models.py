"""POMDP state, observation, action types and belief adapter for OceanBench."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from oceanbench_core.types import (
    ArrayLike,
    Observation,
    ObservationBatch,
    QueryPoints,
)
from oceanbench_models.belief.field.base import FieldBeliefModel, FieldPrediction


# ---------------------------------------------------------------------------
# POMDP primitives
# ---------------------------------------------------------------------------


@dataclass
class POMDPState:
    """Discrete state in the POMCP tree.

    Parameters
    ----------
    node_id:
        Index into the WaypointGraph node list (discrete location).
    time:
        Current simulation time (for dynamic truth fields).
    step:
        Step counter within the episode (0-indexed).
    remaining_budget:
        Seconds of travel budget remaining, or *None* for unlimited.
    metadata:
        Extensible dict for extra bookkeeping (e.g. depth, history).
    """

    node_id: int
    time: np.datetime64
    step: int = 0
    remaining_budget: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class POMDPObservation:
    """An observation received from the environment after taking an action.

    This is the scalar field value (possibly noisy) at the new location.
    """

    value: float
    noise_var: float
    lat: float
    lon: float
    time: Optional[np.datetime64] = None

    def to_observation(self, variable: str) -> Observation:
        """Convert to an OceanBench :class:`Observation`."""
        return Observation(
            lat=self.lat,
            lon=self.lon,
            value=self.value,
            variable=variable,
            time=self.time,
        )


@dataclass
class POMCPAction:
    """A discrete action = move to a neighbour node in the WaypointGraph."""

    target_node_id: int
    lat: float
    lon: float

    # Integer index into the local action list (useful for POMCP internals).
    action_index: int = 0


# ---------------------------------------------------------------------------
# Belief adapter
# ---------------------------------------------------------------------------


class BeliefAdapter:
    """Wraps any OceanBench :class:`FieldBeliefModel` as the POMCP belief/reward source.

    The reward at a candidate location is ``mean + c * std`` (the acquisition
    function from the paper).

    Parameters
    ----------
    field_model:
        An already-fitted (or fittable) OceanBench field model.
    variable:
        The scalar variable name (e.g. ``"temp"``).
    objective_c:
        Exploration coefficient in ``mean + c * std``.
    measurement_noise_var:
        Variance of observation noise added during simulation.
    """

    def __init__(
        self,
        field_model: FieldBeliefModel,
        variable: str,
        *,
        objective_c: float = 1.0,
        measurement_noise_var: float = 0.01,
    ) -> None:
        self.field_model = field_model
        self.variable = variable
        self.objective_c = objective_c
        self.measurement_noise_var = measurement_noise_var
        # Accumulate all observations for models that don't support online update.
        self._all_lats: list[float] = []
        self._all_lons: list[float] = []
        self._all_values: list[float] = []
        self._all_times: list[Any] = []

    def seed_observations(self, observations: ObservationBatch) -> None:
        """Populate the internal observation accumulator with initial data.

        Call this after fitting the model externally so that ``update()``
        can refit from the full history on models without online update.
        """
        self._all_lats.extend(observations.lats.tolist())
        self._all_lons.extend(observations.lons.tolist())
        self._all_values.extend(observations.values.tolist())
        if observations.times is not None:
            self._all_times.extend(observations.times.tolist())
        else:
            self._all_times.extend([None] * observations.size)

    # -- querying ----------------------------------------------------------

    def predict_at(
        self, lat: float, lon: float, time: Optional[np.datetime64] = None,
    ) -> FieldPrediction:
        """Predict mean and std at a single point."""
        qp = QueryPoints(
            lats=np.array([lat]),
            lons=np.array([lon]),
            times=np.array([time], dtype="datetime64[ns]") if time is not None else None,
        )
        return self.field_model.predict(qp)

    def reward_at(
        self, lat: float, lon: float, time: Optional[np.datetime64] = None,
    ) -> float:
        """Acquisition-function reward: ``mean + c * std``."""
        pred = self.predict_at(lat, lon, time)
        mean_val = float(pred.mean[0])
        std_val = float(pred.std[0]) if pred.std is not None else 0.0
        return mean_val + self.objective_c * std_val

    def predict_batch(self, query_points: QueryPoints) -> FieldPrediction:
        """Batch prediction forwarded to the underlying model."""
        return self.field_model.predict(query_points)

    # -- updating ----------------------------------------------------------

    def update(self, observation: POMDPObservation) -> None:
        """Incorporate a new observation into the belief model."""
        self._all_lats.append(observation.lat)
        self._all_lons.append(observation.lon)
        self._all_values.append(observation.value)
        self._all_times.append(observation.time)

        obs_batch = ObservationBatch(
            lats=np.array([observation.lat]),
            lons=np.array([observation.lon]),
            values=np.array([observation.value]),
            variable=self.variable,
            times=(
                np.array([observation.time], dtype="datetime64[ns]")
                if observation.time is not None
                else None
            ),
        )
        if self.field_model.supports_online_update and self.field_model.is_fitted:
            self.field_model.update(obs_batch)
        else:
            # Refit from all accumulated observations.
            self._refit_from_accumulated()

    def update_batch(self, observations: ObservationBatch) -> None:
        """Incorporate a batch of observations."""
        self._all_lats.extend(observations.lats.tolist())
        self._all_lons.extend(observations.lons.tolist())
        self._all_values.extend(observations.values.tolist())
        if observations.times is not None:
            self._all_times.extend(observations.times.tolist())
        else:
            self._all_times.extend([None] * observations.size)

        if self.field_model.supports_online_update and self.field_model.is_fitted:
            self.field_model.update(observations)
        else:
            self._refit_from_accumulated()

    def _refit_from_accumulated(self) -> None:
        """Refit the model from scratch using all accumulated observations."""
        has_time = any(t is not None for t in self._all_times)
        times_arr = None
        if has_time:
            times_arr = np.array(
                [t if t is not None else np.datetime64("NaT") for t in self._all_times],
                dtype="datetime64[ns]",
            )
        full_batch = ObservationBatch(
            lats=np.array(self._all_lats),
            lons=np.array(self._all_lons),
            values=np.array(self._all_values),
            variable=self.variable,
            times=times_arr,
        )
        self.field_model.reset()
        self.field_model.fit(full_batch)

    # -- lifecycle ---------------------------------------------------------

    def get_deep_copy(self) -> "BeliefAdapter":
        """Return an independent clone (used for POMCP rollout simulation)."""
        return copy.deepcopy(self)

    def reset(self) -> None:
        """Reset the underlying model state."""
        self.field_model.reset()

    @property
    def is_fitted(self) -> bool:
        return self.field_model.is_fitted
