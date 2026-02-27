from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario


ArrayLike = np.ndarray


@dataclass
class FieldPrediction:
    """
    Prediction of a scalar field at a batch of query points.

    The core model API is deliberately NumPy-based and backend-agnostic.
    Higher layers (env/viz) are responsible for wrapping these arrays into
    xarray objects when coordinate labels are needed.
    """

    mean: ArrayLike
    std: Optional[ArrayLike] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FieldBeliefModel(ABC):
    """
    Common interface for all field belief (representation) models.

    Implementations should:
    - accept configuration and seeds in the constructor,
    - implement fit / update / predict consistently,
    - clearly document whether online updates and uncertainty are supported.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        self._seed: Optional[int] = seed
        self._rng = np.random.default_rng(seed)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Configuration and lifecycle flags
    # ------------------------------------------------------------------

    @property
    def config(self) -> Mapping[str, Any]:
        """Return the (possibly resolved) configuration for this model."""

        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted / initialized at least once."""

        return self._is_fitted

    @property
    @abstractmethod
    def supports_uncertainty(self) -> bool:
        """
        Whether this model is able to return predictive uncertainty.

        If False, the `std` in :class:`FieldPrediction` should be `None`.
        """

    @property
    @abstractmethod
    def supports_online_update(self) -> bool:
        """
        Whether this model is designed for true incremental updates.

        Models that simply refit from scratch on new data may still implement
        `update`, but should generally report `False` here so that evaluation
        code can distinguish them from genuinely online methods.
        """

    # ------------------------------------------------------------------
    # Core life-cycle
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        """
        Initialize the model from an initial batch of observations.

        This is the "cold start" operation. Implementations should call
        `self._mark_fitted()` when the model is ready for prediction.
        """

    @abstractmethod
    def update(self, observations: ObservationBatch) -> None:
        """
        Incrementally update the model with new observations.

        Models that do not support online updates may either:
        - raise a clear error, or
        - implement a documented "append + refit" strategy.
        """

    @abstractmethod
    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        """
        Predict at arbitrary query points.

        The returned arrays must have leading dimension equal to
        `query_points.size`. Additional trailing dimensions (e.g. for
        multi-output extensions) are not part of this milestone.
        """

    # Convenience wrappers for common usage patterns.

    def predict_mean(self, query_points: QueryPoints) -> ArrayLike:
        """Return only the predictive mean at query points."""

        return self.predict(query_points).mean

    def predict_std(self, query_points: QueryPoints) -> Optional[ArrayLike]:
        """
        Return only the predictive standard deviation at query points,
        or None if this model does not support uncertainty.
        """

        return self.predict(query_points).std

    # ------------------------------------------------------------------
    # Reproducibility / serialization
    # ------------------------------------------------------------------

    def seed(self, seed: int) -> None:
        """
        Reset the random seed used by this model.

        Concrete models should ensure all stochastic components respect this.
        """

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """
        Clear model state but preserve configuration.

        Subclasses overriding this method should call `super().reset()` and
        then clear any cached matrices, parameters, or internal buffers.
        """

        self._is_fitted = False

    def get_state(self) -> Mapping[str, Any]:
        """
        Return a minimal serializable representation of the model state.

        Default is empty; models with non-trivial state should override.
        """

        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        """
        Restore model state from a mapping returned by :meth:`get_state`.

        Default is a no-op; models with non-trivial state should override.
        """

        _ = state  # unused

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        seed: Optional[int] = None,
    ) -> "FieldBeliefModel":
        """Standard constructor hook from a config mapping."""

        return cls(config=config, seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_fitted(self, fitted: bool = True) -> None:
        """Helper for subclasses to update the fitted flag."""

        self._is_fitted = fitted

