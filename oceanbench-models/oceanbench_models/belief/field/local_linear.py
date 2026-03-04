from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction
from .utils import observation_batch_to_numpy, query_points_to_numpy


@dataclass
class LocalLinearConfig:
    """
    Configuration for the local linear regression baseline.

    Parameters
    ----------
    k_neighbors:
        Number of nearest neighbors to use for each query.
    include_time:
        Whether to include time as a feature when available.
    include_depth:
        Whether to include depth as a feature when available.
    """

    k_neighbors: int = 20
    include_time: bool = True
    include_depth: bool = True


class LocalLinearFieldModel(FieldBeliefModel):
    """
    Simple local regression baseline for field prediction using scikit-learn.

    Internally uses `KNeighborsRegressor` with distance-based weights. This
    preserves the FieldBeliefModel interface while delegating the neighbor
    search and weighting to a standard library implementation.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        cfg = config or {}
        self._cfg = LocalLinearConfig(
            k_neighbors=int(cfg.get("k_neighbors", 20)),
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
        )

        self._X: Optional[ArrayLike] = None
        self._y: Optional[ArrayLike] = None
        self._knn: Optional[KNeighborsRegressor] = None
        self._variable: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_uncertainty(self) -> bool:
        # This baseline does not provide rigorous uncertainty estimates.
        return False

    @property
    def supports_online_update(self) -> bool:
        # We keep the implementation simple and encourage refitting for now.
        return False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        observations: ObservationBatch,
        *,
        scenario: Optional[Scenario] = None,
    ) -> None:
        X, y = observation_batch_to_numpy(
            observations,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )

        if X.shape[0] < 1:
            raise RuntimeError("LocalLinearFieldModel requires at least one observation.")

        k = min(self._cfg.k_neighbors, X.shape[0])
        self._knn = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
        )
        self._knn.fit(X, y)

        self._X = X
        self._y = y
        self._variable = observations.variable

        self._mark_fitted()

    def update(self, observations: ObservationBatch) -> None:
        """
        LocalLinearFieldModel does not implement efficient online updates.

        For clarity, we raise an error here rather than silently refitting
        from partial data. Call `fit` again with the desired observation set.
        """

        raise NotImplementedError(
            "LocalLinearFieldModel does not support online updates; "
            "please refit with the combined observations."
        )

    def predict(self, query_points: QueryPoints) -> FieldPrediction:
        if not self.is_fitted or self._knn is None:
            raise RuntimeError("Model must be fitted before prediction.")

        Xq = query_points_to_numpy(
            query_points,
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        preds = self._knn.predict(Xq).astype(float).reshape(-1)
        return FieldPrediction(mean=preds, std=None, metadata={})

    def reset(self) -> None:
        super().reset()
        self._X = None
        self._y = None
        self._knn = None
        self._variable = None

