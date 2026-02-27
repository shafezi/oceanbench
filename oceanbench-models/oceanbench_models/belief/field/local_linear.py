from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.spatial import cKDTree

from oceanbench_core.types import ObservationBatch, QueryPoints, Scenario

from .base import ArrayLike, FieldBeliefModel, FieldPrediction


@dataclass
class LocalLinearConfig:
    """
    Configuration for the local linear regression baseline.

    Parameters
    ----------
    k_neighbors:
        Number of nearest neighbors to use for each query.
    min_neighbors:
        Minimum number of neighbors required to perform a local fit. If fewer
        are available, a simple average of the available neighbors is used.
    include_time:
        Whether to include time as a feature when available.
    include_depth:
        Whether to include depth as a feature when available.
    regularization:
        Small L2 regularization added to the local design matrix for numerical
        stability.
    """

    k_neighbors: int = 20
    min_neighbors: int = 3
    include_time: bool = True
    include_depth: bool = True
    regularization: float = 1e-6


class LocalLinearFieldModel(FieldBeliefModel):
    """
    Simple local linear regression baseline for field prediction.

    For each query point, this model:
    - finds k nearest observed neighbors in feature space,
    - fits a linear model of value as a function of coordinates in that
      neighborhood (with small L2 regularization),
    - evaluates the fitted model at the query point.

    This provides a fast, interpretable baseline and a sanity check for the
    evaluation pipeline. It does not provide rigorous uncertainty estimates.
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
            min_neighbors=int(cfg.get("min_neighbors", 3)),
            include_time=bool(cfg.get("include_time", True)),
            include_depth=bool(cfg.get("include_depth", True)),
            regularization=float(cfg.get("regularization", 1e-6)),
        )

        self._X: Optional[ArrayLike] = None
        self._y: Optional[ArrayLike] = None
        self._tree: Optional[cKDTree] = None
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
        X = observations.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )
        y = observations.values.astype(float)

        self._X = X
        self._y = y
        self._tree = cKDTree(X)
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
        if not self.is_fitted or self._X is None or self._tree is None:
            raise RuntimeError("Model must be fitted before prediction.")

        Xq = query_points.as_features(
            include_time=self._cfg.include_time,
            include_depth=self._cfg.include_depth,
        )

        k = min(self._cfg.k_neighbors, self._X.shape[0])
        if k < self._cfg.min_neighbors:
            raise RuntimeError(
                f"Not enough observations for local linear regression "
                f"(have {self._X.shape[0]}, need at least "
                f"{self._cfg.min_neighbors})."
            )

        dists, idxs = self._tree.query(Xq, k=k)

        # Ensure 2D arrays for unified handling when k == 1.
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        n_queries = Xq.shape[0]
        preds = np.empty(n_queries, dtype=float)

        # Simple distance-based weights; closer points get higher weights.
        # We avoid division by zero by adding a small epsilon.
        eps = 1e-12

        for i in range(n_queries):
            neigh_idx = idxs[i]
            neigh_dists = dists[i]

            mask = np.isfinite(neigh_dists)
            neigh_idx = neigh_idx[mask]
            neigh_dists = neigh_dists[mask]

            if neigh_idx.size < self._cfg.min_neighbors:
                # Fall back to simple average of available neighbors.
                preds[i] = float(self._y[neigh_idx].mean())
                continue

            X_local = self._X[neigh_idx]
            y_local = self._y[neigh_idx]

            # Weights: inverse distance, with clipping for numerical stability.
            w = 1.0 / (neigh_dists + eps)
            w /= w.sum()

            # Design matrix with bias term.
            Phi = np.concatenate(
                [np.ones((X_local.shape[0], 1)), X_local],
                axis=1,
            )

            # Weighted least squares: (Phi^T W Phi + λI)^{-1} Phi^T W y
            W = np.diag(w)
            A = Phi.T @ W @ Phi
            A += self._cfg.regularization * np.eye(A.shape[0])
            b = Phi.T @ W @ y_local

            coef = np.linalg.solve(A, b)

            xq = np.concatenate([[1.0], Xq[i]])
            preds[i] = float(xq @ coef)

        return FieldPrediction(mean=preds, std=None, metadata={})

    def reset(self) -> None:
        super().reset()
        self._X = None
        self._y = None
        self._tree = None
        self._variable = None

