from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from oceanbench_core.eval_grid import EvalGrid
from oceanbench_core.types import QueryPoints, Scenario
from oceanbench_env import OceanTruthField

from .covariance_backends import ArrayLike, CovarianceBackend


@dataclass
class EmpiricalCovConfig:
    """
    Configuration for the empirical covariance backend.

    Parameters
    ----------
    window_days:
        Number of days prior to the scenario start to use for snapshots.
    stride:
        Temporal stride between snapshots. Expressed as a NumPy-compatible
        timedelta string (e.g., ``"6h"``, ``"1d"``).
    max_snapshots:
        Maximum number of snapshots to use.
    use_anomalies:
        If True, subtract the temporal mean at each location before computing
        covariances.
    """

    window_days: int = 30
    stride: str = "1d"
    max_snapshots: int = 50
    use_anomalies: bool = True


class EmpiricalCovarianceBackend(CovarianceBackend):
    """
    Empirical covariance backend based on historical truth snapshots.

    This implementation focuses on providing a covariance model over the dense
    evaluation grid Y. Cross-covariances with arbitrary sample locations are
    approximated by snapping those locations to their nearest evaluation-grid
    neighbours in (lat, lon) space. This choice is explicit and may be
    revisited in future refinements.
    """

    def __init__(
        self,
        eval_grid: EvalGrid,
        values: ArrayLike,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        eval_grid:
            Evaluation grid defining the target set Y (size M).
        values:
            Array of shape (K, M) with K snapshots of truth values on Y.
        config:
            Empirical covariance configuration mapping.
        """
        if values.ndim != 2:
            raise ValueError("values must have shape (n_snapshots, n_points).")

        self._eval_grid = eval_grid
        self._cfg = EmpiricalCovConfig(
            window_days=int((config or {}).get("window_days", 30)),
            stride=str((config or {}).get("stride", "1d")),
            max_snapshots=int((config or {}).get("max_snapshots", 50)),
            use_anomalies=bool((config or {}).get("use_anomalies", True)),
        )

        # Handle NaNs robustly by simple imputation with the per-location mean.
        vals = np.asarray(values, dtype=float)
        if self._cfg.use_anomalies:
            mean_per_point = np.nanmean(vals, axis=0, keepdims=True)
            vals = vals - mean_per_point

        nan_mask = ~np.isfinite(vals)
        if np.any(nan_mask):
            col_means = np.nanmean(vals, axis=0)
            vals = np.where(nan_mask, np.broadcast_to(col_means, vals.shape), vals)

        self._values = vals  # shape (K, M)
        # Sample covariance Σ_YY over evaluation grid Y.
        self._cov_yy = np.cov(self._values, rowvar=False, bias=False)

        # Precompute evaluation grid coordinates for nearest-neighbour snapping.
        qp = eval_grid.query_points
        self._eval_lats = np.asarray(qp.lats, dtype=float)
        self._eval_lons = np.asarray(qp.lons, dtype=float)

    # ------------------------------------------------------------------
    # Construction from provider
    # ------------------------------------------------------------------

    @classmethod
    def from_provider(
        cls,
        *,
        eval_grid: EvalGrid,
        provider: Any,
        config: Mapping[str, Any],
    ) -> "EmpiricalCovarianceBackend":
        """
        Build an empirical covariance backend using the DataProvider and scenario.

        This method:
          - selects a window of historical times before the scenario start,
          - sub-samples times according to the configured stride and cap,
          - evaluates truth on the evaluation grid for each snapshot,
          - constructs an EmpiricalCovarianceBackend from the resulting matrix.
        """
        if eval_grid.scenario is None:
            raise ValueError("EvalGrid.scenario must be set for empirical covariance.")

        scenario: Scenario = eval_grid.scenario
        if scenario.time_range is None:
            raise ValueError("Scenario.time_range must be set for empirical covariance.")

        cfg = EmpiricalCovConfig(
            window_days=int(config.get("window_days", 30)),
            stride=str(config.get("stride", "1d")),
            max_snapshots=int(config.get("max_snapshots", 50)),
            use_anomalies=bool(config.get("use_anomalies", True)),
        )

        start, _ = scenario.time_range
        start = np.datetime64(start, "ns")
        window_start = start - np.timedelta64(cfg.window_days, "D")

        stride = np.timedelta64(1, "D")
        try:
            stride = np.timedelta64(cfg.stride)
        except Exception:
            # Fall back to daily stride if parsing fails.
            stride = np.timedelta64(1, "D")

        times: list[np.datetime64] = []
        t = window_start
        while t < start and len(times) < cfg.max_snapshots:
            times.append(t)
            t = t + stride

        if not times:
            raise RuntimeError("No historical snapshots available for empirical covariance.")

        # Use the provider to fetch a dataset covering the historical window.
        # We assume the provider follows the DataProvider API.
        region = {
            "lon": [
                float(scenario.region["lon_min"]),
                float(scenario.region["lon_max"]),
            ],
            "lat": [
                float(scenario.region["lat_min"]),
                float(scenario.region["lat_max"]),
            ],
        }
        time_range = (str(window_start.astype("datetime64[s]")), str(start.astype("datetime64[s]")))
        variable = scenario.variable

        ds: xr.Dataset = provider.subset(
            product_id=scenario.metadata.get("product_id", ""),
            region=region,
            time=time_range,
            variables=[variable],
            depth_opts=None,
            target_grid=None,
            overwrite=False,
        )
        truth = OceanTruthField(dataset=ds, variable=variable, scenario=scenario)

        qp_base = eval_grid.query_points
        values = []
        for t_snap in times:
            qpt = QueryPoints(
                lats=qp_base.lats,
                lons=qp_base.lons,
                times=np.full(qp_base.size, t_snap, dtype="datetime64[ns]"),
                depths=qp_base.depths,
                metadata=qp_base.metadata,
            )
            arr = truth.query_array(qpt, method="linear", bounds_mode="nan")
            values.append(arr.reshape(-1))

        values_arr = np.stack(values, axis=0)  # shape (K, M)
        return cls(eval_grid=eval_grid, values=values_arr, config=config)

    # ------------------------------------------------------------------
    # Core CovarianceBackend API
    # ------------------------------------------------------------------

    def cov_block(self, Xa: ArrayLike, Xb: ArrayLike) -> ArrayLike:
        """
        Approximate covariance block Σ(S, T) by snapping S and T to the nearest
        evaluation-grid locations.

        Xa, Xb are feature matrices whose first two dimensions are (lat, lon).
        """
        Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))

        idx_a = self._nearest_eval_indices(Xa[:, 0], Xa[:, 1])
        idx_b = self._nearest_eval_indices(Xb[:, 0], Xb[:, 1])
        return self._cov_yy[np.ix_(idx_a, idx_b)]

    def diag_cov(self, X: ArrayLike) -> ArrayLike:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        idx = self._nearest_eval_indices(X[:, 0], X[:, 1])
        return np.asarray(np.diag(self._cov_yy)[idx], dtype=float)

    # ------------------------------------------------------------------
    # Nearest-neighbour helper
    # ------------------------------------------------------------------

    def _nearest_eval_indices(
        self,
        lats: ArrayLike,
        lons: ArrayLike,
    ) -> np.ndarray:
        lats = np.asarray(lats, dtype=float).reshape(-1)
        lons = np.asarray(lons, dtype=float).reshape(-1)

        # Simple Euclidean nearest neighbour search in (lat, lon) space.
        # For moderate grid sizes this is adequate; can be replaced by a
        # dedicated spatial index if needed later.
        lat_grid = self._eval_lats[None, :]
        lon_grid = self._eval_lons[None, :]

        lat_q = lats[:, None]
        lon_q = lons[:, None]

        d2 = (lat_q - lat_grid) ** 2 + (lon_q - lon_grid) ** 2
        idx = np.argmin(d2, axis=1)
        return idx.astype(int)

