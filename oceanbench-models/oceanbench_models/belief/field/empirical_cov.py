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

    Covariance between arbitrary sets of (lat, lon[, time]) locations is
    computed by evaluating truth at those exact locations for each historical
    snapshot, then computing the sample cross-covariance.  When a time
    column is present in the feature matrix, each query point is matched to
    the snapshot whose time is closest, preserving temporal correlation
    structure (Binney et al. 2013, Sec 7.3, Fig. 9).

    For the evaluation grid Y, snapshot values are precomputed at
    construction time and cached.  For arbitrary sample locations, truth is
    queried lazily via the stored OceanTruthField, with results cached per
    unique location set.
    """

    def __init__(
        self,
        eval_grid: EvalGrid,
        truth: OceanTruthField,
        snapshot_times: np.ndarray,
        eval_values: np.ndarray,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        eval_grid:
            Evaluation grid defining the target set Y (size M).
        truth:
            OceanTruthField providing interpolated truth at arbitrary locations.
        snapshot_times:
            Array of shape (K,) with the K snapshot times (datetime64[ns]).
        eval_values:
            Array of shape (K, M) with truth values on Y for each snapshot.
            If use_anomalies is True, these should be raw (un-centred) values;
            anomaly subtraction is applied internally.
        config:
            Empirical covariance configuration mapping.
        """
        self._eval_grid = eval_grid
        self._truth = truth
        self._snapshot_times = np.asarray(snapshot_times, dtype="datetime64[ns]")
        self._cfg = EmpiricalCovConfig(
            window_days=int((config or {}).get("window_days", 30)),
            stride=str((config or {}).get("stride", "1d")),
            max_snapshots=int((config or {}).get("max_snapshots", 50)),
            use_anomalies=bool((config or {}).get("use_anomalies", True)),
        )

        # Determine how many spatial dimensions the eval grid has.
        # Column order convention: [lat, lon, (depth), (time)].
        self._include_depth = eval_grid.query_points.depths is not None
        self._n_space_dims = 3 if self._include_depth else 2

        # Preprocess eval grid values: anomalies + NaN imputation.
        self._eval_values = self._preprocess(np.asarray(eval_values, dtype=float))

        # Cache for lazily evaluated snapshot values at arbitrary locations.
        # Key: bytes of spatial coordinate arrays → values array (K, N).
        self._query_cache: dict[Any, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_values(
        cls,
        eval_grid: EvalGrid,
        values: ArrayLike,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> "EmpiricalCovarianceBackend":
        """
        Build from pre-computed snapshot values (for testing or when
        truth has already been evaluated externally).

        The resulting backend can compute ``cov_block`` for feature
        matrices that correspond to the eval grid, but cannot lazily
        query truth at arbitrary locations (no OceanTruthField).
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            raise ValueError("values must have shape (n_snapshots, n_points).")
        K = values.shape[0]
        # Dummy snapshot times (one per row).
        snapshot_times = np.arange(K).astype("datetime64[D]").astype("datetime64[ns]")

        class _NullTruth:
            """Placeholder that raises if lazily queried."""
            def query_array(self, *a: Any, **kw: Any) -> np.ndarray:
                raise RuntimeError(
                    "EmpiricalCovarianceBackend built via from_values() "
                    "cannot query truth at arbitrary locations."
                )

        return cls(
            eval_grid=eval_grid,
            truth=_NullTruth(),  # type: ignore[arg-type]
            snapshot_times=snapshot_times,
            eval_values=values,
            config=config,
        )

    @classmethod
    def from_provider(
        cls,
        *,
        eval_grid: EvalGrid,
        provider: Any,
        config: Mapping[str, Any],
    ) -> "EmpiricalCovarianceBackend":
        """
        Build an empirical covariance backend using the DataProvider.

        Fetches a window of historical snapshots and evaluates truth on the
        evaluation grid.  The OceanTruthField is retained so that
        ``cov_block`` can lazily query truth at arbitrary sample locations.
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
            stride = np.timedelta64(1, "D")

        times: list[np.datetime64] = []
        t = window_start
        while t < start and len(times) < cfg.max_snapshots:
            times.append(t)
            t = t + stride

        if not times:
            raise RuntimeError("No historical snapshots available for empirical covariance.")

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

        # Evaluate truth on the eval grid for each snapshot.
        qp_base = eval_grid.query_points
        eval_values = []
        for t_snap in times:
            qpt = QueryPoints(
                lats=qp_base.lats,
                lons=qp_base.lons,
                times=np.full(qp_base.size, t_snap, dtype="datetime64[ns]"),
                depths=qp_base.depths,
                metadata=qp_base.metadata,
            )
            arr = truth.query_array(qpt, method="linear", bounds_mode="nan")
            eval_values.append(arr.reshape(-1))

        snapshot_times = np.array(times, dtype="datetime64[ns]")
        eval_values_arr = np.stack(eval_values, axis=0)  # shape (K, M)

        return cls(
            eval_grid=eval_grid,
            truth=truth,
            snapshot_times=snapshot_times,
            eval_values=eval_values_arr,
            config=config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, vals: np.ndarray) -> np.ndarray:
        """Apply anomaly subtraction and NaN imputation."""
        vals = vals.copy()
        if self._cfg.use_anomalies:
            mean_per_point = np.nanmean(vals, axis=0, keepdims=True)
            vals = vals - mean_per_point

        nan_mask = ~np.isfinite(vals)
        if np.any(nan_mask):
            col_means = np.nanmean(vals, axis=0)
            vals = np.where(nan_mask, np.broadcast_to(col_means, vals.shape), vals)
        return vals

    def _query_snapshots_at(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        depths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate truth at (lats, lons[, depths]) for every snapshot time.

        Returns array of shape (K, N) where K is the number of snapshots
        and N = len(lats).  Results are cached by coordinate arrays.
        """
        cache_parts = [lats.tobytes(), lons.tobytes()]
        if depths is not None:
            cache_parts.append(depths.tobytes())
        cache_key = tuple(cache_parts)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        n = lats.size
        rows = []
        for t_snap in self._snapshot_times:
            qpt = QueryPoints(
                lats=lats,
                lons=lons,
                times=np.full(n, t_snap, dtype="datetime64[ns]"),
                depths=depths,
            )
            arr = self._truth.query_array(qpt, method="linear", bounds_mode="nan")
            rows.append(arr.reshape(-1))

        vals = self._preprocess(np.stack(rows, axis=0))  # (K, N)
        self._query_cache[cache_key] = vals
        return vals

    def _snapshot_values_for_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get snapshot-based values for a feature matrix X.

        Column layout: ``[lat, lon, (depth), (time)]``.
        The number of leading spatial columns is ``self._n_space_dims``
        (2 for lat/lon, 3 when depth is included).

        If X has a time column (after spatial dims), each point is matched
        to its nearest snapshot.  Otherwise, all snapshots are used.
        """
        lats = X[:, 0].copy()
        lons = X[:, 1].copy()
        nsd = self._n_space_dims
        depths = X[:, 2].copy() if nsd >= 3 else None
        has_time = X.shape[1] > nsd

        # Query truth at the exact spatial locations for every snapshot.
        all_vals = self._query_snapshots_at(lats, lons, depths)  # (K, N)

        if not has_time:
            return all_vals

        # Time-aware: for each query point, find the nearest snapshot and
        # shift snapshot indices to preserve temporal correlation structure.
        time_seconds = X[:, nsd]  # seconds since epoch
        snap_seconds = self._snapshot_times.astype("int64").astype(float) / 1e9

        nearest_snap = np.argmin(
            np.abs(time_seconds[None, :] - snap_seconds[:, None]),
            axis=0,
        )

        K = all_vals.shape[0]
        N = all_vals.shape[1]
        k_indices = np.arange(K)[:, None]  # (K, 1)
        shifted = np.clip(k_indices + nearest_snap[None, :], 0, K - 1)  # (K, N)
        vals_shifted = all_vals[shifted, np.arange(N)[None, :]]
        return vals_shifted

    # ------------------------------------------------------------------
    # Core CovarianceBackend API
    # ------------------------------------------------------------------

    def _get_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get preprocessed snapshot values for feature matrix X.

        If X corresponds exactly to the eval grid, return the cached
        eval_values.  Otherwise, query truth at the exact locations.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        lats = X[:, 0]
        lons = X[:, 1]
        nsd = self._n_space_dims

        # Check if this is the eval grid (fast path).
        eval_lats = np.asarray(self._eval_grid.query_points.lats, dtype=float)
        eval_lons = np.asarray(self._eval_grid.query_points.lons, dtype=float)
        spatial_match = (
            lats.shape[0] == eval_lats.shape[0]
            and np.allclose(lats, eval_lats, atol=1e-8)
            and np.allclose(lons, eval_lons, atol=1e-8)
        )
        if spatial_match and nsd >= 3 and self._eval_grid.query_points.depths is not None:
            eval_depths = np.asarray(self._eval_grid.query_points.depths, dtype=float)
            spatial_match = spatial_match and np.allclose(X[:, 2], eval_depths, atol=1e-8)

        if spatial_match and X.shape[1] <= nsd:
            return self._eval_values

        return self._snapshot_values_for_features(X)

    def cov_block(self, Xa: ArrayLike, Xb: ArrayLike) -> ArrayLike:
        """
        Compute empirical cross-covariance Σ(Xa, Xb).

        Evaluates truth at the exact (lat, lon) locations of Xa and Xb
        for each historical snapshot, then computes the sample
        cross-covariance.
        """
        Xa = np.atleast_2d(np.asarray(Xa, dtype=float))
        Xb = np.atleast_2d(np.asarray(Xb, dtype=float))

        Va = self._get_values(Xa)  # (K, Na)
        Vb = self._get_values(Xb)  # (K, Nb)

        K = Va.shape[0]
        # Sample cross-covariance: (1/(K-1)) * (Va - mean_a)^T @ (Vb - mean_b)
        Va_centred = Va - np.nanmean(Va, axis=0, keepdims=True)
        Vb_centred = Vb - np.nanmean(Vb, axis=0, keepdims=True)

        # Handle any remaining NaNs by treating them as zero (neutral).
        Va_centred = np.where(np.isfinite(Va_centred), Va_centred, 0.0)
        Vb_centred = np.where(np.isfinite(Vb_centred), Vb_centred, 0.0)

        denom = max(K - 1, 1)
        return (Va_centred.T @ Vb_centred) / denom

    def diag_cov(self, X: ArrayLike) -> ArrayLike:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        V = self._get_values(X)  # (K, N)
        V_centred = V - np.nanmean(V, axis=0, keepdims=True)
        V_centred = np.where(np.isfinite(V_centred), V_centred, 0.0)
        K = V.shape[0]
        denom = max(K - 1, 1)
        return np.sum(V_centred ** 2, axis=0) / denom
