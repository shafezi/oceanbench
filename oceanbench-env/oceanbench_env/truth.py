from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xarray as xr

from oceanbench_core.interpolation import BoundsMode, interpolate_dataset
from oceanbench_core.types import QueryPoints, Scenario


@dataclass
class OceanTruthField:
    """
    Thin wrapper around a provider-returned xarray.Dataset representing truth.

    This object is intentionally lightweight: it does not implement episode
    stepping or agent interaction. Its sole responsibility is to provide
    consistent ground-truth queries for evaluation and (later) simulators.
    """

    dataset: xr.Dataset
    variable: str
    scenario: Optional[Scenario] = None

    def query(
        self,
        query_points: QueryPoints,
        *,
        method: str = "linear",
        bounds_mode: BoundsMode = "nan",
    ) -> xr.DataArray:
        """
        Query ground-truth values at the given points.

        Parameters
        ----------
        query_points:
            Spatial (and optionally temporal/depth) locations.
        method:
            Interpolation method for xarray (e.g. "linear", "nearest").
        bounds_mode:
            How to handle queries outside the dataset domain; see
            :func:`oceanbench_core.interpolation.interpolate_dataset`.
        """

        return interpolate_dataset(
            self.dataset,
            query_points=query_points,
            variable=self.variable,
            method=method,
            bounds_mode=bounds_mode,
        )

    def query_array(
        self,
        query_points: QueryPoints,
        *,
        method: str = "linear",
        bounds_mode: BoundsMode = "nan",
    ) -> np.ndarray:
        """
        Same as :meth:`query`, but returns a NumPy array of values.

        This is convenient for feeding truth directly into metrics or
        model-comparison code that operates on arrays.
        """

        da = self.query(
            query_points=query_points,
            method=method,
            bounds_mode=bounds_mode,
        )
        return np.asarray(da.values, dtype=float)

