from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class Scenario:
    """
    Minimal description of a data / evaluation scenario.

    This is intentionally lightweight and backend-agnostic. It is meant to be
    enough for logging, configuration, and basic subsetting – not a full
    experiment-spec language.
    """

    name: Optional[str] = None
    variable: str = "variable"

    # Geographic region as an axis-aligned bounding box in degrees.
    # Example: {"lat_min": -10.0, "lat_max": 10.0, "lon_min": 140.0, "lon_max": 160.0}
    region: Mapping[str, float] = field(default_factory=dict)

    # Optional time and depth windows.
    time_range: Optional[tuple[np.datetime64, np.datetime64]] = None
    depth_range: Optional[tuple[float, float]] = None

    # Free-form metadata for identifiers, provider config hashes, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """
    A single scalar observation of a field variable at a point in space(-time).

    Coordinates follow the canonical convention:
    - latitude in degrees north
    - longitude in degrees east
    - optional time as numpy.datetime64
    - optional depth in meters (positive down)
    """

    lat: float
    lon: float
    value: float
    variable: str

    time: Optional[np.datetime64] = None
    depth: Optional[float] = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationBatch:
    """
    Batched observations used for fitting and updating field models.

    All arrays are 1-D and must have the same leading dimension N.
    """

    lats: ArrayLike
    lons: ArrayLike
    values: ArrayLike
    variable: str

    times: Optional[ArrayLike] = None
    depths: Optional[ArrayLike] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)
        self.values = np.asarray(self.values, dtype=float)

        n = self.lats.shape[0]
        if self.lons.shape[0] != n or self.values.shape[0] != n:
            raise ValueError("lats, lons, and values must have the same length")

        if self.times is not None:
            self.times = np.asarray(self.times)
            if self.times.shape[0] != n:
                raise ValueError("times must have the same length as lats")

        if self.depths is not None:
            self.depths = np.asarray(self.depths, dtype=float)
            if self.depths.shape[0] != n:
                raise ValueError("depths must have the same length as lats")

    @property
    def size(self) -> int:
        """Number of observations."""

        return int(self.lats.shape[0])

    def as_features(
        self,
        *,
        include_time: bool = True,
        include_depth: bool = True,
    ) -> ArrayLike:
        """
        Return coordinates stacked into a feature matrix of shape (N, D).

        Column order is always [lat, lon, (time), (depth)] where the optional
        columns are included only when requested and available.
        """

        coords: list[ArrayLike] = [self.lats, self.lons]

        if include_time and self.times is not None:
            # Represent time as float in seconds since epoch for models that
            # expect numeric features.
            if np.issubdtype(self.times.dtype, np.datetime64):
                t = self.times.astype("datetime64[ns]").astype("int64") / 1e9
                coords.append(t.astype(float))
            else:
                coords.append(self.times.astype(float))

        if include_depth and self.depths is not None:
            coords.append(self.depths.astype(float))

        return np.stack(coords, axis=-1)

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation],
        *,
        variable: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ObservationBatch":
        """Construct a batch from a sequence of `Observation` objects."""

        if not observations:
            raise ValueError("observations must be non-empty")

        vars_in_obs = {obs.variable for obs in observations}
        if variable is None:
            if len(vars_in_obs) != 1:
                raise ValueError(
                    "variable is ambiguous; pass `variable` explicitly "
                    f"(found variables: {sorted(vars_in_obs)})"
                )
            variable = next(iter(vars_in_obs))

        lats = np.array([obs.lat for obs in observations], dtype=float)
        lons = np.array([obs.lon for obs in observations], dtype=float)
        values = np.array([obs.value for obs in observations], dtype=float)

        times_list: list[np.datetime64] = []
        depths_list: list[float] = []
        has_time = any(obs.time is not None for obs in observations)
        has_depth = any(obs.depth is not None for obs in observations)

        times_array: Optional[ArrayLike]
        depths_array: Optional[ArrayLike]

        if has_time:
            # Missing times become NaT.
            times_list = [
                obs.time if obs.time is not None else np.datetime64("NaT")
                for obs in observations
            ]
            times_array = np.array(times_list, dtype="datetime64[ns]")
        else:
            times_array = None

        if has_depth:
            # Missing depths become NaN.
            depths_list = [
                obs.depth if obs.depth is not None else np.nan
                for obs in observations
            ]
            depths_array = np.array(depths_list, dtype=float)
        else:
            depths_array = None

        return cls(
            lats=lats,
            lons=lons,
            values=values,
            variable=variable,
            times=times_array,
            depths=depths_array,
            metadata=dict(metadata or {}),
        )


@dataclass
class QueryPoints:
    """
    A batch of query points where field models or truth will be evaluated.

    Similar to `ObservationBatch`, but without observed values.
    """

    lats: ArrayLike
    lons: ArrayLike

    times: Optional[ArrayLike] = None
    depths: Optional[ArrayLike] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lats = np.asarray(self.lats, dtype=float)
        self.lons = np.asarray(self.lons, dtype=float)

        n = self.lats.shape[0]
        if self.lons.shape[0] != n:
            raise ValueError("lats and lons must have the same length")

        if self.times is not None:
            self.times = np.asarray(self.times)
            if self.times.shape[0] != n:
                raise ValueError("times must have the same length as lats")

        if self.depths is not None:
            self.depths = np.asarray(self.depths, dtype=float)
            if self.depths.shape[0] != n:
                raise ValueError("depths must have the same length as lats")

    @property
    def size(self) -> int:
        """Number of query points."""

        return int(self.lats.shape[0])

    def as_features(
        self,
        *,
        include_time: bool = True,
        include_depth: bool = True,
    ) -> ArrayLike:
        """
        Return coordinates stacked into a feature matrix of shape (N, D).

        Column order is always [lat, lon, (time), (depth)] where the optional
        columns are included only when requested and available.
        """

        coords: list[ArrayLike] = [self.lats, self.lons]

        if include_time and self.times is not None:
            if np.issubdtype(self.times.dtype, np.datetime64):
                t = self.times.astype("datetime64[ns]").astype("int64") / 1e9
                coords.append(t.astype(float))
            else:
                coords.append(self.times.astype(float))

        if include_depth and self.depths is not None:
            coords.append(self.depths.astype(float))

        return np.stack(coords, axis=-1)

    @classmethod
    def from_iterable(
        cls,
        points: Iterable[Mapping[str, Any]],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "QueryPoints":
        """
        Construct query points from an iterable of mapping-like objects.

        Each mapping should at least contain 'lat' and 'lon', and may contain
        'time' and 'depth'.
        """

        lats: list[float] = []
        lons: list[float] = []
        times: list[Any] = []
        depths: list[float] = []

        has_time = False
        has_depth = False

        for p in points:
            lats.append(float(p["lat"]))
            lons.append(float(p["lon"]))

            if "time" in p and p["time"] is not None:
                has_time = True
                times.append(p["time"])
            else:
                times.append(None)

            if "depth" in p and p["depth"] is not None:
                has_depth = True
                depths.append(float(p["depth"]))
            else:
                depths.append(np.nan)

        if not lats:
            raise ValueError("points must be non-empty")

        times_array: Optional[ArrayLike]
        depths_array: Optional[ArrayLike]

        if has_time:
            times_array = np.array(
                [
                    t if t is not None else np.datetime64("NaT")
                    for t in times
                ],
                dtype="datetime64[ns]",
            )
        else:
            times_array = None

        if has_depth:
            depths_array = np.array(depths, dtype=float)
        else:
            depths_array = None

        return cls(
            lats=np.array(lats, dtype=float),
            lons=np.array(lons, dtype=float),
            times=times_array,
            depths=depths_array,
            metadata=dict(metadata or {}),
        )

