"""Tests for STGPFieldModel (requires time in observations)."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import STGPFieldModel


@pytest.fixture
def obs_batch_with_time(seed):
    from oceanbench_core.types import ObservationBatch
    rng = np.random.default_rng(seed)
    n = 40
    # NumPy requires explicit datetime64 start/stop; build via timedelta offsets.
    times = np.datetime64("2014-01-01") + np.arange(n).astype("timedelta64[D]")
    return ObservationBatch(
        lats=rng.uniform(24, 36, n),
        lons=rng.uniform(-86, -74, n),
        values=20.0 + 0.1 * rng.standard_normal(n),
        variable="temp",
        times=times,
    )


@pytest.fixture
def query_points_with_time(seed):
    from oceanbench_core.types import QueryPoints
    rng = np.random.default_rng(seed + 1)
    n = 20
    times = np.datetime64("2014-01-01") + np.arange(n).astype("timedelta64[D]")
    return QueryPoints(
        lats=rng.uniform(24, 36, n),
        lons=rng.uniform(-86, -74, n),
        times=times,
    )


def test_fit_predict(obs_batch_with_time, query_points_with_time):
    model = STGPFieldModel(seed=42)
    model.fit(obs_batch_with_time)
    pred = model.predict(query_points_with_time)
    assert pred.std is not None
    assert np.all(np.isfinite(pred.mean))
