"""Tests for SparseOnlineGPFieldModel."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import SparseOnlineGPFieldModel


def test_fit_predict(small_observation_batch, small_query_points):
    model = SparseOnlineGPFieldModel({"max_points": 80}, seed=42)
    model.fit(small_observation_batch)
    pred = model.predict(small_query_points)
    assert pred.std is not None
    assert np.all(np.isfinite(pred.mean))


def test_online_update(small_observation_batch, small_query_points):
    model = SparseOnlineGPFieldModel({"max_points": 100}, seed=42)
    n = small_observation_batch.size
    half = n // 2
    first = ObservationBatch(
        lats=small_observation_batch.lats[:half],
        lons=small_observation_batch.lons[:half],
        values=small_observation_batch.values[:half],
        variable=small_observation_batch.variable,
    )
    model.fit(first)
    rest = ObservationBatch(
        lats=small_observation_batch.lats[half:],
        lons=small_observation_batch.lons[half:],
        values=small_observation_batch.values[half:],
        variable=small_observation_batch.variable,
    )
    model.update(rest)
    pred = model.predict(small_query_points)
    assert np.all(np.isfinite(pred.mean))
