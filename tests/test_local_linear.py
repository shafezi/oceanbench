"""Tests for LocalLinearFieldModel."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import LocalLinearFieldModel


def test_fit_predict(small_observation_batch, small_query_points):
    model = LocalLinearFieldModel({"k_neighbors": 10}, seed=42)
    model.fit(small_observation_batch)
    pred = model.predict(small_query_points)
    assert pred.std is None
    assert np.all(np.isfinite(pred.mean))


def test_update_raises(small_observation_batch):
    model = LocalLinearFieldModel(seed=42)
    model.fit(small_observation_batch)
    with pytest.raises(NotImplementedError):
        model.update(small_observation_batch)
