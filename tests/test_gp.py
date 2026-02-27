"""Tests for GPFieldModel."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import GPFieldModel


def test_fit_predict_uncertainty(small_observation_batch, small_query_points):
    model = GPFieldModel({"lengthscale": 1.0, "noise": 0.01}, seed=42)
    model.fit(small_observation_batch)
    pred = model.predict(small_query_points)
    assert pred.std is not None
    assert np.all(pred.std >= 0)
    assert np.all(np.isfinite(pred.mean))


def test_update_raises(small_observation_batch):
    model = GPFieldModel(seed=42)
    model.fit(small_observation_batch)
    with pytest.raises(NotImplementedError):
        model.update(small_observation_batch)
