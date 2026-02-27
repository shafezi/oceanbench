"""Tests for GMRFFieldModel."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import GMRFFieldModel


def test_fit_predict(small_observation_batch, small_query_points):
    model = GMRFFieldModel({"n_lat": 15, "n_lon": 15}, seed=42)
    model.fit(small_observation_batch)
    pred = model.predict(small_query_points)
    assert pred.std is not None
    assert np.all(np.isfinite(pred.mean))
