"""Tests for PseudoInputGPFieldModel."""
import numpy as np
import pytest
from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import PseudoInputGPFieldModel


def test_fit_predict(small_observation_batch, small_query_points):
    model = PseudoInputGPFieldModel({"n_pseudo": 30}, seed=42)
    model.fit(small_observation_batch)
    pred = model.predict(small_query_points)
    assert pred.std is not None
    assert np.all(np.isfinite(pred.mean))
