"""Tests for the common FieldBeliefModel interface across all implementations."""
import numpy as np
import pytest

from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import (
    FieldBeliefModel,
    FieldPrediction,
    LocalLinearFieldModel,
    GPFieldModel,
    SparseOnlineGPFieldModel,
    PseudoInputGPFieldModel,
    STGPFieldModel,
    GMRFFieldModel,
)


MODELS = [
    ("local_linear", LocalLinearFieldModel({"k_neighbors": 10}, seed=42)),
    ("gp", GPFieldModel({"lengthscale": 1.0, "noise": 0.01}, seed=42)),
    ("sparse_online_gp", SparseOnlineGPFieldModel({"max_points": 80}, seed=42)),
    ("pseudo_input_gp", PseudoInputGPFieldModel({"n_pseudo": 30}, seed=42)),
    ("gmrf", GMRFFieldModel({"n_lat": 15, "n_lon": 15}, seed=42)),
]


@pytest.mark.parametrize("name,model", MODELS)
def test_interface_fit_predict(name, model, small_observation_batch, small_query_points):
    assert isinstance(model, FieldBeliefModel)
    assert not model.is_fitted
    model.fit(small_observation_batch)
    assert model.is_fitted
    pred = model.predict(small_query_points)
    assert isinstance(pred, FieldPrediction)
    assert pred.mean.shape == (small_query_points.size,)
    if model.supports_uncertainty and pred.std is not None:
        assert pred.std.shape == (small_query_points.size,)


@pytest.mark.parametrize("name,model", MODELS)
def test_reset(name, model, small_observation_batch, small_query_points):
    model.fit(small_observation_batch)
    model.reset()
    assert not model.is_fitted
    with pytest.raises(RuntimeError):
        model.predict(small_query_points)


@pytest.mark.parametrize("name,model", MODELS)
def test_predict_mean_std(name, model, small_observation_batch, small_query_points):
    model.fit(small_observation_batch)
    mean = model.predict_mean(small_query_points)
    std = model.predict_std(small_query_points)
    assert mean.shape == (small_query_points.size,)
    if model.supports_uncertainty:
        assert std is not None and std.shape == (small_query_points.size,)
    else:
        assert std is None
