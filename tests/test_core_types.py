"""Tests for oceanbench_core types."""
import numpy as np
import pytest
from oceanbench_core.types import Observation, ObservationBatch, QueryPoints, Scenario


def test_observation_batch_from_observations():
    obs_list = [
        Observation(lat=25.0, lon=-80.0, value=20.1, variable="temp"),
        Observation(lat=26.0, lon=-79.0, value=20.2, variable="temp"),
    ]
    batch = ObservationBatch.from_observations(obs_list)
    assert batch.size == 2
    assert batch.variable == "temp"


def test_query_points_as_features():
    qp = QueryPoints(lats=np.array([25.0, 26.0]), lons=np.array([-80.0, -79.0]))
    feat = qp.as_features(include_time=False, include_depth=False)
    assert feat.shape == (2, 2)
