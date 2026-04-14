"""Tests for paper-faithful SOGP model."""

from __future__ import annotations

import numpy as np

from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import SOGPPaperFieldModel


def _toy_obs(n: int = 60) -> ObservationBatch:
    rng = np.random.default_rng(0)
    lats = rng.uniform(24.0, 36.0, n)
    lons = rng.uniform(-86.0, -74.0, n)
    vals = np.sin(0.2 * lats) + np.cos(0.15 * lons) + 0.01 * rng.standard_normal(n)
    return ObservationBatch(lats=lats, lons=lons, values=vals, variable="temp")


def test_sogp_novelty_threshold_behavior():
    obs = _toy_obs(30)
    # Very high novelty threshold: after first basis insertion, most points should
    # be treated as non-novel and not increase BV-set.
    m_high = SOGPPaperFieldModel(
        {
            "max_basis_size": 50,
            "novelty_threshold": 1e9,
            "include_time": False,
            "include_depth": False,
        },
        seed=1,
    )
    m_high.fit(obs)
    assert m_high._alpha.shape[0] <= 2

    # Low threshold: BV-set should grow until capped.
    m_low = SOGPPaperFieldModel(
        {
            "max_basis_size": 8,
            "novelty_threshold": 0.0,
            "include_time": False,
            "include_depth": False,
        },
        seed=1,
    )
    m_low.fit(obs)
    assert m_low._alpha.shape[0] <= 8
    assert m_low._alpha.shape[0] >= 4


def test_sogp_update_changes_predictions():
    obs = _toy_obs(40)
    qp = QueryPoints(lats=obs.lats[:10], lons=obs.lons[:10])
    first = ObservationBatch(
        lats=obs.lats[:20],
        lons=obs.lons[:20],
        values=obs.values[:20],
        variable=obs.variable,
    )
    second = ObservationBatch(
        lats=obs.lats[20:],
        lons=obs.lons[20:],
        values=obs.values[20:],
        variable=obs.variable,
    )
    model = SOGPPaperFieldModel(
        {
            "max_basis_size": 20,
            "novelty_threshold": 1e-6,
            "include_time": False,
            "include_depth": False,
        },
        seed=3,
    )
    model.fit(first)
    pred_before = model.predict(qp).mean
    model.update(second)
    pred_after = model.predict(qp).mean
    assert np.any(np.abs(pred_after - pred_before) > 1e-8)


def test_sogp_uncertainty_non_negative():
    obs = _toy_obs(35)
    model = SOGPPaperFieldModel(
        {
            "max_basis_size": 15,
            "novelty_threshold": 1e-6,
            "include_time": False,
            "include_depth": False,
        },
        seed=7,
    )
    model.fit(obs)
    pred = model.predict(QueryPoints(lats=obs.lats[:12], lons=obs.lons[:12]))
    assert pred.std is not None
    assert np.all(np.isfinite(pred.std))
    assert np.all(pred.std >= 0.0)
