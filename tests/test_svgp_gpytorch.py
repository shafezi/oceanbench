"""Tests for GPyTorch SVGP baseline model."""

from __future__ import annotations

import numpy as np

from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field import SVGPGPyTorchFieldModel


def _toy_obs(n: int = 50) -> ObservationBatch:
    rng = np.random.default_rng(123)
    lats = rng.uniform(24.0, 36.0, n)
    lons = rng.uniform(-86.0, -74.0, n)
    vals = 0.3 * np.sin(lats) + 0.2 * np.cos(lons) + 0.05 * rng.standard_normal(n)
    return ObservationBatch(lats=lats, lons=lons, values=vals, variable="temp")


def test_svgp_fit_predict_toy():
    obs = _toy_obs(40)
    model = SVGPGPyTorchFieldModel(
        {
            "n_inducing": 16,
            "inducing_strategy": "random",
            "fit_iters": 8,
            "update_iters": 3,
            "include_time": False,
            "include_depth": False,
        },
        seed=0,
    )
    model.fit(obs)
    qp = QueryPoints(lats=obs.lats[:10], lons=obs.lons[:10])
    pred = model.predict(qp)
    assert pred.std is not None
    assert pred.mean.shape == (10,)
    assert np.all(np.isfinite(pred.mean))


def test_svgp_online_update_schedule_runs():
    obs = _toy_obs(60)
    first = ObservationBatch(
        lats=obs.lats[:30],
        lons=obs.lons[:30],
        values=obs.values[:30],
        variable=obs.variable,
    )
    second = ObservationBatch(
        lats=obs.lats[30:],
        lons=obs.lons[30:],
        values=obs.values[30:],
        variable=obs.variable,
    )
    model = SVGPGPyTorchFieldModel(
        {
            "n_inducing": 20,
            "inducing_strategy": "kmeans",
            "training_schedule": "per_replan",
            "fit_iters": 6,
            "update_iters": 2,
            "replan_iters": 4,
            "include_time": False,
            "include_depth": False,
        },
        seed=11,
    )
    model.fit(first)
    pred_before = model.predict(QueryPoints(lats=second.lats[:8], lons=second.lons[:8])).mean
    model.update(second)
    model.train_replan()
    pred_after = model.predict(QueryPoints(lats=second.lats[:8], lons=second.lons[:8])).mean
    assert np.any(np.abs(pred_after - pred_before) > 1e-10)


def test_svgp_noise_estimation_enabled():
    obs = _toy_obs(30)
    model = SVGPGPyTorchFieldModel(
        {
            "n_inducing": 12,
            "fit_iters": 6,
            "update_iters": 2,
            "include_time": False,
            "include_depth": False,
            "noise": {
                "mode": "estimate",
                "estimate_method": "gp_likelihood",
                "fixed_sigma2": 1e-3,
            },
        },
        seed=5,
    )
    model.fit(obs)
    qp = QueryPoints(lats=obs.lats[:5], lons=obs.lons[:5])
    _ = model.predict(qp)
    sigma2 = float(model._noise.resolve_sigma2())
    assert np.isfinite(sigma2)
    assert sigma2 > 0.0
