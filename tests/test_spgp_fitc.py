"""Tests for the SPGP (FITC) field model."""

import numpy as np
import pytest

from oceanbench_core.types import ObservationBatch, QueryPoints
from oceanbench_models.belief.field.spgp_fitc import SPGPFITCFieldModel, _fitc_nll


SEED = 42


def _make_synthetic_data(n: int = 80, seed: int = SEED):
    """Smooth synthetic field + noise for testing."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(24, 28, n)
    lons = rng.uniform(-86, -82, n)
    values = 20.0 + 2.0 * np.sin(0.5 * lats) + 1.5 * np.cos(0.3 * lons)
    values += 0.1 * rng.standard_normal(n)
    return ObservationBatch(lats=lats, lons=lons, values=values, variable="temp")


def _make_query_points(n: int = 30, seed: int = SEED + 1):
    rng = np.random.default_rng(seed)
    return QueryPoints(
        lats=rng.uniform(24, 28, n),
        lons=rng.uniform(-86, -82, n),
    )


class TestSPGPFITCBasic:
    def test_fit_predict_shapes(self):
        """Fit and predict should return correct shapes."""
        obs = _make_synthetic_data()
        qp = _make_query_points()
        model = SPGPFITCFieldModel(
            {"n_pseudo": 20, "max_opt_iters": 30}, seed=SEED,
        )
        model.fit(obs)
        pred = model.predict(qp)

        assert pred.mean.shape == (qp.size,)
        assert pred.std.shape == (qp.size,)

    def test_mean_is_finite(self):
        obs = _make_synthetic_data()
        qp = _make_query_points()
        model = SPGPFITCFieldModel({"n_pseudo": 20, "max_opt_iters": 30}, seed=SEED)
        model.fit(obs)
        pred = model.predict(qp)

        assert np.all(np.isfinite(pred.mean))

    def test_std_is_nonnegative_and_finite(self):
        obs = _make_synthetic_data()
        qp = _make_query_points()
        model = SPGPFITCFieldModel({"n_pseudo": 20, "max_opt_iters": 30}, seed=SEED)
        model.fit(obs)
        pred = model.predict(qp)

        assert np.all(pred.std >= 0)
        assert np.all(np.isfinite(pred.std))

    def test_supports_uncertainty(self):
        model = SPGPFITCFieldModel({"n_pseudo": 10}, seed=SEED)
        assert model.supports_uncertainty
        assert not model.supports_online_update


class TestSPGPFITCOptimisation:
    def test_nll_improves_during_training(self):
        """NLL at optimised params should be <= NLL at initial params."""
        obs = _make_synthetic_data(n=60)
        model = SPGPFITCFieldModel(
            {"n_pseudo": 15, "max_opt_iters": 50}, seed=SEED,
        )

        # Compute initial NLL before fitting.
        from oceanbench_models.belief.field.utils import observation_batch_to_numpy
        from oceanbench_models.belief.field.utils import FeatureScaler
        X, y = observation_batch_to_numpy(obs, include_time=False, include_depth=False)
        scaler = FeatureScaler.from_data(X)
        X = scaler.transform(X)

        N, D = X.shape
        M = 15
        rng = np.random.default_rng(SEED)
        idx = rng.choice(N, size=M, replace=False)
        Z_init = X[idx]

        initial_params = np.concatenate([
            [0.0], np.zeros(D), [-2.0], Z_init.ravel(),
        ])
        nll_before = _fitc_nll(initial_params, X, y, M, D)

        # Now fit the model.
        model.fit(obs)

        optimised_params = np.concatenate([
            [model._log_alpha], model._log_ls, [model._log_sigma2],
            model._Z.ravel(),
        ])
        nll_after = _fitc_nll(optimised_params, X, y, M, D)

        assert nll_after <= nll_before + 1.0  # Allow small tolerance


class TestSPGPFITCReproducibility:
    def test_deterministic_with_same_seed(self):
        """Two fits with the same seed should give identical predictions."""
        obs = _make_synthetic_data()
        qp = _make_query_points()

        model1 = SPGPFITCFieldModel({"n_pseudo": 15, "max_opt_iters": 30}, seed=SEED)
        model1.fit(obs)
        pred1 = model1.predict(qp)

        model2 = SPGPFITCFieldModel({"n_pseudo": 15, "max_opt_iters": 30}, seed=SEED)
        model2.fit(obs)
        pred2 = model2.predict(qp)

        np.testing.assert_allclose(pred1.mean, pred2.mean, atol=1e-6)
        np.testing.assert_allclose(pred1.std, pred2.std, atol=1e-6)


class TestSPGPFITCReset:
    def test_reset_clears_state(self):
        obs = _make_synthetic_data()
        model = SPGPFITCFieldModel({"n_pseudo": 10, "max_opt_iters": 20}, seed=SEED)
        model.fit(obs)
        assert model.is_fitted
        model.reset()
        assert not model.is_fitted
        assert model._Z is None
