"""
End-to-end comparison of field models on a single scenario.

Uses either:
  - Synthetic data (default): small random field so the script runs without
    the Data Provider or real data.
  - Real data: set USE_PROVIDER = True and ensure oceanbench-data-provider
    is installed and configured; then a subset is loaded and used.

Outputs: printed metrics, optional saved result JSON, and optional figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add repo root so we can import from sibling packages when run from examples/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

USE_PROVIDER = False  # Set True to use real Data Provider subset
SEED = 42
N_TRAIN = 80
N_EVAL = 200
VARIABLE = "temp"
SAVE_RESULTS_PATH = REPO_ROOT / "results" / "compare_field_models.json"
SAVE_FIG_PATH = REPO_ROOT / "results" / "compare_field_models.png"


def _synthetic_dataset(
    n_train: int,
    n_eval: int,
    seed: int,
    variable: str = "temp",
):
    """Build train/eval ObservationBatch and QueryPoints plus ground truth from a tiny GP."""
    rng = np.random.default_rng(seed)
    lat_min, lat_max = 24.0, 36.0
    lon_min, lon_max = -86.0, -74.0

    lat_train = rng.uniform(lat_min, lat_max, n_train)
    lon_train = rng.uniform(lon_min, lon_max, n_train)
    # Simple smooth field: mean + decay from a few hotspots
    hotspots = np.array([[30.0, -80.0], [28.0, -76.0]])
    def field(lat, lon):
        v = 20.0
        for h in hotspots:
            v += 2.0 * np.exp(-0.01 * ((lat - h[0]) ** 2 + (lon - h[1]) ** 2))
        return v
    values_train = field(lat_train, lon_train) + 0.1 * rng.standard_normal(n_train)

    from oceanbench_core.types import ObservationBatch, QueryPoints

    train_batch = ObservationBatch(
        lats=lat_train,
        lons=lon_train,
        values=values_train,
        variable=variable,
    )
    lat_eval = rng.uniform(lat_min, lat_max, n_eval)
    lon_eval = rng.uniform(lon_min, lon_max, n_eval)
    query_points = QueryPoints(lats=lat_eval, lons=lon_eval)
    y_true = field(lat_eval, lon_eval)
    return train_batch, query_points, y_true, lat_eval, lon_eval, lat_train, lon_train


def main() -> None:
    if USE_PROVIDER:
        try:
            from oceanbench_data_provider import DataProvider
            from oceanbench_env import OceanTruthField
            from oceanbench_core.types import ObservationBatch, QueryPoints
        except ImportError as e:
            print("USE_PROVIDER is True but required packages are missing:", e)
            print("Install oceanbench-data-provider and oceanbench-env, or set USE_PROVIDER=False.")
            sys.exit(1)
        provider = DataProvider()
        ds = provider.subset(
            product_id="hycom_glbv0.08_reanalysis_53x",
            region={"lon": [-86, -74], "lat": [24, 36]},
            time=("2014-01-01", "2014-01-07"),
            variables=[VARIABLE],
        )
        truth = OceanTruthField(dataset=ds, variable=VARIABLE)
        # Build train/eval from grid or random points on the dataset
        lats = np.linspace(24.5, 35.5, 10)
        lons = np.linspace(-85, -75, 10)
        llat, llon = np.meshgrid(lats, lons)
        llat = llat.ravel()
        llon = llon.ravel()
        qp_all = QueryPoints(lats=llat, lons=llon)
        y_all = truth.query_array(qp_all, bounds_mode="clip")
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(llat))
        n_train = N_TRAIN
        train_idx = idx[:n_train]
        eval_idx = idx[n_train:]
        train_batch = ObservationBatch(
            lats=llat[train_idx],
            lons=llon[train_idx],
            values=y_all[train_idx],
            variable=VARIABLE,
        )
        query_points = QueryPoints(lats=llat[eval_idx], lons=llon[eval_idx])
        y_true = y_all[eval_idx]
        lat_eval = llat[eval_idx]
        lon_eval = llon[eval_idx]
        lat_train = llat[train_idx]
        lon_train = llon[train_idx]
    else:
        train_batch, query_points, y_true, lat_eval, lon_eval, lat_train, lon_train = _synthetic_dataset(
            N_TRAIN, N_EVAL, SEED, VARIABLE
        )

    from oceanbench_models.belief.field import (
        LocalLinearFieldModel,
        GPFieldModel,
        SparseOnlineGPFieldModel,
        PseudoInputGPFieldModel,
        GMRFFieldModel,
    )
    from oceanbench_bench import run_field_model_comparison, save_results
    from oceanbench_bench.results import EvaluationResult

    models = [
        ("local_linear", LocalLinearFieldModel({"k_neighbors": 15}, seed=SEED)),
        ("gp", GPFieldModel({"lengthscale": 1.0, "variance": 1.0, "noise": 0.01}, seed=SEED)),
        ("sparse_online_gp", SparseOnlineGPFieldModel({"max_points": 100}, seed=SEED)),
        ("pseudo_input_gp", PseudoInputGPFieldModel({"n_pseudo": 50}, seed=SEED)),
        ("gmrf", GMRFFieldModel({"n_lat": 25, "n_lon": 25}, seed=SEED)),
    ]

    result = run_field_model_comparison(
        train_batch,
        query_points,
        y_true,
        models,
        seed=SEED,
        scenario_name="synthetic" if not USE_PROVIDER else "hycom_subset",
        variable=VARIABLE,
    )

    print("Evaluation result:")
    for run in result.runs:
        print(f"  {run.model_name}: RMSE={run.metrics.get('rmse', float('nan')):.4f}, MAE={run.metrics.get('mae', float('nan')):.4f}, fit_time={run.fit_time_seconds:.3f}s, predict_time={run.predict_time_seconds:.3f}s")

    if SAVE_RESULTS_PATH:
        SAVE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_results(result, SAVE_RESULTS_PATH)
        print(f"Results saved to {SAVE_RESULTS_PATH}")

    if SAVE_FIG_PATH and not USE_PROVIDER:
        try:
            from oceanbench_viz import plot_truth_prediction_uncertainty
            # Pick one model for visualization (e.g. GP)
            gp = next(m for n, m in models if n == "gp")
            gp.reset()
            gp.seed(SEED)
            gp.fit(train_batch)
            pred = gp.predict(query_points)
            plot_truth_prediction_uncertainty(
                lat_eval,
                lon_eval,
                y_true,
                pred.mean,
                uncertainty=pred.std,
                obs_lats=lat_train,
                obs_lons=lon_train,
                title_prefix="GP",
                save_path=SAVE_FIG_PATH,
                show=False,
            )
            print(f"Figure saved to {SAVE_FIG_PATH}")
        except Exception as e:
            print("Could not save figure:", e)


if __name__ == "__main__":
    main()
