# OceanBench architecture

OceanBench is organized around three conceptual layers:

1. **Data Provider** – Source of ocean truth data (HYCOM, Copernicus, etc.). Returns sanitized `xarray.Dataset` subsets in canonical form.

2. **Simulator / Environment** – The “true” world during an episode. For the current milestone, a thin **truth layer** (`oceanbench-env/truth.py`) wraps provider-returned datasets and exposes `OceanTruthField.query()` for evaluation. Full episode stepping and dynamics are stubbed for later.

3. **Agents** – What robots believe and how they decide. The **field representation (field belief) models** live under `oceanbench-models/belief/field/`. They estimate the ocean field from sparse observations and provide predictions (and, when supported, uncertainty). These models are the focus of the current milestone; they will later be used by planners and tasks.

## Field-model branch (current milestone)

- **oceanbench-core**: Shared types (`Scenario`, `Observation`, `ObservationBatch`, `QueryPoints`), interpolation helpers, RNG.
- **oceanbench-env**: `OceanTruthField` for querying ground truth at arbitrary points.
- **oceanbench-models/belief/field**: Common interface `FieldBeliefModel` and implementations (Local Linear, GP, Sparse Online GP, Pseudo-Input GP, STGP, GMRF).
- **oceanbench-bench**: Evaluation harness `run_field_model_comparison`, result logging, save/load.
- **oceanbench-viz**: Maps (truth, prediction, uncertainty, error).
- **oceanbench-tasks**: Minimal mapping task `field_rmse_score` and common metrics (RMSE, MAE).

All field models share the same API: `fit()`, `update()`, `predict()`, `supports_uncertainty`, `supports_online_update`, `reset()`, and config/seed handling.
