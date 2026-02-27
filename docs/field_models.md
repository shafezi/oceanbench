# Field representation models

All field models implement the common interface defined in `oceanbench-models/belief/field/base.py`.

## Implemented models

| Model | Uncertainty | Online update | Notes |
|-------|-------------|---------------|--------|
| **LocalLinearFieldModel** | No | No | k-NN local linear regression; fast baseline. |
| **GPFieldModel** | Yes | No | Full GP with RBF kernel; O(N³) fit. |
| **SparseOnlineGPFieldModel** | Yes | Yes | Sliding window of max_points; refit on window at each update. |
| **PseudoInputGPFieldModel** | Yes | No | FITC with random pseudo-input subset; O(N M² + M³). |
| **STGPFieldModel** | Yes | No | Spatio-temporal GP with separable space-time RBF kernel; requires time in observations. |
| **GMRFFieldModel** | Yes (approx) | No | 2D grid GMRF with Laplacian prior; bilinear interpolation for prediction. |

## Conventions

- **Coordinates**: lat (degrees N), lon (degrees E), optional time (datetime64), optional depth (m).
- **ObservationBatch / QueryPoints**: NumPy arrays; use `as_features(include_time=..., include_depth=...)` for model input.
- **Predictions**: `FieldPrediction(mean, std, metadata)` with NumPy arrays; `std` is `None` when uncertainty is not supported.

## Configuration

Each model accepts a `config` dict and optional `seed`. See docstrings and `config` dataclasses in each module (e.g. `LocalLinearConfig`, `GPConfig`) for hyperparameters.
