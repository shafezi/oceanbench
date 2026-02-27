# OceanBench

**Ocean data and analysis toolkit for Informative Path Planning (IPP)** — a modular collection of sub-packages for oceanographic data access, field representation, evaluation, and visualization.

## Structure

The repository follows the full intended OceanBench architecture. Current focus is the **field-model branch** (field representation and comparison).

| Package | Description |
|--------|-------------|
| **oceanbench-data-provider/** | Unified data provider (HYCOM, Copernicus); returns sanitized `xarray.Dataset` subsets. |
| **oceanbench-core/** | Shared types (`Scenario`, `Observation`, `ObservationBatch`, `QueryPoints`), interpolation, RNG. |
| **oceanbench-env/** | Truth layer `OceanTruthField` for querying ground truth; env/sim stubs for later. |
| **oceanbench-models/** | **Field belief models** (Local Linear, GP, Sparse Online GP, Pseudo-Input GP, STGP, GMRF) plus sensor/dynamics/comms/fusion stubs. |
| **oceanbench-tasks/** | Mapping task `field_rmse_score`, common metrics (RMSE, MAE); other task stubs. |
| **oceanbench-bench/** | Evaluation harness `run_field_model_comparison`, result logging, save/load. |
| **oceanbench-viz/** | Maps (truth, prediction, uncertainty, error). |
| **oceanbench-agents/** | Agent/belief stubs for later. |
| **oceanbench-policies/** | Policy/planner stubs for later. |

See `docs/architecture.md` and `docs/field_models.md` for details.

## Quick Start

**⚠️ Use a virtual environment** to avoid conflicts.

### Field model comparison (current milestone)

```bash
cd oceanbench
python -m venv oceanbench-venv
source oceanbench-venv/bin/activate   # Windows: oceanbench-venv\Scripts\activate

# Install packages in dependency order (from repo root)
pip install -e oceanbench-core
pip install -e oceanbench-env
pip install -e oceanbench-models
pip install -e oceanbench-tasks
pip install -e oceanbench-bench
pip install -e oceanbench-viz

# Run comparison (synthetic data; no provider needed)
python examples/compare_field_models.py
```

Results are printed and optionally saved under `results/`. To use real data, set `USE_PROVIDER = True` in the script and install `oceanbench-data-provider`.

### Data provider only

```bash
pip install oceanbench-data-provider
# or from source: cd oceanbench-data-provider && pip install -e .
```

## Sub-packages

### oceanbench-data-provider

**General-purpose ocean data provider** — provides sanitized, subsettable, and reproducible ocean model data from HYCOM and Copernicus Marine Service. Can be used independently for any oceanographic data access needs, or as part of the OceanBench IPP toolkit.

See [`oceanbench-data-provider/README.md`](oceanbench-data-provider/README.md) for details.

**Installation:**
```bash
pip install oceanbench-data-provider
# or with extras:
pip install "oceanbench-data-provider[movie]"  # for animations
# pip install "oceanbench-data-provider[dev,regrid,movie,notebooks]"  # for all
```

**Usage:**
```python
from oceanbench_data_provider import DataProvider, list_products, describe

provider = DataProvider()
ds = provider.subset(
    product_id="hycom_glbv0.08_reanalysis_53x",
    region={"lon": [-80, -70], "lat": [25, 35]},
    time=("2014-01-01", "2014-01-07"),
    variables=["temp", "sal", "u", "v"],
)
```

## Tests

From the repo root (with packages installed):

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT
