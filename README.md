# OceanBench

**Ocean data and analysis toolkit for Informative Path Planning (IPP)** — a modular collection of sub-packages for oceanographic data access, analysis, and visualization, designed to support IPP research and applications.

## Structure

This repository contains multiple independently installable sub-packages:

- **`oceanbench-data-provider/`** — Unified data provider for ocean products (HYCOM, Copernicus Marine)
- *(Future sub-packages: `oceanbench-ipp/`, `oceanbench-viz/`, etc.)*

## Quick Start

**⚠️ Important: We strongly recommend using a virtual environment** to avoid conflicts with system packages and to keep your Python environment clean.

### From PyPI (recommended for end users)

```bash
# Create and activate a virtual environment (recommended)
# From the oceanbench/ directory
python -m venv oceanbench-venv
source oceanbench-venv/bin/activate  # On Windows: oceanbench-venv\Scripts\activate

# Install the data provider (core)
pip install oceanbench-data-provider

# Or with optional dependencies
pip install "oceanbench-data-provider[movie]"  # for movie generation
# pip install "oceanbench-data-provider[dev,regrid,movie,notebooks]"  # all extras
```

### From source (development)

```bash
# From the oceanbench/ directory:
python -m venv oceanbench-venv
source oceanbench-venv/bin/activate  # On Windows: oceanbench-venv\Scripts\activate
cd oceanbench-data-provider
pip install -e .  # core only
# pip install -e ".[dev,regrid,movie,notebooks]"  # with extras
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

## License

MIT
