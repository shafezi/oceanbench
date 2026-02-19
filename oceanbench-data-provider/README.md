# OceanBench Data Provider

**General-purpose ocean data provider** — provides sanitized, subsettable, and reproducible ocean model data from HYCOM and Copernicus Marine Service. Use it for any oceanographic data access needs, whether for research, analysis, visualization, or as part of larger workflows like Informative Path Planning (IPP).

## Features

- **Unified API** across HYCOM and Copernicus Marine (Physical & BGC)
- **Human-readable variables** (e.g., `temp`, `sal`, `u`, `v`, `ssh`) with canonical mapping
- **Subsetting by region and time** to avoid terabyte-scale downloads
- **Canonical output format** (consistent coordinates, dimensions, units)
- **Deterministic caching** with provenance metadata
- **Dataset Cards** for each product (notebook, CLI, Python API)
- **Quicklook visualization** (maps, sections, time series, optional movies)

## Supported Datasets

| Product | Type | Coverage |
|---------|------|----------|
| **HYCOM** | Global Reanalysis | 1994–2015 (v53.X) |
| **HYCOM** | Global Analysis | 2014–2024 (v56.3, 57.2, 92.8, 57.7, 92.9, 93) |
| **Copernicus Marine (Physical)** | Global Reanalysis | 1993–2025 |
| **Copernicus Marine (Physical)** | Global Analysis | 2022–Now + 10-day forecast |
| **Copernicus Marine (BGC)** | Global Reanalysis | 1993–2025 |
| **Copernicus Marine (BGC)** | Global Analysis | 2021–Now + 10-day forecast |

## Installation

**⚠️ Important: We strongly recommend using a virtual environment** to avoid conflicts with system packages and to keep your Python environment clean.

### From PyPI (recommended for end users)

```bash
# Create and activate a virtual environment (recommended)
# From the parent oceanbench/ directory (if working with multiple sub-packages):
cd ..  # go to oceanbench/
python -m venv oceanbench-venv
source oceanbench-venv/bin/activate  # On Windows: oceanbench-venv\Scripts\activate

# Install the package
pip install oceanbench-data-provider

# Or with optional dependencies
pip install "oceanbench-data-provider[movie]"  # for movie generation
# pip install "oceanbench-data-provider[dev,regrid,movie,notebooks]"  # all extras
```

### From source (development)

```bash
# From the parent oceanbench/ directory:
cd ..  # go to oceanbench/
python -m venv oceanbench-venv
source oceanbench-venv/bin/activate  # On Windows: oceanbench-venv\Scripts\activate
cd oceanbench-data-provider

# Install core package
pip install -e .

# Or with optional dependencies
# pip install -e ".[dev,regrid,movie,notebooks]"  # with extras
```

## Configuration

OceanBench works out of the box with sensible defaults. You only need configuration if you want to change where data is cached or how authentication is provided.

**No config file (default)**  
If you do nothing, OceanBench will cache data in `~/.oceanbench/cache`.

**Quick override (recommended)**  
Set an environment variable to change the cache location (no YAML file needed):

```bash
export OCEANBENCH_CACHE_DIR=/path/to/oceanbench-cache
```

**Optional: config file**  
If you prefer a config file, create `~/.oceanbench/config.yaml` (or set `OCEANBENCH_CONFIG` to point to a different file) with:

```yaml
cache_dir: ~/.oceanbench/cache
# Optional: Copernicus credentials (prefer copernicusmarine login instead)
# copernicusmarine:
#   username: ""
#   password: ""
```
To use a config file stored elsewhere:

```bash
export OCEANBENCH_CONFIG=/path/to/your/config.yaml
```

**Copernicus authentication**: Run `copernicusmarine login` once (or set `COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`). See [Copernicus Marine Toolbox](https://toolbox.marine.copernicus.eu/) docs.

## Quick Start

### Python API

```python
from oceanbench_data_provider import DataProvider
from oceanbench_data_provider.catalog import list_products, describe

# List products
products = list_products()
print(products)

# Describe a product (Dataset Card)
card = describe("hycom_glbv0.08_reanalysis_53x")
print(card)

# Subset and fetch
provider = DataProvider()
ds = provider.subset(
    product_id="hycom_glbv0.08_reanalysis_53x",
    region={"lon": [-80, -70], "lat": [25, 35]},
    time=("2014-01-01", "2014-01-07"),  # HYCOM reanalysis: 1994-2015
    variables=["temp", "sal", "u", "v"],
)
print(ds)
```

### CLI (Command Line)

```bash
oceanbench list-products
oceanbench describe hycom_glbv0.08_reanalysis_53x
oceanbench estimate --product hycom_glbv0.08_reanalysis_53x --region="-80,-70,25,35" --time 2014-01-01,2014-01-07 --vars temp,sal
oceanbench fetch --product hycom_glbv0.08_reanalysis_53x --region="-80,-70,25,35" --time 2014-01-01,2014-01-07 --vars temp,sal
# See what's cached (cache key, product, variables, time)
oceanbench list-cache
oceanbench list-cache --product hycom_glbv0.08_reanalysis_53x   # filter by product
# Quicklook: use cache key from list-cache; --var to pick variable
oceanbench quicklook --cache-key <cache-key> --var temp -o map.png
```

### Notebooks

See `notebooks/` for interactive workflows:
- `complete_workflow.ipynb` — Full workflow overview

## Canonical Data Model

All outputs use:
- **Dimensions**: `time`, `lat`, `lon`, `depth`
- **Longitude**: `[-180, 180]` (documented)
- **Latitude**: monotonically increasing
- **Time**: `datetime64` with explicit metadata
- **Missing values**: `NaN` (fill values converted)
- **Land mask**: documented policy (no interpolation across land)

## Documentation

- [Dataset Cards](docs/dataset_cards.md) — per-product metadata
- [Configuration](docs/configuration.md) — cache, credentials, paths
- [Sanitization](docs/sanitization.md) — canonical model, processing summary

## License

MIT
