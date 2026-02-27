"""
Use the Data Provider to inspect a subset (dims, variables, extent).

Example:
  python examples/inspect_subset.py

Requires oceanbench-data-provider to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    try:
        from oceanbench_data_provider import DataProvider
    except ImportError:
        print("oceanbench-data-provider is not installed. Install it and try again.")
        sys.exit(1)

    provider = DataProvider()
    ds = provider.subset(
        product_id="hycom_glbv0.08_reanalysis_53x",
        region={"lon": [-86, -74], "lat": [24, 36]},
        time=("2014-01-01", "2014-01-07"),
        variables=["temp", "sal"],
    )
    print("Dataset dimensions:", dict(ds.dims))
    print("Variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))
    if "lat" in ds.coords:
        print("Lat range:", float(ds.lat.min()), float(ds.lat.max()))
    if "lon" in ds.coords:
        print("Lon range:", float(ds.lon.min()), float(ds.lon.max()))


if __name__ == "__main__":
    main()
