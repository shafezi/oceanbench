"""
CLI for oceanbench: list-products, describe, estimate, fetch, list-cache, quicklook.
"""

import argparse
import json
import os
import sys


def _load_config() -> dict:
    config_path = os.environ.get("OCEANBENCH_CONFIG", os.path.expanduser("~/.oceanbench/config.yaml"))
    config = {}
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass
    if "cache_dir" not in config:
        config["cache_dir"] = os.environ.get("OCEANBENCH_CACHE_DIR", os.path.expanduser("~/.oceanbench/cache"))
    return config


def cmd_list_products(args: argparse.Namespace) -> int:
    from oceanbench_data_provider.catalog import list_products
    products = list_products()
    for p in products:
        print(p)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    from oceanbench_data_provider.catalog import describe
    try:
        card = describe(args.product_id)
        if args.json:
            print(json.dumps(card.to_dict(), indent=2))
        else:
            print(card)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    config = _load_config()
    provider = __import__("oceanbench_data_provider.provider", fromlist=["DataProvider"]).DataProvider(config)
    region = _parse_region(args.region)
    time_range = _parse_time(args.time)
    variables = args.vars.split(",") if args.vars else ["temp", "sal"]
    try:
        est = provider.estimate_size(args.product, region, time_range, variables)
        if args.json:
            print(json.dumps(est, indent=2))
        else:
            print(f"Estimated size: {est.get('estimate_mb', 0):.1f} MB")
            if est.get("note"):
                print(f"Note: {est['note']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    config = _load_config()
    provider = __import__("oceanbench_data_provider.provider", fromlist=["DataProvider"]).DataProvider(config)
    region = _parse_region(args.region)
    time_range = _parse_time(args.time)
    variables = args.vars.split(",") if args.vars else ["temp", "sal"]
    try:
        ds = provider.subset(
            product_id=args.product,
            region=region,
            time=time_range,
            variables=variables,
            overwrite=args.force,
        )
        print(f"Fetched: {list(ds.data_vars)}")
        print(f"Dims: {dict(ds.dims)}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_list_cache(args: argparse.Namespace) -> int:
    from oceanbench_data_provider.cache.store import CacheStore
    config = _load_config()
    store = CacheStore(config.get("cache_dir"))
    status = store.status(product_id=args.product)
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    entries = status.get("entries", [])
    if not entries:
        print("No cached datasets.")
        print(f"Cache dir: {status.get('cache_dir', '')}")
        return 0
    print(f"Cache dir: {status.get('cache_dir', '')}")
    print(f"Cached datasets: {len(entries)}\n")
    for e in entries:
        req = e.get("request", {})
        product_id = req.get("product_id", "?")
        time_range = req.get("time", ())
        time_str = f"{time_range[0]} to {time_range[1]}" if len(time_range) == 2 else "?"
        variables = req.get("variables", [])
        vars_str = ", ".join(variables) if variables else "?"
        print(f"  {e['cache_key']}")
        print(f"    product: {product_id}")
        print(f"    variables: {vars_str}")
        print(f"    time: {time_str}")
        print()
    return 0


def cmd_quicklook(args: argparse.Namespace) -> int:
    from oceanbench_data_provider.cache.store import CacheStore
    config = _load_config()
    store = CacheStore(config.get("cache_dir"))
    cache_key = args.cache_key
    zarr_path = store.root / cache_key / f"{cache_key}.zarr"
    if not zarr_path.exists():
        print(f"Cache key not found: {cache_key}", file=sys.stderr)
        return 1
    import xarray as xr
    ds = xr.open_zarr(zarr_path).load()
    var = args.var or (list(ds.data_vars)[0] if ds.data_vars else None)
    if not var:
        print("No variables in dataset", file=sys.stderr)
        return 1
    from oceanbench_data_provider.viz.quicklook import quicklook_map
    ax = quicklook_map(ds, var, save_path=args.output)
    if not args.output:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


def _parse_region(s: str) -> dict:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("Region must be lon_min,lon_max,lat_min,lat_max")
    return {"lon": [parts[0], parts[1]], "lat": [parts[2], parts[3]]}


def _parse_time(s: str) -> tuple[str, str]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Time must be start,end (e.g. 2020-01-01,2020-01-07)")
    return (parts[0], parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(prog="oceanbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-products")

    lc = sub.add_parser("list-cache", help="List cached datasets and their variables")
    lc.add_argument("--product", "-p", help="Filter by product id")
    lc.add_argument("--json", action="store_true", help="Output as JSON")

    d = sub.add_parser("describe")
    d.add_argument("product_id")
    d.add_argument("--json", action="store_true")

    e = sub.add_parser("estimate")
    e.add_argument("--product", "-p", required=True)
    e.add_argument("--region", "-r", required=True, help="lon_min,lon_max,lat_min,lat_max")
    e.add_argument("--time", "-t", required=True, help="start,end")
    e.add_argument("--vars", "-v", default="temp,sal")
    e.add_argument("--json", action="store_true")

    f = sub.add_parser("fetch")
    f.add_argument("--product", "-p", required=True)
    f.add_argument("--region", "-r", required=True)
    f.add_argument("--time", "-t", required=True)
    f.add_argument("--vars", "-v", default="temp,sal")
    f.add_argument("--force", action="store_true", help="Overwrite cache")

    q = sub.add_parser("quicklook")
    q.add_argument("--cache-key", "-k", required=True)
    q.add_argument("--var", "-v", help="Variable to plot")
    q.add_argument("--output", "-o", help="Save path")

    args = parser.parse_args()
    handlers = {
        "list-products": cmd_list_products,
        "list-cache": cmd_list_cache,
        "describe": cmd_describe,
        "estimate": cmd_estimate,
        "fetch": cmd_fetch,
        "quicklook": cmd_quicklook,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
