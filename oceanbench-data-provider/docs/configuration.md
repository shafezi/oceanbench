# Configuration

## Config File

Default path: `~/.oceanbench/config.yaml`

Override with environment variable: `OCEANBENCH_CONFIG`

Example:

```yaml
cache_dir: ~/.oceanbench/cache
# Optional: Copernicus credentials (prefer copernicusmarine login)
# copernicusmarine:
#   username: ""
#   password: ""
```

## Cache Location

Default: `~/.oceanbench/cache`

Override: set `OCEANBENCH_CACHE_DIR` or `cache_dir` in config.

## Copernicus Marine Authentication

**Recommended**: Run once:

```bash
copernicusmarine login
```

Or set environment variables:
- `COPERNICUSMARINE_SERVICE_USERNAME`
- `COPERNICUSMARINE_SERVICE_PASSWORD`

See [Copernicus Marine Toolbox](https://toolbox.marine.copernicus.eu/) for details.
