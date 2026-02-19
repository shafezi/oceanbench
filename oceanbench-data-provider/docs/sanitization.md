# Sanitization Pipeline

## What "Sanitized" Means

All outputs follow a **canonical data model**:

- **Dimensions**: `time`, `lat`, `lon`, `depth`
- **Longitude**: `[-180, 180]`
- **Latitude**: monotonically increasing
- **Time**: `datetime64[ns]` where possible
- **Missing values**: `NaN` (fill values converted)
- **Land mask**: no interpolation across land; land points remain NaN

## Pipeline Steps

1. Normalize coordinates (rename dims, convert lon 0–360 → -180–180)
2. QC: fill value → NaN, apply masks
3. Standardize variable names (canonical mapping)
4. Units normalization
5. Optional regrid to target lat–lon grid

## Provenance

Each cached subset includes:

- `request.json` — original request
- `dataset_card.json` — product metadata
- `processing.json` — processing summary
- `log.txt` — pipeline version, cache key

## Regridding

Optional target grid: `target_grid={"lat_res": 0.1, "lon_res": 0.1, "method": "bilinear"}`

Methods: `bilinear`, `nearest`, `conservative` (when xESMF available).
