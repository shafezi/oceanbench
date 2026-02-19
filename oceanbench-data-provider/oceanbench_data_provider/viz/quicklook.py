"""
Quicklook plots: maps, sections, time series.
"""

from pathlib import Path
from typing import Any, Optional, Dict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from oceanbench_data_provider.viz.colormaps import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr


def quicklook_map(
    ds: xr.Dataset,
    var: str,
    time_idx: int = 0,
    depth_idx: Optional[int] = None,
    ax: Optional[Any] = None,
    save_path: Optional[str] = None,
    variable_info: Optional[Dict[str, Any]] = None,
    show_date: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot 2D map of variable at given time (and optional depth).

    Args:
        ds: xarray Dataset in canonical form.
        var: Variable name.
        time_idx: Time index.
        depth_idx: Depth index (for 3D vars); None = surface.
        ax: Matplotlib axes (optional).
        save_path: Path to save figure.
        variable_info: Optional dict mapping var names to VariableInfo objects (for units/description).
        show_date: If True and time dimension exists, show actual date in title instead of index.
        **kwargs: Additional arguments passed to xarray plot().
    """
    if var not in ds:
        raise KeyError(f"Variable {var} not in dataset. Available: {list(ds.data_vars)}")

    da = ds[var]
    
    # Build time label
    time_label = ""
    if "time" in da.dims:
        da = da.isel(time=time_idx)
        if show_date:
            try:
                actual_time = pd.to_datetime(ds.time.values[time_idx])
                time_label = f" - {actual_time.strftime('%Y-%m-%d %H:%M')}"
            except (ValueError, TypeError, IndexError):
                time_label = f" (t={time_idx})"
        else:
            time_label = f" (t={time_idx})"
    else:
        time_label = " (no time dimension)"
    
    # Build depth label
    depth_label = ""
    if "depth" in da.dims:
        if depth_idx is None:
            depth_idx = 0
        da = da.isel(depth=depth_idx)
        if "depth" in ds.coords:
            depth_val = float(ds.depth.values[depth_idx])
            depth_label = f" at {depth_val:.1f}m depth"
        else:
            depth_label = f" at depth index {depth_idx}"
    
    # Get variable info for units and description
    units = ""
    description = var
    if variable_info and var in variable_info:
        var_info = variable_info[var]
        if hasattr(var_info, "units"):
            units = var_info.units
        if hasattr(var_info, "description"):
            description = var_info.description

    # Fallback to xarray attributes when not provided via variable_info
    if not units:
        units = da.attrs.get("units", "")
    if description == var:
        description = da.attrs.get("long_name", da.attrs.get("standard_name", var))
    
    # Build title
    title_parts = [description]
    if time_label:
        title_parts.append(time_label.strip())
    if depth_label:
        title_parts.append(depth_label)
    title = "".join(title_parts)
    
    # Build colorbar label
    cbar_label = f"{var}"
    if units:
        cbar_label = f"{var} [{units}]"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    else:
        fig = ax.figure

    # Set up colorbar kwargs
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    cbar_kwargs.setdefault("shrink", 0.8)
    cbar_kwargs.setdefault("label", cbar_label)

    da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs=cbar_kwargs,
        cmap=kwargs.pop("cmap", get_cmap(var)),
        **kwargs,
    )
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.gridlines(draw_labels=True, alpha=0.5)

    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def quicklook_section(
    ds: xr.Dataset,
    var: str,
    time_idx: int = 0,
    along: str = "lon",
    lat_or_lon: Optional[float] = None,
    ax: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot vertical section along lat or lon.

    Args:
        ds: xarray Dataset.
        var: Variable name.
        time_idx: Time index.
        along: 'lon' (section along lon at fixed lat) or 'lat' (along lat at fixed lon).
        lat_or_lon: Fixed lat or lon value for the section.
    """
    if var not in ds:
        raise KeyError(f"Variable {var} not in dataset.")

    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=time_idx)
    if "depth" not in da.dims:
        raise ValueError("Variable has no depth dimension.")

    if along == "lon":
        if lat_or_lon is None:
            lat_or_lon = float(da.lat.mean())
        da = da.sel(lat=lat_or_lon, method="nearest")
        xdim = "lon"
    else:
        if lat_or_lon is None:
            lat_or_lon = float(da.lon.mean())
        da = da.sel(lon=lat_or_lon, method="nearest")
        xdim = "lat"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    da.plot(y="depth", yincrease=False, ax=ax)
    ax.set_title(f"{var} section along {along} at {along}={lat_or_lon}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def quicklook_timeseries(
    ds: xr.Dataset,
    var: str,
    lat: float,
    lon: float,
    depth_idx: Optional[int] = None,
    ax: Optional[Any] = None,
    save_path: Optional[str] = None,
) -> Any:
    """Plot time series at a point."""
    if var not in ds:
        raise KeyError(f"Variable {var} not in dataset.")

    da = ds[var].sel(lat=lat, lon=lon, method="nearest")
    if "depth" in da.dims:
        da = da.isel(depth=depth_idx if depth_idx is not None else 0)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    da.plot(ax=ax)
    ax.set_title(f"{var} at ({lat:.2f}, {lon:.2f})")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_region_bounds(
    region: Dict[str, Any],
    margin: float = 10,
    ax: Optional[Any] = None,
    title: str = "Selected Geographic Region",
    figsize: tuple = (10, 6),
) -> Any:
    """
    Plot geographic region bounds on a map (lon/lat box).

    Args:
        region: Dict with "lon" and "lat" keys, each [min, max].
        margin: Degrees to pad around the box for map extent.
        ax: Matplotlib axes (optional). If None, creates new figure with cartopy.
        title: Figure title.
        figsize: Figure size when ax is None.

    Returns:
        Axes object.
    """

    lon_min, lon_max = region["lon"][0], region["lon"][1]
    lat_min, lat_max = region["lat"][0], region["lat"][1]

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())
        )
    ax.coastlines(resolution="50m")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    lon_box = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lat_box = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        lon_box,
        lat_box,
        "r-",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label="Selected Region",
    )
    ax.fill(lon_box, lat_box, color="red", alpha=0.2, transform=ccrs.PlateCarree())

    ax.set_xlim(lon_min - margin, lon_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)
    ax.gridlines(draw_labels=True, alpha=0.5)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    return ax


def interactive_globe(
    bbox: Optional[Dict[str, Any]] = None,
    step_deg: float = 5,
    height: int = 600,
) -> Any:
    """
    Create an interactive Plotly globe to help users choose lon/lat bounds.
    
    Args:
        bbox: Optional dict with "lon" and "lat" keys, each [min, max], to overlay a bounding box.
        step_deg: Grid spacing in degrees for hoverable points (default: 5).
        height: Figure height in pixels (default: 600).
    
    Returns:
        Plotly figure object.
    """
    
    # Create a light lon/lat grid of hoverable points
    lons = np.arange(-180, 181, step_deg)
    lats = np.arange(-90, 91, step_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    fig = go.Figure(
        go.Scattergeo(
            lon=lon_grid.ravel(),
            lat=lat_grid.ravel(),
            mode="markers",
            marker=dict(size=3, opacity=0.15, color="black"),
            hovertemplate="lon=%{lon:.1f}°, lat=%{lat:.1f}°<extra></extra>",
            name="lon/lat",
        )
    )
    
    # Overlay bounding box if provided
    if bbox is not None:
        lon_min, lon_max = bbox["lon"]
        lat_min, lat_max = bbox["lat"]
        box_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
        box_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
        fig.add_trace(
            go.Scattergeo(
                lon=box_lons,
                lat=box_lats,
                mode="lines",
                line=dict(color="red", width=3),
                hoverinfo="skip",
                name="bbox preview",
            )
        )
    
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True,
        coastlinecolor="rgba(80,80,80,0.8)",
        showland=True,
        landcolor="rgb(245,245,245)",
        showocean=True,
        oceancolor="rgb(210,230,255)",
        lonaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.25)"),
        lataxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.25)"),
    )
    fig.update_layout(
        title="Globe (hover to read lon/lat) — drag to rotate",
        margin=dict(l=0, r=0, t=50, b=0),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    
    return fig
