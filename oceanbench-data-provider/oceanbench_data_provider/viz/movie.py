"""
Optional movie generation (mp4/gif) from time series of maps.
"""

import io
from pathlib import Path
from typing import Any, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from oceanbench_data_provider.viz.colormaps import get_cmap


def make_movie(
    ds: xr.Dataset,
    var: str,
    output_path: str,
    format: str = "gif",
    fps: int = 4,
    depth_idx: Optional[int] = None,
    variable_info: Optional[Dict[str, Any]] = None,
    show_date: bool = True,
) -> str:
    """
    Generate movie (gif or mp4) of variable over time.

    Args:
        ds: xarray Dataset.
        var: Variable name.
        output_path: Output file path.
        format: 'gif' or 'mp4'.
        fps: Frames per second.
        depth_idx: Depth index for 3D vars.
        variable_info: Optional dict mapping var names to VariableInfo objects (for units/description).
        show_date: If True and time dimension exists, show actual date in title instead of index.

    Returns:
        Path to created file.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("Movie generation requires 'imageio'. pip install imageio")

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if var not in ds:
        raise KeyError(f"Variable {var} not in dataset.")

    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=depth_idx if depth_idx is not None else 0)

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
        units = ds[var].attrs.get("units", "")
    if description == var:
        description = ds[var].attrs.get(
            "long_name", ds[var].attrs.get("standard_name", var)
        )
    
    # Build depth label
    depth_label = ""
    if "depth" in ds.coords and depth_idx is not None:
        depth_val = float(ds.depth.values[depth_idx])
        depth_label = f" at {depth_val:.1f}m depth"
    elif "depth" in ds.coords:
        depth_val = float(ds.depth.values[0])
        depth_label = f" at {depth_val:.1f}m depth"

    # Fix colorbar range across all frames for consistent comparison
    vmin = float(da.min().values)
    vmax = float(da.max().values)
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0  # avoid singular range

    frames = []
    n_times = len(da.time) if "time" in da.dims else 1

    print(f"\nGenerating {n_times} frames...")

    for i in range(n_times):
        if "time" in da.dims:
            frame_data = da.isel(time=i)
            if show_date:
                try:
                    actual_time = pd.to_datetime(ds.time.values[i])
                    time_label = actual_time.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError, IndexError):
                    time_label = f"t={i}"
            else:
                time_label = f"t={i}"
        else:
            frame_data = da
            time_label = "(no time)"
        
        # Build title
        title = f"{description} - {time_label}{depth_label}"
        
        # Build colorbar label
        cbar_label = f"{var}"
        if units:
            cbar_label = f"{var} [{units}]"
        
        if has_cartopy:
            fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=ccrs.PlateCarree()))
            ax.coastlines(resolution="50m")
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
            ax.gridlines(draw_labels=True, alpha=0.5)
            frame_data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cbar_kwargs={"shrink": 0.8, "label": cbar_label},
                cmap=get_cmap(var),
                vmin=vmin,
                vmax=vmax,
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            frame_data.plot(
                ax=ax,
                cbar_kwargs={"shrink": 0.8, "label": cbar_label},
                cmap=get_cmap(var),
                vmin=vmin,
                vmax=vmax,
            )
        
        ax.set_title(title)
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        try:
            img = imageio.v2.imread(buf)
        except AttributeError:
            img = imageio.imread(buf)
        frames.append(img)
        buf.close()
        plt.close(fig)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "gif":
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    else:
        imageio.mimsave(str(output_path), frames, fps=fps)

    return str(output_path)
