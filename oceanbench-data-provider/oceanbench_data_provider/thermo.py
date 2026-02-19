from __future__ import annotations

from typing import Hashable

import xarray as xr
import gsw


def add_pressure_and_sound_speed(
    ds: xr.Dataset,
    *,
    depth_name: Hashable = "depth",
    lat_name: Hashable = "lat",
    temp_name: Hashable = "temp",
    sal_name: Hashable = "sal",
    pressure_name: Hashable = "pressure",
    sound_speed_name: Hashable = "sound_speed",
) -> xr.Dataset:
    """
    Add TEOS-10 sea pressure and sound speed to a dataset using GSW.

    This assumes:
    - depth is in meters, positive downward
    - lat is in degrees north
    - sal is Absolute Salinity (g/kg)
    - temp is close to Conservative Temperature (degC)

    The computed variables are:
    - pressure [dbar]
    - sound_speed [m/s]
    """
    for name in (depth_name, lat_name, temp_name, sal_name):
        if name not in ds:
            raise KeyError(f"Dataset is missing required variable/coordinate {name!r}")

    depth = ds[depth_name]
    lat = ds[lat_name]

    # Compute 2D pressure field (depth, lat) then broadcast to the full grid
    depth_2d, lat_2d = xr.broadcast(depth, lat)

    pressure_2d = xr.apply_ufunc(
        gsw.p_from_z,
        -depth_2d,  # gsw expects z (m, negative below sea surface)
        lat_2d,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )

    # Broadcast pressure to the full 3D grid of a reference ocean variable
    ref = ds[temp_name]
    pressure = pressure_2d.broadcast_like(ref)

    # Mask pressure where the reference variable is missing (e.g. land points)
    pressure = pressure.where(ref.notnull())
    pressure.name = pressure_name
    pressure.attrs.update(
        {
            "long_name": "Sea pressure",
            "standard_name": "sea_water_pressure",
            "units": "dbar",
            "comment": "Computed from depth and latitude using TEOS-10 gsw.p_from_z",
        }
    )

    sal = ds[sal_name]
    temp = ds[temp_name]

    sound_speed = xr.apply_ufunc(
        gsw.sound_speed,
        sal,
        temp,
        pressure,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )
    sound_speed = sound_speed.rename(sound_speed_name)
    sound_speed.attrs.update(
        {
            "long_name": "Speed of sound in seawater",
            "units": "m/s",
            "comment": "Computed with TEOS-10 gsw.sound_speed from salinity, temperature, and pressure",
        }
    )

    return ds.assign({pressure_name: pressure, sound_speed_name: sound_speed})

