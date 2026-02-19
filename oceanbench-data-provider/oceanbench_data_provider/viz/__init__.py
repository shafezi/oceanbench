"""Visualization: quicklook maps, sections, time series, optional movies."""

from oceanbench_data_provider.viz.colormaps import get_cmap
from oceanbench_data_provider.viz.quicklook import (
    quicklook_map,
    quicklook_section,
    quicklook_timeseries,
    plot_region_bounds,
    interactive_globe,
)
from oceanbench_data_provider.viz.movie import make_movie

__all__ = [
    "get_cmap",
    "quicklook_map",
    "quicklook_section",
    "quicklook_timeseries",
    "plot_region_bounds",
    "interactive_globe",
    "make_movie",
]
