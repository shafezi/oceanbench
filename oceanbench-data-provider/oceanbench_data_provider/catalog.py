"""
Dataset Card schema and metadata fetching/validation.

Each product has a Dataset Card with human-readable metadata visible in
notebook, CLI, and Python API.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class VariableInfo:
    """Canonical variable info."""

    canonical_name: str
    provider_names: list[str]
    description: str
    units: str
    is_2d: bool  # surface-only vs 3D


@dataclass
class DatasetCard:
    """
    Dataset Card: human-readable metadata for a product.
    """

    product_id: str
    provider_name: str
    time_coverage_start: str
    time_coverage_end: str
    temporal_resolution: str
    spatial_resolution: str
    grid_type: str
    variables: list[VariableInfo]
    vertical_coord_info: Optional[str] = None
    access_method: str = ""
    credential_required: bool = False
    known_caveats: list[str] = field(default_factory=list)
    source_adapter: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Export to dict for JSON serialization."""
        return {
            "product_id": self.product_id,
            "provider_name": self.provider_name,
            "time_coverage": {
                "start": self.time_coverage_start,
                "end": self.time_coverage_end,
            },
            "temporal_resolution": self.temporal_resolution,
            "spatial_resolution": self.spatial_resolution,
            "grid_type": self.grid_type,
            "variables": [
                {
                    "canonical_name": v.canonical_name,
                    "provider_names": v.provider_names,
                    "description": v.description,
                    "units": v.units,
                    "is_2d": v.is_2d,
                }
                for v in self.variables
            ],
            "vertical_coord_info": self.vertical_coord_info,
            "access_method": self.access_method,
            "credential_required": self.credential_required,
            "known_caveats": self.known_caveats,
            "source_adapter": self.source_adapter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetCard":
        """Load from dict."""
        vars_list = [
            VariableInfo(
                canonical_name=v["canonical_name"],
                provider_names=v["provider_names"],
                description=v["description"],
                units=v["units"],
                is_2d=v["is_2d"],
            )
            for v in d.get("variables", [])
        ]
        tc = d.get("time_coverage", {})
        return cls(
            product_id=d["product_id"],
            provider_name=d["provider_name"],
            time_coverage_start=tc.get("start", ""),
            time_coverage_end=tc.get("end", ""),
            temporal_resolution=d.get("temporal_resolution", ""),
            spatial_resolution=d.get("spatial_resolution", ""),
            grid_type=d.get("grid_type", ""),
            variables=vars_list,
            vertical_coord_info=d.get("vertical_coord_info"),
            access_method=d.get("access_method", ""),
            credential_required=d.get("credential_required", False),
            known_caveats=d.get("known_caveats", []),
            source_adapter=d.get("source_adapter", ""),
        )

    def __str__(self) -> str:
        lines = [
            f"=== Dataset Card: {self.product_id} ===",
            f"Provider: {self.provider_name}",
            f"Time coverage: {self.time_coverage_start} — {self.time_coverage_end}",
            f"Temporal resolution: {self.temporal_resolution}",
            f"Spatial resolution: {self.spatial_resolution}",
            f"Grid type: {self.grid_type}",
        ]
        if self.vertical_coord_info:
            lines.append(f"Vertical levels: {self.vertical_coord_info}")
        lines.append(f"Access: {self.access_method}")
        if self.credential_required:
            lines.append("Credentials: Required (Copernicus Marine login)")
        lines.append("\nVariables (canonical):")
        for v in self.variables:
            dims = "2D" if v.is_2d else "3D"
            lines.append(f"  - {v.canonical_name} ({dims}): {v.description} [{v.units}]")
        if self.known_caveats:
            lines.append("\nCaveats:")
            for c in self.known_caveats:
                lines.append(f"  - {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Product registry: maps product_id -> DatasetCard factory
# ---------------------------------------------------------------------------

_CANONICAL_VARS_PHYSICAL = [
    VariableInfo("temp", ["water_temp", "thetao", "water_temperature"], "Sea water potential temperature", "degC", False),
    VariableInfo("sal", ["salinity", "so", "salinity"], "Sea water salinity", "g/kg", False),
    VariableInfo("u", ["water_u", "uo", "eastward_velocity"], "Eastward velocity", "m/s", False),
    VariableInfo("v", ["water_v", "vo", "northward_velocity"], "Northward velocity", "m/s", False),
    VariableInfo("ssh", ["surf_el", "zos", "sea_surface_height"], "Sea surface height", "m", True),
]

_CANONICAL_VARS_BGC = [
    VariableInfo("chl", ["chlorophyll", "chl"], "Chlorophyll concentration", "mg/m3", False),
    VariableInfo("no3", ["nitrate", "no3"], "Nitrate", "mmol/m3", False),
    VariableInfo("o2", ["oxygen", "o2"], "Dissolved oxygen", "mmol/m3", False),
    VariableInfo("po4", ["phosphate", "po4"], "Phosphate", "mmol/m3", False),
]

# HYCOM analysis version date ranges (from THREDDS catalog)
_HYCOM_ANALYSIS_VERSIONS = {
    "56.3": ("2014-07-01", "2016-04-30"),
    "57.2": ("2016-05-01", "2017-01-31"),
    "92.8": ("2017-02-01", "2017-05-31"),
    "57.7": ("2017-06-01", "2017-09-30"),
    "92.9": ("2017-10-01", "2017-12-31"),
    "93": ("2018-01-01", "2024-09-04"),  # spans multiple grids; end from GLBy0.08
}


def _make_hycom_reanalysis_card() -> DatasetCard:
    return DatasetCard(
        product_id="hycom_glbv0.08_reanalysis_53x",
        provider_name="HYCOM",
        time_coverage_start="1994-01-01",
        time_coverage_end="2015-12-30",
        temporal_resolution="3-hourly",
        spatial_resolution="1/12° (~9 km)",
        grid_type="curvilinear (lat/lon)",
        variables=_CANONICAL_VARS_PHYSICAL,
        vertical_coord_info="41 layers (z-level)",
        access_method="OPeNDAP/THREDDS (tds.hycom.org)",
        credential_required=False,
        known_caveats=[
            "Fill values vary by variable; pipeline converts to NaN.",
            "Longitude may be 0–360; canonical output uses [-180, 180].",
        ],
        source_adapter="hycom",
    )


def _make_hycom_analysis_card(version: str) -> DatasetCard:
    start, end = _HYCOM_ANALYSIS_VERSIONS.get(version, ("?", "?"))
    return DatasetCard(
        product_id=f"hycom_glbv0.08_analysis_{version.replace('.', '_')}",
        provider_name="HYCOM",
        time_coverage_start=start,
        time_coverage_end=end,
        temporal_resolution="3-hourly",
        spatial_resolution="1/12° (~9 km)",
        grid_type="curvilinear (lat/lon)",
        variables=_CANONICAL_VARS_PHYSICAL,
        vertical_coord_info="41 layers (z-level)",
        access_method="OPeNDAP/THREDDS (tds.hycom.org)",
        credential_required=False,
        known_caveats=[
            "Multiple experiments cover different date ranges; version determines dates.",
            "Longitude may be 0–360; canonical output uses [-180, 180].",
        ],
        source_adapter="hycom",
    )


def _make_copernicus_phy_reanalysis_card() -> DatasetCard:
    return DatasetCard(
        product_id="copernicus_phy_reanalysis_001_030",
        provider_name="Copernicus Marine",
        time_coverage_start="1993-01-01",
        time_coverage_end="2025-12-31",
        temporal_resolution="daily",
        spatial_resolution="1/12° (~9 km)",
        grid_type="ORCA grid (tripolar)",
        variables=_CANONICAL_VARS_PHYSICAL,
        vertical_coord_info="50 levels (0–5500 m)",
        access_method="Copernicus Marine Toolbox (subset)",
        credential_required=True,
        known_caveats=[
            "Requires Copernicus Marine login (copernicusmarine login).",
            "Dataset: GLOBAL_MULTIYEAR_PHY_001_030.",
        ],
        source_adapter="copernicus_phy",
    )


def _make_copernicus_phy_analysis_card() -> DatasetCard:
    return DatasetCard(
        product_id="copernicus_phy_analysis_001_024",
        provider_name="Copernicus Marine",
        time_coverage_start="2022-01-01",
        time_coverage_end="rolling + 10-day forecast",
        temporal_resolution="6-hourly (varies by variable)",
        spatial_resolution="1/12° (~9 km)",
        grid_type="ORCA grid (tripolar)",
        variables=_CANONICAL_VARS_PHYSICAL,
        vertical_coord_info="50 levels (0–5500 m)",
        access_method="Copernicus Marine Toolbox (subset)",
        credential_required=True,
        known_caveats=[
            "Requires Copernicus Marine login.",
            "Dataset: GLOBAL_ANALYSISFORECAST_PHY_001_024.",
        ],
        source_adapter="copernicus_phy",
    )


def _make_copernicus_bgc_reanalysis_card() -> DatasetCard:
    return DatasetCard(
        product_id="copernicus_bgc_reanalysis_001_029",
        provider_name="Copernicus Marine (BGC)",
        time_coverage_start="1993-01-01",
        time_coverage_end="2025-12-31",
        temporal_resolution="daily",
        spatial_resolution="1/4° (~25 km)",
        grid_type="ORCA grid (tripolar)",
        variables=_CANONICAL_VARS_BGC,  # chl, no3, o2, po4, si, nppv (no temp/sal)
        vertical_coord_info="75 levels",
        access_method="Copernicus Marine Toolbox (subset)",
        credential_required=True,
        known_caveats=[
            "Requires Copernicus Marine login.",
            "Dataset: GLOBAL_MULTIYEAR_BGC_001_029.",
        ],
        source_adapter="copernicus_bgc",
    )


def _make_copernicus_bgc_analysis_card() -> DatasetCard:
    return DatasetCard(
        product_id="copernicus_bgc_analysis_001_028",
        provider_name="Copernicus Marine (BGC)",
        time_coverage_start="2021-01-01",
        time_coverage_end="rolling + 10-day forecast",
        temporal_resolution="daily",
        spatial_resolution="1/4° (~25 km)",
        grid_type="ORCA grid (tripolar)",
        variables=_CANONICAL_VARS_BGC + _CANONICAL_VARS_PHYSICAL[:2],
        vertical_coord_info="75 levels",
        access_method="Copernicus Marine Toolbox (subset)",
        credential_required=True,
        known_caveats=[
            "Requires Copernicus Marine login.",
            "Dataset: GLOBAL_ANALYSISFORECAST_BGC_001_028.",
        ],
        source_adapter="copernicus_bgc",
    )


_PRODUCT_REGISTRY: dict[str, DatasetCard] = {}
for _card in [
    _make_hycom_reanalysis_card(),
    _make_copernicus_phy_reanalysis_card(),
    _make_copernicus_phy_analysis_card(),
    _make_copernicus_bgc_reanalysis_card(),
    _make_copernicus_bgc_analysis_card(),
]:
    _PRODUCT_REGISTRY[_card.product_id] = _card

# Add HYCOM analysis versions
for _ver in _HYCOM_ANALYSIS_VERSIONS:
    _card = _make_hycom_analysis_card(_ver)
    _PRODUCT_REGISTRY[_card.product_id] = _card


def list_products() -> list[str]:
    """List available product IDs."""
    return sorted(_PRODUCT_REGISTRY.keys())


def describe(product_id: str) -> DatasetCard:
    """
    Return the Dataset Card for a product.
    Raises KeyError if product_id is unknown.
    """
    if product_id not in _PRODUCT_REGISTRY:
        raise KeyError(
            f"Unknown product: {product_id}. Available: {list_products()}"
        )
    return _PRODUCT_REGISTRY[product_id]
