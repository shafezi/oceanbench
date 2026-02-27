"""
Thin environment-side wrappers for truth data and (later) simulators.

For this milestone, the primary entry point is :class:`OceanTruthField`,
which exposes convenient query methods around provider-returned datasets.
"""

from .truth import OceanTruthField

__all__ = ["OceanTruthField"]

