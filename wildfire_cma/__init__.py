"""Wildfire Constrained-Mutation Automaton (GUARDIAN) package."""

from .cma import (
    BURNED_OUT,
    BURNING,
    UNBURNED,
    RasterStack,
    Theta,
    WildfireCMA,
)
from .damage import DamageMapper, DamageState
from .gis import SOCAL_BOUNDS, build_socal_raster, synthetic_socal

__all__ = [
    "WildfireCMA",
    "RasterStack",
    "Theta",
    "UNBURNED",
    "BURNING",
    "BURNED_OUT",
    "DamageMapper",
    "DamageState",
    "build_socal_raster",
    "synthetic_socal",
    "SOCAL_BOUNDS",
]
