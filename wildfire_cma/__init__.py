"""Wildfire Constrained-Mutation Automaton (GUARDIAN) package."""

from .cma import (
    BURNED_OUT,
    BURNING,
    HOUSE_FUEL_CLASS,
    UNBURNED,
    RasterStack,
    Theta,
    WildfireCMA,
    BASE_ROS_BY_FUEL,
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
    "HOUSE_FUEL_CLASS",
    "BASE_ROS_BY_FUEL",
    "DamageMapper",
    "DamageState",
    "build_socal_raster",
    "synthetic_socal",
    "SOCAL_BOUNDS",
]
