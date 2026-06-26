"""Tiny end-to-end smoke test for the CI ``smoke`` stage.

This is deliberately fast and light: it builds a *small* synthetic raster,
runs the GUARDIAN cellular automaton for a few steps, and checks the fire
spreads — without loading the 5.9 MB grid or solving a power flow. It exercises
the import graph and the core CMA <-> GIS contract so a broken integration is
caught on every push, while the full grid co-simulation stays in the manual
``system`` stage.

Marked ``unit`` so it runs in the fast CI stage.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wildfire_cma.cma import BURNED_OUT, BURNING, Theta, WildfireCMA  # noqa: E402
from wildfire_cma.gis import SOCAL_BOUNDS, synthetic_socal  # noqa: E402

pytestmark = pytest.mark.unit


def test_imports_and_bounds():
    # SoCal footprint sanity: covers LA + San Diego + Bakersfield longitudes.
    minlon, minlat, maxlon, maxlat = SOCAL_BOUNDS
    assert minlon < -118.0 < maxlon
    assert minlat < 34.0 < maxlat


def test_small_raster_fire_spreads():
    # small + fast: 60 x 80 cells
    raster = synthetic_socal(nrows=60, ncols=80, seed=3)
    theta = Theta(
        ignition_rc=[(30, 40)],
        wind_speed=15.0,
        wind_dir_deg=225.0,
        dead_fuel_moisture=0.03,
        kappa=3.0,
    )
    cma = WildfireCMA(raster, theta, dt_cma_min=5.0, t_burn_steps=6, seed=3)
    start = int(np.count_nonzero(cma.state != 0))
    cma.advance(minutes=60)  # one "hour"
    stats = cma.stats()
    affected = stats["affected_cells"]
    assert affected > start, "fire should spread from the seeded cell"
    assert stats["front_size"] >= 0
    # state values are only ever UNBURNED/BURNING/BURNED_OUT
    assert set(np.unique(cma.state)).issubset({0, BURNING, BURNED_OUT})
