"""Unit tests for the wildfire cellular automaton (GUARDIAN tau operator).

Locks in the verified physical behaviours of WildfireCMA:
  * geographic (lon,lat) ignition seeds the correct cell,
  * fire spreads on burnable fuel and burns out after t_burn_steps,
  * non-burnable cells (fuel class 0) block / never ignite,
  * wind drives directional (anisotropic) spread,
  * high dead-fuel moisture damps / extinguishes spread (extinction moisture),
  * Theta.clamp() keeps parameters geophysically plausible.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wildfire_cma.cma import (  # noqa: E402
    BURNED_OUT,
    BURNING,
    UNBURNED,
    RasterStack,
    Theta,
    WildfireCMA,
)


def _flat_burnable(nrows=60, ncols=60, fuel_class=3, delta_m=30.0,
                   bounds=(-119.0, 34.0, -118.0, 35.0)):
    fuel = np.full((nrows, ncols), fuel_class, dtype=np.int16)
    dem = np.zeros((nrows, ncols), dtype=float)  # flat -> no slope effect
    return RasterStack(fuel=fuel, dem=dem, delta_m=delta_m, bounds=bounds)


def test_theta_clamp():
    t = Theta(wind_speed=999, wind_dir_deg=765, dead_fuel_moisture=0.001,
              kappa=99).clamp()
    assert 0 <= t.wind_speed <= 60
    assert 0 <= t.wind_dir_deg < 360
    assert 0.01 <= t.dead_fuel_moisture <= 0.40
    assert 1.0 <= t.kappa <= 8.0


def test_lonlat_ignition_seeds_one_cell():
    raster = _flat_burnable()
    # ignite at the centre of the bounds
    lon = (-119.0 + -118.0) / 2
    lat = (34.0 + 35.0) / 2
    cma = WildfireCMA(raster, Theta(ignition_points=[(lon, lat)]),
                      dt_cma_min=5, t_burn_steps=6, seed=1)
    assert int((cma.state == BURNING).sum()) == 1
    r, c = np.argwhere(cma.state == BURNING)[0]
    # cell should be near the raster centre
    assert abs(r - raster.shape[0] // 2) <= 1
    assert abs(c - raster.shape[1] // 2) <= 1


def test_fire_spreads_and_burns_out():
    raster = _flat_burnable()
    cma = WildfireCMA(
        raster,
        Theta(ignition_rc=[(30, 30)], wind_speed=10.0, kappa=2.0,
              dead_fuel_moisture=0.04),
        dt_cma_min=5, t_burn_steps=4, seed=1,
    )
    cma.advance(minutes=120)
    s = cma.stats()
    print("after 2h:", s)
    assert s["affected_cells"] > 1, "fire must spread beyond ignition"
    assert s["burned_cells"] > 0, "cells must burn out after t_burn_steps"


def test_nonburnable_blocks_spread():
    # a full firebreak column of non-burnable fuel splits the domain
    raster = _flat_burnable()
    raster.fuel[:, 35] = 0  # vertical non-burnable break
    cma = WildfireCMA(
        raster,
        Theta(ignition_rc=[(30, 20)], wind_speed=8.0, kappa=2.0,
              dead_fuel_moisture=0.05),
        dt_cma_min=5, t_burn_steps=6, seed=1,
    )
    cma.advance(minutes=240)
    fire = (cma.state == BURNING) | (cma.state == BURNED_OUT)
    # nothing should have jumped the firebreak to the far side
    assert fire[:, 36:].sum() == 0, "fire must not cross the non-burnable break"
    assert fire[:, :35].sum() > 1, "fire should spread on the ignition side"


def test_wind_drives_directional_spread():
    raster = _flat_burnable(nrows=81, ncols=81)
    # wind FROM the north-east (45 deg) -> fire pushed toward the south-west
    cma = WildfireCMA(
        raster,
        Theta(ignition_rc=[(40, 40)], wind_speed=20.0, wind_dir_deg=45.0,
              kappa=2.0, dead_fuel_moisture=0.04),
        dt_cma_min=5, t_burn_steps=10, seed=1,
    )
    cma.advance(minutes=180)
    fire = (cma.state == BURNING) | (cma.state == BURNED_OUT)
    rr, cc = np.where(fire)
    # downwind (SW) = higher row (south) and lower col (west) than ignition
    sw = int(((rr > 40) & (cc < 40)).sum())
    ne = int(((rr < 40) & (cc > 40)).sum())
    print(f"SW (downwind) cells={sw}  NE (upwind) cells={ne}")
    assert sw > ne, "fire should spread further downwind (SW) than upwind (NE)"


def test_high_moisture_extinguishes():
    raster = _flat_burnable()
    cma = WildfireCMA(
        raster,
        # moisture above the extinction threshold for chaparral
        Theta(ignition_rc=[(30, 30)], wind_speed=10.0, kappa=2.0,
              dead_fuel_moisture=0.40),
        dt_cma_min=5, t_burn_steps=6, seed=1,
    )
    cma.advance(minutes=180)
    s = cma.stats()
    print("wet fuel after 3h:", s)
    # the ignition cell burns out but the fire should not run away
    assert s["affected_cells"] <= 3, "high moisture must suppress spread"


if __name__ == "__main__":
    for fn in [
        test_theta_clamp,
        test_lonlat_ignition_seeds_one_cell,
        test_fire_spreads_and_burns_out,
        test_nonburnable_blocks_spread,
        test_wind_drives_directional_spread,
        test_high_moisture_extinguishes,
    ]:
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"PASS {fn.__name__}")
    print("\nALL CMA TESTS PASSED")
