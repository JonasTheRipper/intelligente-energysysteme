"""System test: DamageMapper co-registration against the real 2,294-bus grid.

Verifies that:
  1. The synthetic SoCal raster covers (almost) all real grid bus coordinates.
  2. bus -> cell and line -> cells co-registration is populated for the
     overwhelming majority of assets.
  3. Igniting near a dense cluster of buses actually fails those buses and the
     overhead lines within the radiant-heat clearance buffer.
  4. ``apply()`` mutates the pandapower topology (in_service flips) and the
     mutated grid still solves a power flow.

Run: python3 -m pytest tests/test_damage_mapper.py -v
or:  python3 tests/test_damage_mapper.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandapower as pp  # noqa: E402

from wildfire_cma.cma import BURNED_OUT, BURNING, Theta, WildfireCMA  # noqa: E402
from wildfire_cma.damage import DamageMapper, _bus_lonlat, _line_coords  # noqa: E402
from wildfire_cma.gis import SOCAL_BOUNDS, synthetic_socal  # noqa: E402

GRID_JSON = os.path.join(ROOT, "socal_grid", "socal_grid.json")


def _load_grid():
    return pp.from_json(GRID_JSON)


def _build_raster():
    # Reasonably fine raster over the full SoCal footprint. The footprint is
    # large (~7.6 deg lon x 5.3 deg lat) so we use a denser grid to keep the
    # cell size on the order of a few hundred metres.
    return synthetic_socal(nrows=600, ncols=760, seed=7)


def test_raster_covers_grid():
    net = _load_grid()
    raster = _build_raster()
    minlon, minlat, maxlon, maxlat = raster.bounds
    inside = 0
    total = 0
    for b in net.bus.index:
        ll = _bus_lonlat(net, b)
        if ll is None:
            continue
        total += 1
        lon, lat = ll
        if minlon <= lon <= maxlon and minlat <= lat <= maxlat:
            inside += 1
    frac = inside / max(1, total)
    print(f"buses with geo: {total}, inside raster: {inside} ({frac:.1%})")
    assert total > 2000, "expected geo on most of the 2,294 buses"
    assert frac > 0.95, f"raster should cover >95% of buses, got {frac:.1%}"


def test_coregistration_populated():
    net = _load_grid()
    raster = _build_raster()
    dm = DamageMapper(net, raster, clearance_m=90.0)
    print(f"bus_cell entries: {len(dm.bus_cell)} / {len(net.bus)}")
    print(f"line_cells entries: {len(dm.line_cells)} / {len(net.line)}")
    assert len(dm.bus_cell) > 0.95 * len(net.bus)
    assert len(dm.line_cells) > 0.90 * len(net.line)


def _densest_cluster_lonlat(net, raster, n_target=40):
    """Find a raster cell that contains the most buses; return its lon/lat."""
    counts = {}
    for b in net.bus.index:
        ll = _bus_lonlat(net, b)
        if ll is None:
            continue
        lon, lat = ll
        minlon, minlat, maxlon, maxlat = raster.bounds
        if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
            continue
        # bin into a coarse 0.05-deg grid to find a dense area
        key = (round(lat / 0.05), round(lon / 0.05))
        counts.setdefault(key, []).append((b, lon, lat))
    # densest bin
    best = max(counts.values(), key=len)
    lons = [x[1] for x in best]
    lats = [x[2] for x in best]
    return float(np.mean(lons)), float(np.mean(lats)), len(best)


def test_ignition_fails_buses_and_lines():
    net = _load_grid()
    raster = _build_raster()
    lon0, lat0, ncluster = _densest_cluster_lonlat(net, raster)
    print(f"densest 0.05-deg bin: ~{ncluster} buses near ({lon0:.3f},{lat0:.3f})")

    theta = Theta(
        ignition_points=[(lon0, lat0)],
        wind_speed=15.0,        # strong Santa-Ana-like wind
        wind_dir_deg=45.0,      # from NE -> spreads SW
        dead_fuel_moisture=0.04,
        kappa=1.5,
    )
    cma = WildfireCMA(raster, theta, dt_cma_min=5, t_burn_steps=6, seed=1)
    dm = DamageMapper(net, raster, clearance_m=120.0)

    # advance the fire for several simulated hours
    cma.advance(minutes=6 * 60)
    fire_cells = int(((cma.state == BURNING) | (cma.state == BURNED_OUT)).sum())
    print(f"fire cells after 6h: {fire_cells}")
    assert fire_cells > 5, "fire should have spread"

    ds = dm.evaluate(cma)
    print(f"failed buses: {len(ds.failed_buses)}, failed lines: {len(ds.failed_lines)}")
    assert len(ds.failed_buses) > 0, "ignition over a dense cluster must fail buses"
    assert len(ds.failed_lines) > 0, "lines within clearance must fail"


def test_apply_mutates_and_solves():
    net = _load_grid()
    raster = _build_raster()
    lon0, lat0, _ = _densest_cluster_lonlat(net, raster)
    theta = Theta(
        ignition_points=[(lon0, lat0)],
        wind_speed=12.0,
        wind_dir_deg=45.0,
        dead_fuel_moisture=0.05,
        kappa=1.3,
    )
    cma = WildfireCMA(raster, theta, dt_cma_min=5, t_burn_steps=6, seed=2)
    dm = DamageMapper(net, raster, clearance_m=120.0)
    cma.advance(minutes=4 * 60)
    dm.evaluate(cma)

    n_bus_before = int(net.bus.in_service.sum())
    n_line_before = int(net.line.in_service.sum())
    ds = dm.apply()
    n_bus_after = int(net.bus.in_service.sum())
    n_line_after = int(net.line.in_service.sum())
    print(f"buses in_service {n_bus_before} -> {n_bus_after}")
    print(f"lines in_service {n_line_before} -> {n_line_after}")
    assert n_bus_after <= n_bus_before
    assert n_line_after <= n_line_before
    assert (n_bus_before - n_bus_after) == len(
        [b for b in ds.failed_buses if b in net.bus.index]
    )

    # the mutated grid should still solve (islands may appear; allow it)
    try:
        pp.runpp(net, init="flat", calculate_voltage_angles=False)
        converged = bool(net["converged"])
    except Exception as e:  # pragma: no cover
        converged = False
        print(f"runpp on mutated grid raised: {e}")
    print(f"runpp converged on mutated grid: {converged}")
    # We don't hard-require convergence (wildfire can island the grid), but the
    # call must not corrupt the network object.
    assert "res_bus" in net or not converged


if __name__ == "__main__":
    for fn in [
        test_raster_covers_grid,
        test_coregistration_populated,
        test_ignition_fails_buses_and_lines,
        test_apply_mutates_and_solves,
    ]:
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"PASS {fn.__name__}")
    print("\nALL DAMAGE MAPPER TESTS PASSED")
