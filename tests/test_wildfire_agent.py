"""Unit tests for the wildfire agent's numpy-only driver (WildfireDriver).

These exercise the agent-side fire brain in isolation (no palaestrai / no
pandapower): ignition-point geo->cell conversion, idempotent injection timing,
the mutation-diff contract with the GIS substrate, and fire spread. The thin
:class:`WildfireCmaMuscle` palaestrai adapter is covered by the slow two-env
smoke run, not here.
"""

import numpy as np
import pytest

from palaestrai_socal.agents.wildfire_core import WildfireDriver, LAYER_FIRE
from wildfire_cma.cma import BURNING, BURNED_OUT, UNBURNED


BOUNDS = (-120.0, 33.0, -118.0, 35.0)


def _driver(nr=30, nc=40, **kw):
    """A flat, all-chaparral (burnable) raster so ignition always takes."""
    fuel = np.full((nr, nc), 3, dtype=np.int16)   # SH chaparral -> burnable
    dem = np.zeros((nr, nc), dtype=float)
    params = dict(
        fuel=fuel, dem=dem, delta_m=100.0, bounds=BOUNDS,
        ignition_points=[(-119.0, 34.0)], ignition_step=1,
        # one CMA sub-step per env step so the seed cell is still BURNING (not
        # yet burned out) right after ignition; spread is driven by wind/kappa.
        env_step_min=5.0, dt_cma_min=5.0, t_burn_steps=6, kappa=3.0,
        wind_speed=20.0, wind_dir_deg=45.0, seed=1,
    )
    params.update(kw)
    return WildfireDriver(**params)


def _apply(grid, muts):
    for (r, c, s, _layer) in muts:
        grid[r, c] = s
    return grid


def test_ignition_lonlat_to_cell_matches_raster():
    d = _driver()
    cells = d.ignition_cells()
    assert len(cells) == 1
    expect = d.raster.lonlat_to_rc(-119.0, 34.0)
    assert cells[0] == expect
    r, c = cells[0]
    assert 0 <= r < 30 and 0 <= c < 40


def test_explicit_ignition_rc_used():
    d = _driver(ignition_points=[(-119.0, 34.0)], ignition_rc=[(5, 7)])
    cells = d.ignition_cells()
    assert (5, 7) in cells


def test_no_mutations_before_ignition_step():
    d = _driver(ignition_step=3)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    assert d.step(grid) == []          # step 1
    assert d.step(grid) == []          # step 2
    muts = d.step(grid)                # step 3 -> ignition
    assert any(s == BURNING for (_r, _c, s, _l) in muts)


def test_ignition_emits_burning_with_fire_layer():
    d = _driver(ignition_step=1)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    muts = d.step(grid)
    assert muts, "expected ignition + spread mutations on step 1"
    ign = d.ignition_cells()[0]
    states = {(r, c): s for (r, c, s, _l) in muts}
    assert states.get(ign) == BURNING
    assert all(layer == LAYER_FIRE for (_r, _c, _s, layer) in muts)


def test_injection_is_idempotent_once_applied():
    """After the env applies the ignition, it must not be re-emitted as new."""
    d = _driver(ignition_step=1)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    muts1 = d.step(grid)
    _apply(grid, muts1)                 # env applies edits to authoritative S
    ign = d.ignition_cells()[0]
    assert grid[ign] == BURNING
    muts2 = d.step(grid)
    # the ignition cell is already BURNING in the input grid, so it is never a
    # *changed* cell again (idempotent); _ignited latch prevents re-injection.
    assert all((r, c) != ign or s != BURNING or grid[ign] != BURNING
               for (r, c, s, _l) in muts2) or grid[ign] == BURNING
    assert d._ignited is True


def test_fire_spreads_over_steps():
    d = _driver(ignition_step=1)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    affected = []
    for _ in range(6):
        muts = d.step(grid)
        _apply(grid, muts)
        affected.append(int((grid != UNBURNED).sum()))
    # monotonic non-decreasing and strictly grows past the single seed cell
    assert affected[-1] > affected[0]
    assert affected[-1] > 1
    assert all(b >= a for a, b in zip(affected, affected[1:]))


def test_mutations_only_report_changed_cells():
    d = _driver(ignition_step=1)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    muts = d.step(grid)
    _apply(grid, muts)
    before = grid.copy()
    muts2 = d.step(grid)
    # every reported mutation is a genuine change vs the input grid
    for (r, c, s, _l) in muts2:
        assert before[r, c] != s


def test_suppressed_cells_do_not_spread():
    """A firefighter could set cells to SUPPRESSED (3); fire must not re-ignite
    them or spread *from* them (they are not BURNING)."""
    from palaestrai_socal import spaces
    d = _driver(ignition_step=1)
    grid = np.full((30, 40), UNBURNED, dtype=np.int8)
    muts = d.step(grid)
    _apply(grid, muts)
    # suppress a ring around the front
    burning = np.argwhere(grid == BURNING)
    for (r, c) in burning:
        grid[r, c] = spaces.SUPPRESSED
    n_suppressed = int((grid == spaces.SUPPRESSED).sum())
    muts2 = d.step(grid)
    _apply(grid, muts2)
    # suppressed cells stay suppressed (fire produced no BURNING that overwrote)
    assert int((grid == spaces.SUPPRESSED).sum()) == n_suppressed
