"""Regression + behaviour tests for the SUPPRESSED firebreak in the CMA spread.

Two guarantees the v0.3 design hangs on (see ``_v0.3_IMPL_BRIEF.md``):

1. **Firebreak** -- a SUPPRESSED retardant line is non-ignitable: the fire never
   spreads *into* or *through* it.
2. **Bit-for-bit no-op** -- with ZERO SUPPRESSED cells present, the spread step
   is identical to v0.2. We prove this by replaying each transition with a
   reference implementation that omits the SUPPRESSED guard entirely; because
   the guard sits before any RNG draw and is never taken when no SUPPRESSED cell
   exists, the two must agree cell-for-cell AND draw the RNG identically.

Numpy-only (``wildfire_cma.cma`` imports numpy alone).
"""

import numpy as np

from wildfire_cma.cma import (
    WildfireCMA, RasterStack, Theta, _MOORE,
    UNBURNED, BURNING, BURNED_OUT, SUPPRESSED,
)


def _raster(nr=25, nc=25, delta_m=100.0):
    fuel = np.full((nr, nc), 3, dtype=np.int16)   # all chaparral -> burnable
    dem = np.zeros((nr, nc), dtype=float)          # flat -> slope factor == 1
    return RasterStack(fuel=fuel, dem=dem, delta_m=delta_m,
                       bounds=(-119.0, 34.0, -118.0, 35.0))


def _theta(**kw):
    p = dict(ignition_rc=[(12, 12)], wind_speed=12.0, wind_dir_deg=45.0,
             dead_fuel_moisture=0.05, kappa=4.0)
    p.update(kw)
    return Theta(**p)


# -- behaviour: SUPPRESSED is a firebreak ---------------------------------
def test_fire_does_not_cross_suppressed_wall():
    cma = WildfireCMA(_raster(), _theta(), dt_cma_min=5.0, t_burn_steps=99, seed=1)
    # a full vertical SUPPRESSED wall at col 14, fire ignited west of it.
    cma.state[:, 14] = SUPPRESSED
    for _ in range(40):
        cma.step()
    east = cma.state[:, 15:]
    # nothing east of the wall ever ignites; the wall itself stays SUPPRESSED.
    assert np.count_nonzero(east == BURNING) == 0
    assert np.count_nonzero(east == BURNED_OUT) == 0
    assert np.all(cma.state[:, 14] == SUPPRESSED)


def test_suppressed_cell_never_ignites():
    cma = WildfireCMA(_raster(7, 7), _theta(ignition_rc=[(3, 3)]),
                      dt_cma_min=5.0, t_burn_steps=99, seed=3)
    cma.state[3, 4] = SUPPRESSED               # right next to the seed
    for _ in range(20):
        cma.step()
    assert cma.state[3, 4] == SUPPRESSED


# -- regression: zero SUPPRESSED -> bit-for-bit identical to the no-guard path --
def _ref_transition(cma, state, burn_timer):
    """A copy of ``WildfireCMA._transition`` WITHOUT the SUPPRESSED guard.

    Identical loop order and identical RNG draw order to the real transition, so
    when no SUPPRESSED cell exists the real (guarded) step must match this one
    cell-for-cell. If anyone ever draws RNG inside/ before the guard, this test
    diverges.
    """
    nrows, ncols = cma.raster.shape
    burning = np.argwhere(state == BURNING)
    new_ignitions = []
    for (r, c) in burning:
        for (dr, dc, diag) in _MOORE:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            if state[nr, nc] != UNBURNED:
                continue
            if not cma._burnable(nr, nc):
                continue
            p = cma.spread_prob(r, c, dr, dc, diag)
            if p > 0 and cma.rng.random() < p:
                new_ignitions.append((nr, nc))
    burn_timer[state == BURNING] += 1
    burned = (state == BURNING) & (burn_timer >= cma.t_burn_steps)
    state[burned] = BURNED_OUT
    for (nr, nc) in new_ignitions:
        if state[nr, nc] == UNBURNED:
            state[nr, nc] = BURNING
            burn_timer[nr, nc] = 0
    return len(new_ignitions)


def test_spread_unchanged_with_no_suppressed_cells():
    real = WildfireCMA(_raster(), _theta(), dt_cma_min=5.0, t_burn_steps=6, seed=7)
    ref = WildfireCMA(_raster(), _theta(), dt_cma_min=5.0, t_burn_steps=6, seed=7)
    # identical seed/raster/theta -> identical initial state, timer, and RNG.
    assert np.array_equal(real.state, ref.state)
    for step in range(30):
        real.step()                            # guarded transition (production)
        _ref_transition(ref, ref.state, ref.burn_timer)  # no-guard reference
        assert np.array_equal(real.state, ref.state), f"diverged at step {step}"
    # and the fire actually did something (so the equality is meaningful).
    assert np.count_nonzero(real.state != UNBURNED) > 1
