"""CONTAINED (v0.4) arbitration + CA-guard tests.

HARD req 3 (DESIGN §1): CONTAINED is a distinct integer arbitrated strictly
*between* SUPPRESSED and BURNED_OUT, the priority is total/tie-free, and a
CONTAINED cell is a non-ignitable firebreak (like SUPPRESSED) that, when absent,
leaves the v0.2 spread bit-for-bit unchanged. Plus: CONTAINED does NOT age out.
"""
import numpy as np

from palaestrai_socal import spaces


# -- priority is total, tie-free, and correctly ordered -------------------
def test_state_priority_total_and_ordered():
    pr = spaces.STATE_PRIORITY
    # every valid state has a priority, and all priorities are distinct.
    assert set(pr.keys()) == set(spaces.VALID_STATES)
    assert len(set(pr.values())) == len(pr)        # tie-free
    # the documented ordering: BURNED_OUT > CONTAINED > SUPPRESSED > ... .
    assert pr[spaces.SUPPRESSED] < pr[spaces.CONTAINED] < pr[spaces.BURNED_OUT]
    assert pr[spaces.CONTAINED] == 5 - 1   # between SUPPRESSED(3) and BURNED_OUT(5)
    assert spaces.CONTAINED == 5 and spaces.CONTAINED in spaces.VALID_STATES


# -- arbitration: CONTAINED beats SUPPRESSED/BURNING, loses to BURNED_OUT --
def test_contained_outranks_suppressed_and_burning():
    state = np.zeros((4, 4), dtype=np.int8)
    muts = [
        (1, 1, spaces.BURNING, spaces.LAYER_FIRE),
        (1, 1, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION),
        (1, 1, spaces.CONTAINED, spaces.LAYER_SUPPRESSION),
    ]
    out = spaces.arbitrate_mutations(state, muts)
    assert out[1, 1] == spaces.CONTAINED       # highest priority among the three


def test_contained_never_overwrites_burned_out():
    state = np.zeros((4, 4), dtype=np.int8)
    state[2, 2] = spaces.BURNED_OUT            # terminal
    out = spaces.arbitrate_mutations(
        state, [(2, 2, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)])
    assert out[2, 2] == spaces.BURNED_OUT


def test_arbitration_order_independent():
    state = np.zeros((3, 3), dtype=np.int8)
    a = [(0, 0, spaces.BURNING, spaces.LAYER_FIRE),
         (0, 0, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)]
    out1 = spaces.arbitrate_mutations(state, a)
    out2 = spaces.arbitrate_mutations(state, list(reversed(a)))
    assert np.array_equal(out1, out2)
    assert out1[0, 0] == spaces.CONTAINED


# -- CA: CONTAINED is a firebreak; absent -> bit-for-bit v0.2 -------------
def test_fire_does_not_cross_contained_wall():
    from wildfire_cma.cma import (
        WildfireCMA, RasterStack, Theta, BURNING, BURNED_OUT, CONTAINED,
    )
    fuel = np.full((25, 25), 3, dtype=np.int16)
    dem = np.zeros((25, 25), dtype=float)
    raster = RasterStack(fuel=fuel, dem=dem, delta_m=100.0,
                         bounds=(-119.0, 34.0, -118.0, 35.0))
    theta = Theta(ignition_rc=[(12, 12)], wind_speed=12.0, wind_dir_deg=45.0,
                  dead_fuel_moisture=0.05, kappa=4.0)
    cma = WildfireCMA(raster, theta, dt_cma_min=5.0, t_burn_steps=99, seed=1)
    cma.state[:, 14] = CONTAINED               # full vertical containment wall
    for _ in range(40):
        cma.step()
    east = cma.state[:, 15:]
    assert np.count_nonzero(east == BURNING) == 0
    assert np.count_nonzero(east == BURNED_OUT) == 0
    assert np.all(cma.state[:, 14] == CONTAINED)


def test_contained_cell_never_ignites():
    from wildfire_cma.cma import WildfireCMA, RasterStack, Theta, CONTAINED
    fuel = np.full((7, 7), 3, dtype=np.int16)
    dem = np.zeros((7, 7), dtype=float)
    raster = RasterStack(fuel=fuel, dem=dem, delta_m=100.0,
                         bounds=(-119.0, 34.0, -118.0, 35.0))
    theta = Theta(ignition_rc=[(3, 3)], wind_speed=12.0, wind_dir_deg=45.0,
                  dead_fuel_moisture=0.05, kappa=4.0)
    cma = WildfireCMA(raster, theta, dt_cma_min=5.0, t_burn_steps=99, seed=3)
    cma.state[3, 4] = CONTAINED
    for _ in range(20):
        cma.step()
    assert cma.state[3, 4] == CONTAINED


# -- CONTAINED does NOT age out (unlike SUPPRESSED) -----------------------
def test_contained_does_not_age_out():
    # age_suppressed only touches SUPPRESSED cells; CONTAINED must persist.
    from palaestrai_socal.agents.firefighter_core import age_suppressed
    age = np.zeros((5, 5), dtype=np.int32)
    state = np.full((5, 5), spaces.UNBURNED, dtype=np.int8)
    state[1, 1] = spaces.SUPPRESSED
    state[3, 3] = spaces.CONTAINED
    # run many ageing steps well past any SUPPRESS_PERSIST horizon.
    for _ in range(50):
        state, age = age_suppressed(state, age)
    assert state[3, 3] == spaces.CONTAINED     # ground line persists
    assert state[1, 1] != spaces.CONTAINED     # retardant aged (not promoted)
