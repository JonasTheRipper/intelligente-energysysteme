"""Planner (IncidentCommand) tests -- the two HARD identity requirements.

1. **v0.3 identity** -- a command with only a TankerFleet + indirect doctrine
   returns EXACTLY ``select_retardant_line`` cells as SUPPRESSED edits, in the
   same order (DESIGN §1 req 1).
2. **No-op identity** -- when every resource's budget is 0 the proposal is ``[]``
   (DESIGN §1 req 2), so the fire CA is bit-for-bit the v0.2 baseline.

Plus value-raster construction and the merge/dedup arbitration. Numpy-only.
"""
import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents.firefighter_core import select_retardant_line
from palaestrai_socal.agents.firefighting.planner import (
    IncidentCommand,
    value_raster_from_buses,
)
from palaestrai_socal.agents.firefighting.resources import (
    Dozers,
    HandCrews,
    TankerFleet,
    build_resources,
)

NR, NC = 24, 24
WIND_DIR = 0.0
STEP_MIN, CELL_M = 60.0, 50.0


def _front():
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    S[12, 9:15] = spaces.BURNING
    return S


def _fuel():
    return np.full((NR, NC), 3.0)


# -- HARD req 1: v0.3 identity --------------------------------------------
def test_v03_identity_tankers_only_indirect():
    S, F = _front(), _fuel()
    cmd = IncidentCommand(resources=[TankerFleet(n=3)], doctrine="indirect")
    muts = cmd.propose(S, F, wind_speed=8.0, wind_dir_deg=WIND_DIR,
                       step_min=STEP_MIN, cell_m=CELL_M)
    budget = TankerFleet(n=3).capacity(8.0, STEP_MIN, CELL_M)
    expected = select_retardant_line(S, F, WIND_DIR, budget)
    assert [(r, c) for (r, c, _s, _l) in muts] == list(expected)
    assert all(st == spaces.SUPPRESSED and lyr == spaces.LAYER_SUPPRESSION
               for (_r, _c, st, lyr) in muts)


def test_v03_identity_via_build_resources_auto():
    # the muscle's default path: build_resources(n_planes=3) + auto doctrine at
    # 8 m/s chaparral must still reduce to the v0.3 retardant line.
    S, F = _front(), _fuel()
    cmd = IncidentCommand(resources=build_resources(n_planes=3),
                          doctrine="auto")
    muts = cmd.propose(S, F, wind_speed=8.0, wind_dir_deg=WIND_DIR,
                       step_min=STEP_MIN, cell_m=CELL_M)
    budget = TankerFleet(n=3).capacity(8.0, STEP_MIN, CELL_M)
    expected = select_retardant_line(S, F, WIND_DIR, budget)
    assert [(r, c) for (r, c, _s, _l) in muts] == list(expected)
    assert all(st == spaces.SUPPRESSED for (_r, _c, st, _l) in muts)


# -- HARD req 2: no-op identity -------------------------------------------
def test_noop_identity_zero_budget():
    S, F = _front(), _fuel()
    # grounded tankers (wind 25) + zero ground resources -> nothing.
    cmd = IncidentCommand(resources=build_resources(n_planes=5), doctrine="auto")
    muts = cmd.propose(S, F, wind_speed=25.0, wind_dir_deg=WIND_DIR,
                       step_min=STEP_MIN, cell_m=CELL_M)
    assert muts == []


def test_noop_identity_empty_fleet():
    S, F = _front(), _fuel()
    cmd = IncidentCommand(resources=build_resources(), doctrine="auto")
    assert cmd.propose(S, F, 8.0, WIND_DIR, step_min=STEP_MIN,
                       cell_m=CELL_M) == []


# -- ground resources add CONTAINED line ----------------------------------
def test_air_plus_ground_adds_contained_line():
    S, F = _front(), _fuel()
    air_only = IncidentCommand(resources=[TankerFleet(n=3)],
                               doctrine="indirect")
    mixed = IncidentCommand(
        resources=[TankerFleet(n=3), HandCrews(n=4), Dozers(n=2)],
        doctrine="indirect")
    m_air = air_only.propose(S, F, 8.0, WIND_DIR, slope_deg=0.0,
                             step_min=STEP_MIN, cell_m=CELL_M)
    m_mix = mixed.propose(S, F, 8.0, WIND_DIR, slope_deg=0.0,
                          step_min=STEP_MIN, cell_m=CELL_M)
    # ground resources contribute CONTAINED cells; air-only never does. Where
    # air + ground overlap the same ahead-of-head geometry, the higher-priority
    # CONTAINED state wins the arbitration (so SUPPRESSED is upgraded, not kept).
    assert spaces.CONTAINED in {st for (_r, _c, st, _l) in m_mix}
    assert spaces.CONTAINED not in {st for (_r, _c, st, _l) in m_air}
    assert m_mix != m_air
    # determinism: identical inputs -> identical output.
    again = mixed.propose(S, F, 8.0, WIND_DIR, slope_deg=0.0,
                          step_min=STEP_MIN, cell_m=CELL_M)
    assert m_mix == again


def test_merge_dedup_upgrades_to_higher_priority():
    # two groups hitting the same cell: CONTAINED must win over SUPPRESSED.
    g_sup = [(5, 5, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)]
    g_con = [(5, 5, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)]
    merged = IncidentCommand._merge([g_sup, g_con])
    assert merged == [(5, 5, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)]
    # order independence of the WINNER (lower priority second does not override).
    merged2 = IncidentCommand._merge([g_con, g_sup])
    assert merged2 == [(5, 5, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)]


# -- value raster from bus->cell registration -----------------------------
def test_value_raster_from_buses():
    bus_cell = {1: (2, 3), 2: (2, 3), 3: (5, 5)}
    bus_value = {1: 4.0, 2: 6.0, 3: 2.0}
    V = value_raster_from_buses((NR, NC), bus_cell, bus_value)
    assert V.shape == (NR, NC)
    assert V[2, 3] == 10.0       # colliding buses sum
    assert V[5, 5] == 2.0
    assert V[0, 0] == 0.0
    # default weight 1.0 when no value supplied.
    V2 = value_raster_from_buses((NR, NC), {1: (1, 1)})
    assert V2[1, 1] == 1.0
