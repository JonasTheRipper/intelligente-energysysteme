"""Capacity tests for the v0.4 firefighting resources (numpy-only).

These touch only :mod:`palaestrai_socal.agents.firefighting.resources` and the
v0.3 :mod:`palaestrai_socal.agents.firefighter_core`, so they run in the light
CI stage without palaestrai. They pin the complementarity the doctrine relies
on (DESIGN §4): air resources ground in wind, ground resources do not but
derate on slope, and each resource's productivity scales with its count.
"""
import numpy as np  # noqa: F401  (kept for parity with the rest of the suite)

from palaestrai_socal.agents.firefighter_core import retardant_budget
from palaestrai_socal.agents.firefighting.resources import (
    Dozers,
    Engines,
    HandCrews,
    HeloFleet,
    TankerFleet,
    build_resources,
)
from palaestrai_socal import spaces

STEP_MIN = 60.0
CELL_M = 50.0


# -- tankers reproduce v0.3 exactly ---------------------------------------
def test_tanker_capacity_equals_retardant_budget():
    fleet = TankerFleet(n=3)
    for wind in (0.0, 8.0, 13.0, 16.0, 18.0, 25.0):
        assert fleet.capacity(wind, STEP_MIN, CELL_M) == retardant_budget(
            3, wind, STEP_MIN, CELL_M)
    assert fleet.state == spaces.SUPPRESSED
    assert fleet.layer == spaces.LAYER_SUPPRESSION


def test_tanker_grounds_at_18ms():
    fleet = TankerFleet(n=5)
    assert fleet.capacity(8.0, STEP_MIN, CELL_M) > 0
    assert fleet.capacity(18.0, STEP_MIN, CELL_M) == 0   # hard ground
    assert fleet.capacity(25.0, STEP_MIN, CELL_M) == 0


# -- helos tolerate MORE wind than fixed-wing tankers ----------------------
def test_helo_grounds_higher_than_tanker():
    helo = HeloFleet(n=3)
    tanker = TankerFleet(n=3)
    # at 18 m/s the tanker is grounded but the helo (grounds at 22) still flies.
    assert tanker.capacity(18.0, STEP_MIN, CELL_M) == 0
    assert helo.capacity(18.0, STEP_MIN, CELL_M) > 0
    # both grounded well above 22 m/s.
    assert helo.capacity(22.0, STEP_MIN, CELL_M) == 0
    assert helo.capacity(30.0, STEP_MIN, CELL_M) == 0


def test_helo_capacity_scales_with_count():
    w = 8.0
    c1 = HeloFleet(n=1).capacity(w, STEP_MIN, CELL_M)
    c3 = HeloFleet(n=3).capacity(w, STEP_MIN, CELL_M)
    assert 0 < c1 <= c3


# -- hand crews: NOT wind-grounded, but slope-limited ----------------------
def test_crew_not_wind_grounded():
    crew = HandCrews(n=4)
    # the wind that grounds aircraft does not stop crews.
    flat_calm = crew.capacity(0.0, STEP_MIN, CELL_M, slope_deg=0.0)
    flat_gale = crew.capacity(30.0, STEP_MIN, CELL_M, slope_deg=0.0)
    assert flat_calm > 0
    assert flat_calm == flat_gale          # wind-independent
    assert crew.grounded(30.0) is False
    assert crew.state == spaces.CONTAINED


def test_crew_slope_derate_and_cutoff():
    crew = HandCrews(n=4)
    flat = crew.capacity(0.0, STEP_MIN, CELL_M, slope_deg=0.0)
    mid = crew.capacity(0.0, STEP_MIN, CELL_M, slope_deg=30.0)   # half_deg
    steep = crew.capacity(0.0, STEP_MIN, CELL_M, slope_deg=45.0)  # cutoff
    assert flat > mid > 0
    assert steep == 0                       # too steep to build line


# -- dozers: faster than crews, lower slope tolerance ----------------------
def test_dozer_faster_but_lower_slope_cutoff():
    dozer = Dozers(n=2)
    crew = HandCrews(n=2)
    # on flat ground a dozer out-produces an equal count of hand crews.
    assert dozer.capacity(0.0, STEP_MIN, CELL_M) > crew.capacity(
        0.0, STEP_MIN, CELL_M)
    # dozers cut out at 35 deg where crews (cutoff 45) can still work (given
    # enough crews to clear one cell after the slope derate).
    assert Dozers(n=4).capacity(0.0, STEP_MIN, CELL_M, slope_deg=36.0) == 0
    assert HandCrews(n=8).capacity(0.0, STEP_MIN, CELL_M, slope_deg=36.0) > 0
    assert dozer.state == spaces.CONTAINED


# -- engines: count of protectable points; wind/slope independent ----------
def test_engine_capacity_is_point_count():
    eng = Engines(n=3)
    assert eng.capacity(0.0, STEP_MIN, CELL_M) >= 1
    # over a 60-min step: 3 engines x 1.5 points/h = 4.5 -> floor 4.
    assert eng.capacity(0.0, 60.0, CELL_M) == 4
    assert eng.grounded(30.0) is False
    assert eng.state == spaces.CONTAINED


def test_zero_count_resources_zero_budget():
    for cls in (TankerFleet, HeloFleet, HandCrews, Dozers, Engines):
        assert cls(n=0).capacity(8.0, STEP_MIN, CELL_M) == 0


# -- build_resources order + counts ---------------------------------------
def test_build_resources_fixed_order():
    res = build_resources(n_planes=1, n_helos=2, n_crews=3, n_dozers=4,
                          n_engines=5)
    assert [type(r).__name__ for r in res] == [
        "TankerFleet", "HeloFleet", "HandCrews", "Dozers", "Engines"]
    assert [r.n for r in res] == [1, 2, 3, 4, 5]
