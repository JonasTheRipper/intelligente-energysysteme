"""Unit tests for the firefighter's numpy-only driver (firefighter_core).

These exercise the aero-tanker decision logic in isolation (no palaestrai / no
pandapower): the wind grounding/degrade curve, the linear retardant budget, the
downwind head detection, the ahead-of-head contiguous line selection, and the
retardant-line ageing. The thin :class:`FirefighterMuscle` palaestrai adapter is
covered by ``test_firefighter_agent.py`` (skipped without palaestrai).
"""

import numpy as np
import pytest

from palaestrai_socal.agents import firefighter_core as fc
from palaestrai_socal.agents.firefighter_core import (
    DEGRADE_WIND_MS, GROUND_WIND_MS, SUPPRESS_PERSIST_STEPS,
    UNBURNED, BURNING, SUPPRESSED,
    wind_efficiency, is_grounded, drops_this_step, line_km_this_step,
    retardant_budget, downwind_offset, fire_head, select_retardant_line,
    age_suppressed,
)


# -- wind efficiency / grounding ------------------------------------------
def test_wind_efficiency_unity_below_degrade():
    assert wind_efficiency(0.0) == 1.0
    assert wind_efficiency(DEGRADE_WIND_MS) == 1.0
    assert wind_efficiency(DEGRADE_WIND_MS - 1.0) == 1.0


def test_wind_efficiency_zero_at_and_above_ground():
    assert wind_efficiency(GROUND_WIND_MS) == 0.0
    assert wind_efficiency(GROUND_WIND_MS + 5.0) == 0.0
    assert wind_efficiency(25.0) == 0.0          # Eaton high-wind baseline


def test_wind_efficiency_linear_ramp_between():
    mid = 0.5 * (DEGRADE_WIND_MS + GROUND_WIND_MS)
    assert wind_efficiency(mid) == pytest.approx(0.5)
    # monotone non-increasing across the ramp
    ws = np.linspace(DEGRADE_WIND_MS, GROUND_WIND_MS, 11)
    eff = [wind_efficiency(w) for w in ws]
    assert all(b <= a + 1e-12 for a, b in zip(eff, eff[1:]))


def test_is_grounded_threshold():
    assert not is_grounded(GROUND_WIND_MS - 0.1)
    assert is_grounded(GROUND_WIND_MS)
    assert is_grounded(GROUND_WIND_MS + 10.0)


def test_drops_zero_when_grounded():
    assert drops_this_step(5, 60.0, 25.0) == 0.0
    assert drops_this_step(5, 60.0, 8.0) > 0.0


def test_line_km_zero_when_grounded():
    assert line_km_this_step(5, 60.0, GROUND_WIND_MS) == 0.0


# -- retardant budget -----------------------------------------------------
def test_budget_scales_linearly_with_n_planes():
    kw = dict(wind_speed=8.0, env_step_min=60.0, cell_size_m=50.0)
    b1 = retardant_budget(1, **kw)
    b2 = retardant_budget(2, **kw)
    b4 = retardant_budget(4, **kw)
    assert b1 > 0
    # floor(k * x) so exact integer multiples hold for these tuned constants
    assert b2 == 2 * b1
    assert b4 == 4 * b1


def test_budget_zero_at_and_above_ground_wind():
    assert retardant_budget(7, GROUND_WIND_MS, 60.0, 50.0) == 0
    assert retardant_budget(7, 25.0, 60.0, 50.0) == 0


def test_budget_zero_for_degenerate_inputs():
    assert retardant_budget(0, 8.0, 60.0, 50.0) == 0
    assert retardant_budget(-3, 8.0, 60.0, 50.0) == 0
    assert retardant_budget(3, 8.0, 60.0, 0.0) == 0


def test_budget_finer_grid_more_cells():
    # the same line length covers MORE cells on a finer (smaller) raster.
    coarse = retardant_budget(3, 8.0, 60.0, 947.0)
    fine = retardant_budget(3, 8.0, 60.0, 50.0)
    assert fine > coarse


# -- downwind geometry ----------------------------------------------------
def test_downwind_offset_blows_toward_opposite():
    # wind FROM the north (0 deg) blows the fire TOWARD the south (+row).
    assert downwind_offset(0.0) == (1, 0)
    # wind FROM the south (180) -> fire moves north (-row).
    assert downwind_offset(180.0) == (-1, 0)
    # wind FROM the west (270) -> fire moves east (+col).
    assert downwind_offset(270.0) == (0, 1)
    # wind FROM the east (90) -> fire moves west (-col).
    assert downwind_offset(90.0) == (0, -1)


def test_fire_head_is_downwind_front():
    S = np.full((5, 5), UNBURNED, dtype=np.int8)
    S[2, 2] = BURNING
    # wind from north -> head cell (2,2) has UNBURNED neighbour to the south.
    heads = fire_head(S, 1, 0)
    assert heads == [(2, 2)]
    # if the downwind neighbour is already burning there is no head there.
    S[3, 2] = BURNING
    assert fire_head(S, 1, 0) == [(3, 2)]


# -- line selection -------------------------------------------------------
def _line_grid():
    S = np.full((9, 9), UNBURNED, dtype=np.int8)
    S[4, 3:6] = BURNING                       # a small horizontal front
    fuel = np.full((9, 9), 3, dtype=np.int16)
    return S, fuel


def test_select_line_is_ahead_of_head():
    S, fuel = _line_grid()
    # wind from north -> retardant should be laid one row SOUTH of the head row.
    cells = select_retardant_line(S, fuel, wind_dir_deg=0.0, budget=10)
    assert cells, "expected a non-empty line"
    assert all(S[r, c] == UNBURNED for (r, c) in cells)
    assert all(r == 5 for (r, c) in cells)    # row just downwind of row-4 head


def test_select_line_respects_budget_cap():
    S, fuel = _line_grid()
    cells = select_retardant_line(S, fuel, wind_dir_deg=0.0, budget=2)
    assert len(cells) <= 2


def test_select_line_empty_when_budget_zero():
    S, fuel = _line_grid()
    assert select_retardant_line(S, fuel, wind_dir_deg=0.0, budget=0) == []


def test_select_line_empty_without_fire():
    S = np.full((9, 9), UNBURNED, dtype=np.int8)
    assert select_retardant_line(S, None, wind_dir_deg=0.0, budget=10) == []


def test_select_line_no_duplicate_cells():
    S, fuel = _line_grid()
    cells = select_retardant_line(S, fuel, wind_dir_deg=0.0, budget=50)
    assert len(cells) == len(set(cells))


def test_select_line_prefers_high_fuel():
    S = np.full((9, 9), UNBURNED, dtype=np.int8)
    S[4, 2:7] = BURNING
    fuel = np.zeros((9, 9), dtype=np.int16)
    fuel[5, 4] = 6                              # one rich-fuel target downwind
    cells = select_retardant_line(S, fuel, wind_dir_deg=0.0, budget=1)
    assert cells == [(5, 4)]


# -- retardant-line ageing ------------------------------------------------
def test_age_suppressed_reverts_after_persist_steps():
    S = np.full((3, 3), UNBURNED, dtype=np.int8)
    S[1, 1] = SUPPRESSED
    A = np.zeros((3, 3), dtype=np.int16)
    for step in range(SUPPRESS_PERSIST_STEPS - 1):
        age_suppressed(S, A, SUPPRESS_PERSIST_STEPS)
        assert S[1, 1] == SUPPRESSED           # still holding
    # the SUPPRESS_PERSIST_STEPS-th ageing reverts it to UNBURNED
    age_suppressed(S, A, SUPPRESS_PERSIST_STEPS)
    assert S[1, 1] == UNBURNED
    assert A[1, 1] == 0


def test_age_suppressed_resets_timer_when_not_suppressed():
    S = np.full((2, 2), UNBURNED, dtype=np.int8)
    A = np.zeros((2, 2), dtype=np.int16)
    S[0, 0] = SUPPRESSED
    age_suppressed(S, A, SUPPRESS_PERSIST_STEPS)
    assert A[0, 0] == 1 and A[1, 1] == 0
    # cell reverts to UNBURNED externally -> timer must reset on next ageing.
    S[0, 0] = UNBURNED
    age_suppressed(S, A, SUPPRESS_PERSIST_STEPS)
    assert A[0, 0] == 0


def test_age_suppressed_leaves_fire_untouched():
    S = np.full((2, 2), UNBURNED, dtype=np.int8)
    S[0, 0] = BURNING
    A = np.zeros((2, 2), dtype=np.int16)
    age_suppressed(S, A, SUPPRESS_PERSIST_STEPS)
    assert S[0, 0] == BURNING
