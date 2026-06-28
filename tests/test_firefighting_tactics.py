"""Tactic-primitive tests (numpy-only) for palaestrai_socal.agents.firefighting.

Each primitive maps a budget to ``(row, col, state, layer)`` edits; these pin
the per-tactic state/layer and the deterministic ordering, plus the no-op
identity (zero budget -> no edits).
"""
import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents.firefighter_core import select_retardant_line
from palaestrai_socal.agents.firefighting import tactics as T

NR, NC = 20, 20
WIND_DIR = 0.0      # advancing "down" (south); downwind offset is deterministic


def _front():
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    S[10, 8:12] = spaces.BURNING
    return S


def _fuel():
    return np.full((NR, NC), 3.0)


# -- indirect_line is the v0.3 retardant line, wrapped --------------------
def test_indirect_line_matches_select_retardant_line():
    S, F = _front(), _fuel()
    budget = 6
    cells = select_retardant_line(S, F, WIND_DIR, budget)
    muts = T.indirect_line(S, F, WIND_DIR, budget)
    assert [(r, c) for (r, c, _s, _l) in muts] == list(cells)
    for (_r, _c, st, lyr) in muts:
        assert st == spaces.SUPPRESSED
        assert lyr == spaces.LAYER_SUPPRESSION


def test_indirect_line_zero_budget_is_noop():
    assert T.indirect_line(_front(), _fuel(), WIND_DIR, 0) == []


# -- containment_line writes CONTAINED on the same geometry ---------------
def test_containment_line_is_contained_state():
    S, F = _front(), _fuel()
    sup = T.indirect_line(S, F, WIND_DIR, 6)
    con = T.containment_line(S, F, WIND_DIR, 6)
    # same cells, different state.
    assert [(r, c) for (r, c, _s, _l) in con] == \
        [(r, c) for (r, c, _s, _l) in sup]
    assert all(st == spaces.CONTAINED for (_r, _c, st, _l) in con)
    assert T.handline is T.containment_line
    assert T.dozer_line is T.containment_line


# -- direct_attack hits burning cells -> SUPPRESSED -----------------------
def test_direct_attack_targets_burning():
    S = _front()
    muts = T.direct_attack(S, budget=2, wind_dir_deg=WIND_DIR)
    assert len(muts) == 2
    for (r, c, st, _l) in muts:
        assert S[r, c] == spaces.BURNING       # targeted an active cell
        assert st == spaces.SUPPRESSED
    assert T.direct_attack(S, budget=0) == []


# -- point_protect hardens highest-value UNBURNED cells -------------------
def test_point_protect_orders_by_value():
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    V = np.zeros((NR, NC))
    V[2, 2] = 5.0
    V[7, 9] = 9.0      # highest value -> protected first
    V[3, 1] = 1.0
    muts = T.point_protect(S, V, budget=2)
    assert [(r, c) for (r, c, _s, _l) in muts] == [(7, 9), (2, 2)]
    assert all(st == spaces.CONTAINED for (_r, _c, st, _l) in muts)


def test_point_protect_skips_burning_and_noops():
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    V = np.zeros((NR, NC))
    V[5, 5] = 4.0
    S[5, 5] = spaces.BURNING            # already burning -> not protectable
    assert T.point_protect(S, V, budget=3) == []
    assert T.point_protect(S, None, budget=3) == []
    assert T.point_protect(S, V, budget=0) == []


# -- burnout ignites UNBURNED upwind of the line --------------------------
def test_burnout_ignites_upwind():
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    line = [(10, 10)]
    muts = T.burnout(S, line, wind_dir_deg=WIND_DIR, budget=4)
    assert len(muts) == 1
    (r, c, st, lyr) = muts[0]
    assert st == spaces.BURNING and lyr == spaces.LAYER_FIRE
    assert T.burnout(S, [], WIND_DIR, 4) == []
    assert T.burnout(S, line, WIND_DIR, 0) == []
