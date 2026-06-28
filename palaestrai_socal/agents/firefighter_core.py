"""Numpy-only firefighter driver behind the :class:`FirefighterMuscle`.

The v0.3 milestone adds the first *responder* agent: a fleet of ``n_planes``
Large Air Tankers (Erickson MD-87 class) that lay long-term fire retardant ahead
of the wildfire head, building a temporary firebreak. As with the wildfire and
damage agents, all the decision logic lives here -- free of any palaestrAI /
pandapower import -- so it is unit-testable with numpy alone
(``tests/test_firefighter_core.py``).

The fleet enters the model through a single operational knob, ``n_planes``;
every other quantity is a fixed, documented constant derived from real aero
tanker data (see ``DESIGN_v0.3_FIREFIGHTER.md`` §2-3). The pipeline is:

    n_planes --(constants + wind)--> retardant_budget (cells this env step)
    cell_state + wind --> fire head (downwind front cells)
    head + fuel + budget --> a contiguous SUPPRESSED line ahead of the head

The muscle turns the selected cells into ``(row, col, SUPPRESSED,
LAYER_SUPPRESSION)`` mutations on the ``gis.cell_mutations`` actuator; the GIS
env applies them with suppression-over-spread priority and ages them back to
``UNBURNED`` after :data:`SUPPRESS_PERSIST_STEPS` steps (retardant breakdown).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# cell-state codes (mirror palaestrai_socal.spaces / wildfire_cma.cma)
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2
SUPPRESSED = 3

# -- operational constants (real aero-tanker data; NOT runtime parameters) ----
# Erickson MD-87 Type-1 LAT productivity; see DESIGN_v0.3_FIREFIGHTER.md §2.
DROPS_PER_PLANE_PER_HOUR = 4.0   # Erickson "4-5 drops/h" -> 4 (conservative)
LINE_KM_PER_DROP = 0.6           # effective retardant line per 3,000-gal drop
GROUND_WIND_MS = 18.0            # ~40 mph sustained -> fleet grounded at/above
DEGRADE_WIND_MS = 13.0           # ~30 mph -> effectiveness ramps down to ground
SUPPRESS_PERSIST_STEPS = 12      # env steps a retardant line holds, then ages


def wind_efficiency(wind_speed: float) -> float:
    """Fraction of nominal productivity retained at ``wind_speed`` (m/s).

    ``1.0`` for calm/moderate wind (``<= DEGRADE_WIND_MS``), a linear ramp down
    to ``0.0`` at ``GROUND_WIND_MS``, and **exactly 0** at or above
    ``GROUND_WIND_MS`` (hard grounding -- the dominant Santa-Ana constraint that
    keeps the Eaton high-wind run unchanged).
    """
    w = float(wind_speed)
    if w <= DEGRADE_WIND_MS:
        return 1.0
    if w >= GROUND_WIND_MS:
        return 0.0
    return (GROUND_WIND_MS - w) / (GROUND_WIND_MS - DEGRADE_WIND_MS)


def is_grounded(wind_speed: float) -> bool:
    """True when sustained wind grounds the fleet (``>= GROUND_WIND_MS``)."""
    return float(wind_speed) >= GROUND_WIND_MS


def drops_this_step(n_planes: int, env_step_min: float,
                    wind_speed: float) -> float:
    """Drops the fleet flies this env step (0 when grounded)."""
    if is_grounded(wind_speed):
        return 0.0
    return float(n_planes) * DROPS_PER_PLANE_PER_HOUR * (float(env_step_min) / 60.0)


def line_km_this_step(n_planes: int, env_step_min: float, wind_speed: float) -> float:
    """Effective retardant line length [km] laid this env step."""
    drops = float(n_planes) * DROPS_PER_PLANE_PER_HOUR * (float(env_step_min) / 60.0)
    return drops * LINE_KM_PER_DROP * wind_efficiency(wind_speed)


def retardant_budget(
    n_planes: int,
    wind_speed: float,
    env_step_min: float,
    cell_size_m: float,
) -> int:
    """Number of cells the fleet can set ``SUPPRESSED`` this env step.

    ``floor(line_km_this_step * cells_per_km)`` with ``cells_per_km =
    1000 / cell_size_m`` (design §3). Scales linearly with ``n_planes`` and is
    exactly 0 at/above :data:`GROUND_WIND_MS`. Returns 0 for degenerate inputs.
    """
    if n_planes <= 0 or cell_size_m <= 0:
        return 0
    line_km = line_km_this_step(n_planes, env_step_min, wind_speed)
    cells_per_km = 1000.0 / float(cell_size_m)
    return int(math.floor(line_km * cells_per_km))


def downwind_offset(wind_dir_deg: float) -> Tuple[int, int]:
    """Discrete ``(drow, dcol)`` step in the direction the fire head advances.

    ``wind_dir_deg`` is the meteorological direction the wind blows *from*; the
    fire (and retardant target) move *toward* ``(dir + 180) % 360``. Bearings
    follow ``wildfire_cma.cma`` (``atan2(dc, -dr)``, clockwise from north), so
    the offset co-registers cell-for-cell with the spread step. Each component
    is clamped to ``{-1, 0, 1}``; a degenerate ``(0, 0)`` defaults to south.
    """
    toward = math.radians((float(wind_dir_deg) + 180.0) % 360.0)
    ddr = -math.cos(toward)   # +row is south
    ddc = math.sin(toward)    # +col is east
    drow = int(max(-1, min(1, round(ddr))))
    dcol = int(max(-1, min(1, round(ddc))))
    if drow == 0 and dcol == 0:
        drow = 1
    return drow, dcol


def fire_head(state: np.ndarray, drow: int, dcol: int) -> List[Tuple[int, int]]:
    """BURNING cells whose downwind neighbour is still UNBURNED (the head).

    These are the front cells where the fire *will* advance next, i.e. where a
    retardant line laid just downwind does the most good.
    """
    S = np.asarray(state)
    nr, nc = S.shape
    heads: List[Tuple[int, int]] = []
    for (r, c) in np.argwhere(S == BURNING):
        tr, tc = int(r) + drow, int(c) + dcol
        if 0 <= tr < nr and 0 <= tc < nc and S[tr, tc] == UNBURNED:
            heads.append((int(r), int(c)))
    return heads


def _adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """8-neighbour adjacency (Moore), excluding identity."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr <= 1 and dc <= 1) and not (dr == 0 and dc == 0)


def select_retardant_line(
    state: np.ndarray,
    fuel: Optional[np.ndarray],
    wind_dir_deg: float,
    budget: int,
) -> List[Tuple[int, int]]:
    """Pick up to ``budget`` UNBURNED cells for a line ahead of the fire head.

    Doctrine (design §4): for each head cell, the candidate is the UNBURNED cell
    one step downwind (where the fire is about to run). Candidates are then laid
    into a **contiguous** line, greedily preferring high ``fuel`` class cells,
    until the budget is exhausted. Fully deterministic given the inputs.

    Returns ``[]`` when ``budget <= 0`` or there is no eligible head, so a
    grounded fleet (budget 0) emits no mutations.
    """
    if budget <= 0:
        return []
    S = np.asarray(state)
    nr, nc = S.shape
    drow, dcol = downwind_offset(wind_dir_deg)
    heads = fire_head(S, drow, dcol)
    if not heads:
        return []

    # candidate target cells: the UNBURNED cell just downwind of each head.
    candidates = set()
    for (r, c) in heads:
        tr, tc = r + drow, c + dcol
        if 0 <= tr < nr and 0 <= tc < nc and S[tr, tc] == UNBURNED:
            candidates.add((tr, tc))
    if not candidates:
        return []

    def _fuel(rc: Tuple[int, int]) -> int:
        return int(fuel[rc]) if fuel is not None else 0

    # priority order: high fuel first, then row/col for determinism.
    ordered = sorted(candidates, key=lambda rc: (-_fuel(rc), rc[0], rc[1]))

    chosen: List[Tuple[int, int]] = []
    chosen_set = set()
    for seed in ordered:
        if len(chosen) >= budget:
            break
        if seed in chosen_set:
            continue
        # grow a connected run from the seed across adjacent candidates.
        frontier = [seed]
        while frontier and len(chosen) < budget:
            cell = frontier.pop(0)
            if cell in chosen_set:
                continue
            chosen.append(cell)
            chosen_set.add(cell)
            nbrs = sorted(
                (rc for rc in ordered
                 if rc not in chosen_set and _adjacent(rc, cell)),
                key=lambda rc: (-_fuel(rc), rc[0], rc[1]),
            )
            frontier.extend(nbrs)
    return chosen[:budget]


def age_suppressed(
    state: np.ndarray,
    suppress_age: np.ndarray,
    persist_steps: int = SUPPRESS_PERSIST_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the SUPPRESSED-line persistence timer by one env step (in place).

    Retardant is long-term but not permanent: a cell that has been SUPPRESSED
    for ``persist_steps`` env steps reverts to ``UNBURNED`` (re-burnable),
    modelling retardant breakdown / burn-through. Cells that are not currently
    SUPPRESSED have their timer reset to 0, so the count is "consecutive steps
    held". Mutates and returns ``(state, suppress_age)``.
    """
    S = state
    A = suppress_age
    is_supp = (S == SUPPRESSED)
    A[~is_supp] = 0
    A[is_supp] += 1
    revert = is_supp & (A >= int(persist_steps))
    S[revert] = UNBURNED
    A[revert] = 0
    return S, A
