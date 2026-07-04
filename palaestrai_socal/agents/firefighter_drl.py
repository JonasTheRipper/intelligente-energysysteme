"""Shared spec + feature extraction for the v0.7 Deep-RL firefighter.

This module is the single source of truth for the DRL firefighter's
observation/action contract (confirmed 2026-07-04). It is imported by BOTH the
offline teacher-transition harvester
(:mod:`palaestrai_socal.agents.harvest_teacher_transitions`) and the online
learning muscle (:mod:`palaestrai_socal.agents.firefighter_drl_agent`), so the
17-dim observation the CQL offline buffer is bootstrapped from is *bit-for-bit*
the same vector the muscle presents to the hARL SAC brain at inference time.

Observation (``OBS_DIM == 17``)
-------------------------------
A compact, grid-size-invariant summary of the coupled fire + power-grid state::

     0  burning_frac        (# BURNING cells)      / n_cells
     1  burned_frac         (# BURNED_OUT cells)   / n_cells
     2  suppressed_frac      (# SUPPRESSED cells)   / n_cells
     3  contained_frac       (# CONTAINED cells)    / n_cells
     4  front_size_norm      (# active fire-head cells) / n_cells
     5  mean_front_fuel/4    mean fuel class on the burning front, /4 (chaparral)
     6  wind_speed/20        m/s normalised by ~72 km/h
     7  sin(wind_dir)        wind bearing, radians
     8  cos(wind_dir)
     9  mean_slope_deg/45    representative terrain slope, /45 deg
    10  served_mw/base       served load this step / baseline served load
    11  saidi_norm           cumulative SAIDI / saidi_scale
    12  dSAIDI_last_step     step change in SAIDI / saidi_scale  (>=0)
    13  tankers_avail        1.0 if the tanker fleet can fly this step, else 0
    14  ground_crews_avail   1.0 if hand crews / dozers are available, else 0
    15  engines_avail        1.0 if engines are available, else 0
    16  step/max_steps       episode progress in [0, 1]

All features are float32 and (softly) bounded to [-1, 1] / [0, 1].

Action (``N_TACTICS == 4``; ``Discrete(4)``)
--------------------------------------------
The DRL muscle picks a *doctrine* each env step; the deterministic
:class:`~palaestrai_socal.agents.firefighting.planner.IncidentCommand` then
spends the configured fleet on that doctrine (identical machinery to the
scripted teacher, so tactics stay physically consistent)::

    0  no-op                 hold resources (bit-for-bit v0.2 baseline)
    1  indirect line         lay retardant / handline ahead of the front (v0.3)
    2  direct attack         water/foam on the burning edge
    3  triage / point-protect protect grid-critical asset cells

Reward
------
``reward = -delta_saidi / SAIDI_SCALE`` (<= 0): each step the agent is charged
the SAIDI accrued that step, so minimising cumulative SAIDI == maximising
return. See :mod:`palaestrai_socal.agents.saidi_objective`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents import firefighter_core as _core

# ---- the contract (confirmed 2026-07-04) --------------------------------
OBS_DIM = 17
N_TACTICS = 4

# action ids -> the IncidentCommand doctrine + resource gate the muscle applies
ACT_NOOP = 0
ACT_INDIRECT = 1
ACT_DIRECT = 2
ACT_TRIAGE = 3

# normalisation constants (documented above)
WIND_SCALE = 20.0
SLOPE_SCALE = 45.0
FUEL_SCALE = 4.0
SAIDI_SCALE = 60.0
BASE_SERVED_MW = 1.0
# planning figure that turns served MW into customers for the SAIDI maths;
# identical to the value the environment/reducer and SaidiObjective use.
CUSTOMERS_PER_MW = 200.0


def _frac(state: np.ndarray, code: int) -> float:
    n = state.size
    return float((state == code).sum()) / n if n else 0.0


def _front_cells(state: np.ndarray) -> int:
    """Count BURNING cells that border a non-burning, burnable neighbour.

    A coarse "active fire head" proxy: the perimeter of the burning region,
    which is where suppression tactics act. 4-connected.
    """
    burning = state == spaces.BURNING
    if not burning.any():
        return 0
    # a burning cell is on the front if any 4-neighbour is not burning
    pad = np.pad(burning, 1, mode="constant", constant_values=True)
    interior = (
        pad[1:-1, 2:] & pad[1:-1, :-2] & pad[2:, 1:-1] & pad[:-2, 1:-1]
    )
    front = burning & ~interior
    return int(front.sum())


def _mean_front_fuel(state: np.ndarray, fuel: Optional[np.ndarray]) -> float:
    if fuel is None:
        return 3.0  # chaparral default, matches IncidentCommand
    F = np.asarray(fuel, dtype=float)
    burning = state == spaces.BURNING
    if not burning.any():
        return float(F.mean()) if F.size else 3.0
    return float(F[burning].mean())


def mean_slope_deg(
    dem: Optional[np.ndarray], cell_size_m: Optional[float]
) -> float:
    """Representative slope [deg] from a DEM (0 if unavailable)."""
    if dem is None or not cell_size_m:
        return 0.0
    D = np.asarray(dem, dtype=float)
    if D.ndim != 2 or D.size == 0:
        return 0.0
    gy, gx = np.gradient(D, float(cell_size_m))
    grade = np.sqrt(gy * gy + gx * gx)
    return float(np.degrees(np.arctan(np.nanmean(grade))))


def extract_obs(
    *,
    state: np.ndarray,
    fuel: Optional[np.ndarray],
    dem: Optional[np.ndarray],
    cell_size_m: Optional[float],
    wind_speed: float,
    wind_dir_deg: float,
    served_mw: float,
    base_served_mw: float,
    saidi: float,
    prev_saidi: float,
    tankers_avail: bool,
    ground_crews_avail: bool,
    engines_avail: bool,
    step: int,
    max_steps: int,
    saidi_scale: float = SAIDI_SCALE,
) -> np.ndarray:
    """Build the 17-dim float32 observation from a decoded env step.

    ``state`` is the ``gis.cell_state`` grid (2-D int array). All other inputs
    are scalars/rasters already decoded from the same step's sensor readings.
    The exact same call is made by the harvester (from stored world_states) and
    by the live muscle (from its sensor list), guaranteeing train/serve parity.
    """
    S = np.asarray(state)
    n_cells = max(1, S.size)
    front = _front_cells(S)
    front_fuel = _mean_front_fuel(S, fuel)
    slope = mean_slope_deg(dem, cell_size_m)

    base = base_served_mw if base_served_mw > 0 else 1.0
    served_ratio = float(np.clip(served_mw / base, 0.0, 2.0))
    saidi_norm = float(saidi) / saidi_scale if saidi_scale else 0.0
    dsaidi = max(0.0, float(saidi) - float(prev_saidi)) / (
        saidi_scale if saidi_scale else 1.0
    )
    wind_rad = math.radians(float(wind_dir_deg))

    obs = np.array(
        [
            _frac(S, spaces.BURNING),
            _frac(S, spaces.BURNED_OUT),
            _frac(S, spaces.SUPPRESSED),
            _frac(S, spaces.CONTAINED),
            float(front) / n_cells,
            float(np.clip(front_fuel / FUEL_SCALE, 0.0, 1.0)),
            float(np.clip(float(wind_speed) / WIND_SCALE, 0.0, 1.0)),
            math.sin(wind_rad),
            math.cos(wind_rad),
            float(np.clip(slope / SLOPE_SCALE, 0.0, 1.0)),
            served_ratio,
            float(np.clip(saidi_norm, 0.0, 1.0)),
            float(np.clip(dsaidi, 0.0, 1.0)),
            1.0 if tankers_avail else 0.0,
            1.0 if ground_crews_avail else 0.0,
            1.0 if engines_avail else 0.0,
            float(np.clip(step / max(1, max_steps), 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
    return obs


def teacher_action_from_mutations(
    muts: List[Tuple[int, int, int, int]],
) -> int:
    """Infer the DRL doctrine id (0..3) the scripted teacher effectively took.

    The scripted firefighter emits ``(row, col, state, layer)`` cell edits.
    We map the *dominant* edit back onto the DRL's 4-way doctrine so the
    harvested transitions carry a Discrete(4) action label:

    * no edits                          -> ACT_NOOP (0)
    * any CONTAINED edit present         -> ACT_TRIAGE (3)   (point-protect/ground)
    * SUPPRESSED edits, mostly on burning cells -> ACT_DIRECT (2)
    * SUPPRESSED edits ahead of the front       -> ACT_INDIRECT (1)

    The teacher does not expose which tactic produced each edit, so this is a
    best-effort, deterministic reconstruction sufficient for CQL bootstrapping.
    """
    if not muts:
        return ACT_NOOP
    states = [int(m[2]) for m in muts]
    if any(s == spaces.CONTAINED for s in states):
        return ACT_TRIAGE
    n_supp = sum(1 for s in states if s == spaces.SUPPRESSED)
    if n_supp == 0:
        return ACT_NOOP
    # heuristic split: retardant lines dominate the tanker/indirect teacher;
    # direct attack is rarer. Default to indirect (the v0.3 doctrine).
    return ACT_INDIRECT


def resource_availability(
    *,
    n_planes: int,
    n_helos: int,
    n_crews: int,
    n_dozers: int,
    n_engines: int,
    wind_speed: float,
) -> Dict[str, bool]:
    """Resource-availability flags used by obs features 13-15."""
    grounded = _core.is_grounded(float(wind_speed))
    tankers = (n_planes + n_helos) > 0 and not grounded
    ground = (n_crews + n_dozers) > 0  # crews/dozers are not wind-grounded
    engines = n_engines > 0
    return {
        "tankers_avail": bool(tankers),
        "ground_crews_avail": bool(ground),
        "engines_avail": bool(engines),
    }
