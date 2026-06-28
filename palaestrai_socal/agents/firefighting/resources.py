"""Firefighting resources -- one dataclass per resource type (v0.4).

Each resource owns a single operational knob (its *count*) and a documented set
of fixed productivity / constraint constants sourced to representative wildland
firefighting data. Its :meth:`capacity` returns how many GIS cells the resource
can convert this env step -- the multi-resource analogue of v0.3's
:func:`palaestrai_socal.agents.firefighter_core.retardant_budget`.

Capacity is a *count*; **which** cells get worked (and the slope/road/fuel
filtering) is the planner/tactics' job (:mod:`.planner`, :mod:`.tactics`). The
only spatial input capacity itself takes is a representative slope (so ground
crews/dozers degrade on steep terrain) and wind (so aircraft ground out).

All numbers below are module constants -- NOT runtime parameters (DESIGN §8:
"constants over parameters"). Only the resource *counts* are knobs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from palaestrai_socal import spaces
from palaestrai_socal.agents.firefighter_core import (
    GROUND_WIND_MS,
    retardant_budget,
    wind_efficiency,
)

# ===========================================================================
# Air tankers -- v0.3 baseline (delegates to firefighter_core for exactness)
# ===========================================================================
@dataclass(frozen=True)
class TankerFleet:
    """Large Air Tankers laying long-term retardant line (the v0.3 resource).

    Capacity delegates verbatim to :func:`firefighter_core.retardant_budget`, so
    a tankers-only + indirect plan reproduces v0.3 cell-for-cell.
    """

    n: int = 0
    tactic: str = "indirect"
    state: int = spaces.SUPPRESSED
    layer: int = spaces.LAYER_SUPPRESSION

    def grounded(self, wind_speed: float) -> bool:
        return float(wind_speed) >= GROUND_WIND_MS

    def capacity(self, wind_speed: float, step_min: float, cell_m: float,
                 slope_deg: float = 0.0) -> int:
        return retardant_budget(self.n, wind_speed, step_min, cell_m)


# ===========================================================================
# Helicopters -- bucket water/foam, direct attack on the fire edge
# ===========================================================================
# Type-1 helo (e.g. CH-47 / S-64) productivity. Helos cycle faster than LATs
# (they dip from nearby water) but each drop covers a shorter effective line.
HELO_DROPS_PER_HOUR = 8.0          # ~8 turnarounds/h on a near water source
HELO_LINE_KM_PER_DROP = 0.25       # ~2,000-gal water/foam drop, short hold
# Helos tolerate more wind than fixed-wing LATs before grounding (lower stall /
# can fly nap-of-the-earth), but still ground in extreme Santa-Ana gusts.
HELO_GROUND_WIND_MS = 22.0         # ~49 mph sustained
HELO_DEGRADE_WIND_MS = 16.0        # effectiveness ramps down above this


def helo_wind_efficiency(wind_speed: float) -> float:
    """Helo productivity fraction; 1.0 below degrade, 0 at/above ground wind."""
    w = float(wind_speed)
    if w <= HELO_DEGRADE_WIND_MS:
        return 1.0
    if w >= HELO_GROUND_WIND_MS:
        return 0.0
    return (HELO_GROUND_WIND_MS - w) / (HELO_GROUND_WIND_MS - HELO_DEGRADE_WIND_MS)


@dataclass(frozen=True)
class HeloFleet:
    """Helicopters doing direct attack (water/foam) on burning cells."""

    n: int = 0
    tactic: str = "direct"
    state: int = spaces.SUPPRESSED     # wetline -- short hold, ages like retardant
    layer: int = spaces.LAYER_SUPPRESSION

    def grounded(self, wind_speed: float) -> bool:
        return float(wind_speed) >= HELO_GROUND_WIND_MS

    def capacity(self, wind_speed: float, step_min: float, cell_m: float,
                 slope_deg: float = 0.0) -> int:
        if self.n <= 0 or cell_m <= 0:
            return 0
        drops = self.n * HELO_DROPS_PER_HOUR * (float(step_min) / 60.0)
        line_km = drops * HELO_LINE_KM_PER_DROP * helo_wind_efficiency(wind_speed)
        return int(math.floor(line_km * (1000.0 / float(cell_m))))


# ===========================================================================
# Hand crews -- handline (mineral-soil break). NOT wind-grounded; slope-limited
# ===========================================================================
# A Type-1 Interagency Hotshot Crew builds line slowly but is unaffected by the
# wind that grounds aircraft -- the key complementarity the doctrine exploits.
CREW_LINE_M_PER_HOUR = 90.0        # ~3 chains/h in chaparral (1 chain = 20.1 m)
CREW_SLOPE_CUTOFF_DEG = 45.0       # too steep to build line safely above this
CREW_SLOPE_HALF_DEG = 30.0         # productivity halves by this slope


def _slope_derate(slope_deg: float, half_deg: float, cutoff_deg: float) -> float:
    """Linear-ish slope productivity factor: 1 at flat, 0 at/above cutoff."""
    s = abs(float(slope_deg))
    if s >= cutoff_deg:
        return 0.0
    if s <= 0.0:
        return 1.0
    # 1.0 at 0 deg, 0.5 at half_deg, 0.0 at cutoff (piecewise linear).
    if s <= half_deg:
        return 1.0 - 0.5 * (s / half_deg)
    return 0.5 * (cutoff_deg - s) / max(cutoff_deg - half_deg, 1e-9)


@dataclass(frozen=True)
class HandCrews:
    """Hand crews building permanent CONTAINED handline; wind-independent."""

    n: int = 0
    tactic: str = "ground"
    state: int = spaces.CONTAINED
    layer: int = spaces.LAYER_SUPPRESSION

    def grounded(self, wind_speed: float) -> bool:
        return False                    # crews work in any wind

    def capacity(self, wind_speed: float, step_min: float, cell_m: float,
                 slope_deg: float = 0.0) -> int:
        if self.n <= 0 or cell_m <= 0:
            return 0
        derate = _slope_derate(slope_deg, CREW_SLOPE_HALF_DEG, CREW_SLOPE_CUTOFF_DEG)
        line_m = self.n * CREW_LINE_M_PER_HOUR * (float(step_min) / 60.0) * derate
        return int(math.floor(line_m / float(cell_m)))


# ===========================================================================
# Dozers -- wide containment line; medium speed; slope/rock-limited
# ===========================================================================
DOZER_LINE_M_PER_HOUR = 400.0      # tractor-plow line rate in moderate terrain
DOZER_SLOPE_CUTOFF_DEG = 35.0      # can't safely cut line on steeper slopes
DOZER_SLOPE_HALF_DEG = 20.0


@dataclass(frozen=True)
class Dozers:
    """Dozers cutting wide CONTAINED line; faster than crews, slope-limited."""

    n: int = 0
    tactic: str = "ground"
    state: int = spaces.CONTAINED
    layer: int = spaces.LAYER_SUPPRESSION

    def grounded(self, wind_speed: float) -> bool:
        return False

    def capacity(self, wind_speed: float, step_min: float, cell_m: float,
                 slope_deg: float = 0.0) -> int:
        if self.n <= 0 or cell_m <= 0:
            return 0
        derate = _slope_derate(slope_deg, DOZER_SLOPE_HALF_DEG,
                               DOZER_SLOPE_CUTOFF_DEG)
        line_m = self.n * DOZER_LINE_M_PER_HOUR * (float(step_min) / 60.0) * derate
        return int(math.floor(line_m / float(cell_m)))


# ===========================================================================
# Engines -- structure / point protection; road-access cells only
# ===========================================================================
# Engines wrap/foam a small number of high-value asset cells per step; their
# "budget" is a count of protectable points, not a line length.
ENGINE_ASSETS_PER_HOUR = 1.5       # points an engine can secure per hour


@dataclass(frozen=True)
class Engines:
    """Type-3 engines protecting grid-asset cells (point protection)."""

    n: int = 0
    tactic: str = "protect"
    state: int = spaces.CONTAINED      # protected asset is non-ignitable
    layer: int = spaces.LAYER_SUPPRESSION

    def grounded(self, wind_speed: float) -> bool:
        return False

    def capacity(self, wind_speed: float, step_min: float, cell_m: float,
                 slope_deg: float = 0.0) -> int:
        if self.n <= 0:
            return 0
        return int(math.floor(self.n * ENGINE_ASSETS_PER_HOUR *
                              (float(step_min) / 60.0)))


def build_resources(
    n_planes: int = 0,
    n_helos: int = 0,
    n_crews: int = 0,
    n_dozers: int = 0,
    n_engines: int = 0,
) -> List[object]:
    """Construct the fleet mix from operational counts (the only knobs).

    Returns the resources in a fixed deterministic order
    (tankers, helos, crews, dozers, engines). Zero-count resources are still
    included; their capacity is 0 so they contribute no mutations.
    """
    return [
        TankerFleet(n=int(n_planes)),
        HeloFleet(n=int(n_helos)),
        HandCrews(n=int(n_crews)),
        Dozers(n=int(n_dozers)),
        Engines(n=int(n_engines)),
    ]
