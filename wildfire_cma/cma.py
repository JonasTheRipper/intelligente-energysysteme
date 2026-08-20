"""GUARDIAN Wildfire Constrained-Mutation Automaton (CMA).

This implements the wildfire cellular automaton described in the GUARDIAN
paper as a **Constrained Mutation Automaton** -- a four-tuple

    CMA = (S, tau, D, Theta)

that mutates the topology of a power grid ``G`` as a wildfire spreads across a
co-registered raster stack ``R``.

* ``S``      cellular state grid, co-registered with the raster stack ``R`` at
             resolution ``delta`` (metres). Cell states for wildfire:
             ``0 = unburned``, ``1 = burning``, ``2 = burned-out``.
* ``tau``    transition function ``S x R x Theta -> S`` -- the CA update using
             the Rothermel-style rate-of-spread (eq. 6) and the ignition
             probability (eq. 7).
* ``D``      damage mapper ``S x G -> dG`` -- fails grid nodes whose cell is
             burning/burned and overhead lines whose footprint is within the
             radiant-heat clearance buffer of the fire front.
* ``Theta``  the Overseer-Adversary parameter vector: ignition point(s),
             wind vector ``u``, dead-fuel-moisture offset, and a global ROS
             multiplier ``kappa``.

Rate of spread (eq. 6):
    R_{ij->i'j'} = R^0_{ij} * phi_w(u, theta) * phi_s(z, theta)
where ``R^0_{ij}`` is the no-wind/no-slope ROS from the LANDFIRE fuel class and
fuel moisture, ``phi_w`` the wind factor and ``phi_s`` the slope factor from the
DEM gradient. Corner (diagonal) cells are scaled by ``sqrt(2) * delta``.

Spread probability (eq. 7):
    p = 1 - exp(-R_{ij->i'j'} * dt_CMA / delta)
A burning cell becomes burned-out after ``T_burn`` steps (fuel heat content).

The implementation is dependency-light (numpy only for the core) so it is fast
and unit-testable in CI. Raster inputs (DEM, fuel, optional canopy) are plain
numpy arrays; :mod:`wildfire_cma.gis` builds them from California GIS data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- cell states ----------------------------------------------------------
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2
# SUPPRESSED (3) is written by the v0.3 FirefighterAgent (retardant line). The
# spread step treats it as a non-ignitable firebreak. Defined here so the core
# stays import-light; mirrors palaestrai_socal.spaces.SUPPRESSED.
SUPPRESSED = 3
# CONTAINED (5) is a completed v0.4 ground containment line (handline / dozer)
# or a point-protected grid asset: like SUPPRESSED it is non-ignitable, but it
# does not age out within the episode. Mirrors palaestrai_socal.spaces.CONTAINED.
CONTAINED = 5

# Built-up ("houses") fuel class. Burnable on purpose: the January-2025 fires
# destroyed ~9,400 (Eaton) and ~6,800 (Palisades) structures, so modelling
# settlement as non-burnable class 0 is exactly wrong. Named here, beside the
# ROS table it keys into, so every fuel builder and every consumer that needs
# the house mask shares one definition instead of a literal 9.
HOUSE_FUEL_CLASS = 9

# --- Anderson/Scott-Burgan style no-wind/no-slope base ROS [m/min] --------
# Keyed by a coarse fuel class id. These are representative baseline rates of
# spread for the no-wind, no-slope condition; the wind/slope factors and the
# Overseer kappa multiplier scale them up to the extreme (Santa-Ana) regime.
BASE_ROS_BY_FUEL: Dict[int, float] = {
    0: 0.0,    # non-burnable (water, urban, barren, agriculture)
    1: 4.0,    # grass / GR (fast)
    2: 2.5,    # grass-shrub / GS
    3: 3.2,    # shrub / chaparral SH (the dominant SoCal wildfire fuel)
    4: 1.2,    # timber-understory TU
    5: 0.8,    # timber-litter TL
    6: 1.5,    # slash-blowdown SB
    9: 1.0,    # developed houses
}

# fuel moisture of extinction by class (fraction); above this no spread
FUEL_MX_EXTINCTION: Dict[int, float] = {
    0: 0.0, 1: 0.15, 2: 0.20, 3: 0.30, 4: 0.30, 5: 0.30, 6: 0.25, 9: 0.20,
}


@dataclass
class Theta:
    """Overseer-Adversary parameter vector controlling the wildfire CMA.

    ``ignition_points`` are GEOGRAPHIC ``(lon, lat)`` coordinates in EPSG:4326.
    This is the natural coordinate reference for the Overseer-Adversary (and
    for the palaestrAI actuator vector): it is independent of the raster
    resolution / extent and co-registers directly with the grid's bus/line
    geometry. The CMA converts each ``(lon, lat)`` to a raster ``(row, col)``
    via ``RasterStack.lonlat_to_rc`` at ignition time.

    For tests / low-level use that need to pin an exact raster cell, pass
    ``ignition_rc`` ``(row, col)`` integer pairs instead.
    """

    ignition_points: List[Tuple[float, float]] = field(default_factory=list)  # (lon, lat)
    ignition_rc: List[Tuple[int, int]] = field(default_factory=list)          # (row, col)
    wind_speed: float = 10.0       # m/s   (NOAA-sourced when driven by the env)
    wind_dir_deg: float = 45.0     # meteorological direction wind blows FROM
    dead_fuel_moisture: float = 0.06   # fraction (Santa-Ana ~ 0.03-0.08)
    kappa: float = 1.0             # global ROS multiplier (>= 1)

    def clamp(self) -> "Theta":
        """Geophysical-plausibility filter: keep parameters in valid bounds."""
        self.wind_speed = float(np.clip(self.wind_speed, 0.0, 60.0))
        self.wind_dir_deg = float(self.wind_dir_deg % 360.0)
        self.dead_fuel_moisture = float(np.clip(self.dead_fuel_moisture, 0.01, 0.40))
        self.kappa = float(np.clip(self.kappa, 1.0, 8.0))
        return self


@dataclass
class RasterStack:
    """Co-registered raster inputs for the CMA.

    All arrays share shape ``(nrows, ncols)``. ``transform`` maps array
    (row, col) -> (lon, lat) for grid co-registration. ``delta_m`` is the cell
    size in metres.

    ``source`` records WHICH builder produced the stack (``"srtm_gl3"``,
    ``"synthetic"``, ``"rasters"``). The real-DEM mosaic is git-ignored, so a
    run on a machine without it silently gets a different fuel map -- and a
    different set of class-9 house cells. Carrying the provenance on the stack
    lets the environment publish it into ``static_world_model``, turning that
    from an invisible difference into a recorded one.
    """

    fuel: np.ndarray          # int fuel-class id per cell
    dem: np.ndarray           # elevation [m]
    delta_m: float            # cell size [m]
    bounds: Tuple[float, float, float, float]  # (minlon, minlat, maxlon, maxlat)
    canopy: Optional[np.ndarray] = None
    source: str = "unknown"   # which builder made this stack (provenance)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.fuel.shape

    def lonlat_to_rc(self, lon: float, lat: float) -> Tuple[int, int]:
        minlon, minlat, maxlon, maxlat = self.bounds
        nrows, ncols = self.shape
        col = int((lon - minlon) / (maxlon - minlon) * (ncols - 1))
        # row 0 is the north edge (max lat)
        row = int((maxlat - lat) / (maxlat - minlat) * (nrows - 1))
        col = int(np.clip(col, 0, ncols - 1))
        row = int(np.clip(row, 0, nrows - 1))
        return row, col

    def rc_to_lonlat(self, row: int, col: int) -> Tuple[float, float]:
        minlon, minlat, maxlon, maxlat = self.bounds
        nrows, ncols = self.shape
        lon = minlon + col / max(ncols - 1, 1) * (maxlon - minlon)
        lat = maxlat - row / max(nrows - 1, 1) * (maxlat - minlat)
        return lon, lat


# 8-neighbour Moore offsets and whether each is a diagonal (corner) move
_MOORE = [
    (-1, -1, True), (-1, 0, False), (-1, 1, True),
    (0, -1, False),                 (0, 1, False),
    (1, -1, True),  (1, 0, False),  (1, 1, True),
]


class WildfireCMA:
    """Wildfire Constrained-Mutation Automaton (S, tau, D, Theta).

    Parameters
    ----------
    raster:
        The co-registered raster stack (fuel, DEM, ...).
    theta:
        Overseer-Adversary parameter vector.
    dt_cma_min:
        CMA sub-timestep in minutes (eq. 7 ``dt_CMA``). May be smaller than the
        RL timestep for numerical stability of the spread probability.
    t_burn_steps:
        Number of CMA steps a cell stays ``BURNING`` before ``BURNED_OUT``
        (proxy for fuel heat content).
    seed:
        RNG seed for the stochastic ignition draws (eq. 7).
    """

    def __init__(
        self,
        raster: RasterStack,
        theta: Optional[Theta] = None,
        dt_cma_min: float = 5.0,
        t_burn_steps: int = 6,
        seed: int = 0,
    ):
        self.raster = raster
        self.theta = (theta or Theta()).clamp()
        self.dt_cma_min = float(dt_cma_min)
        self.t_burn_steps = int(t_burn_steps)
        self.rng = np.random.default_rng(seed)

        nrows, ncols = raster.shape
        self.state = np.full((nrows, ncols), UNBURNED, dtype=np.int8)
        self.burn_timer = np.zeros((nrows, ncols), dtype=np.int16)
        self.step_count = 0
        self._wind_field = None  # optional per-cell wind: (nrows, ncols, 2) = [speed m/s, dir_deg]

        # precompute slope (rise/run) magnitude and aspect from the DEM
        gy, gx = np.gradient(raster.dem.astype(float), raster.delta_m)
        self._slope = np.hypot(gx, gy)          # tan(slope)
        self._aspect = np.arctan2(-gy, gx)      # uphill direction (rad)

        # Θ ignition points are geographic (lon, lat); convert to raster cells.
        rc_points = [self.raster.lonlat_to_rc(lon, lat)
                     for (lon, lat) in self.theta.ignition_points]
        # plus any explicit (row, col) cells (test / low-level use)
        rc_points += list(self.theta.ignition_rc)
        self._ignite(rc_points)

    def set_wind_field(self, wind_field) -> None:
        """Optional per-cell wind [speed, from-dir-deg]; None => scalar theta wind.

        When set, _phi_wind reads per-cell speed/dir; the scalar theta path is the
        fallback so the no-field behaviour is bit-for-bit identical (all tests preserved).
        """
        if wind_field is None:
            self._wind_field = None
            return
        wf = np.asarray(wind_field, dtype=float)
        assert wf.shape == (self.raster.shape[0], self.raster.shape[1], 2), wf.shape
        self._wind_field = wf

    # -- helpers -----------------------------------------------------------
    def _burnable(self, row: int, col: int) -> bool:
        f = int(self.raster.fuel[row, col])
        return BASE_ROS_BY_FUEL.get(f, 0.0) > 0.0

    def _ignite(self, points: Sequence[Tuple[int, int]]) -> None:
        nrows, ncols = self.raster.shape
        for (r, c) in points:
            r = int(np.clip(r, 0, nrows - 1))
            c = int(np.clip(c, 0, ncols - 1))
            if self._burnable(r, c):
                self.state[r, c] = BURNING
                self.burn_timer[r, c] = 0

    def ignite_lonlat(self, lon: float, lat: float) -> Tuple[int, int]:
        rc = self.raster.lonlat_to_rc(lon, lat)
        self._ignite([rc])
        return rc

    # -- physics: eq. 6 rate of spread -------------------------------------
    def _phi_wind(self, dr: int, dc: int, row=None, col=None) -> float:
        """Wind factor for spread into direction (dr, dc).

        Aligns the wind vector (blowing toward dir+180) with the spread
        direction; an exponential midflame wind speed factor (Rothermel-style).
        When _wind_field is set and row/col are provided, reads per-cell values;
        otherwise falls back to scalar theta (bit-for-bit identical to pre-v0.5).
        """
        if self._wind_field is not None and row is not None:
            u = float(self._wind_field[row, col, 0])
            wdir = float(self._wind_field[row, col, 1])
        else:
            u = self.theta.wind_speed
            wdir = self.theta.wind_dir_deg
        # direction the wind blows TOWARD (meteorological 'from' + 180)
        toward = math.radians((wdir + 180.0) % 360.0)
        # spread bearing: dc -> east(+x), dr -> south(+y). bearing from north.
        spread_bearing = math.atan2(dc, -dr)
        # angle between wind-toward and spread direction
        cos_align = math.cos(toward - spread_bearing)
        # midflame wind coefficient; tuned so extreme Santa-Ana (~20 m/s)
        # gives the 60-80 m/min fronts reported for the Jan-2025 fires.
        c = 0.25
        return float(math.exp(c * u * max(cos_align, -0.5)))

    def _phi_slope(self, row: int, col: int, dr: int, dc: int) -> float:
        """Slope factor for spread into (dr, dc): faster uphill."""
        tan_slope = float(self._slope[row, col])
        # spread direction unit vector in map space (x east, y north)
        sx, sy = dc, -dr
        norm = math.hypot(sx, sy) or 1.0
        sx, sy = sx / norm, sy / norm
        # uphill unit vector from aspect
        ax, ay = math.cos(self._aspect[row, col]), math.sin(self._aspect[row, col])
        upslope_align = max(0.0, sx * ax + sy * ay)
        # Rothermel slope factor phi_s = 5.275 * (tan slope)^2 (bounded)
        phi_s = 5.275 * (min(tan_slope, 1.0) ** 2) * upslope_align
        return 1.0 + phi_s

    def _ros_base(self, row: int, col: int) -> float:
        """No-wind/no-slope ROS [m/min] adjusted for fuel moisture (eq. 6 R^0)."""
        f = int(self.raster.fuel[row, col])
        r0 = BASE_ROS_BY_FUEL.get(f, 0.0)
        if r0 <= 0:
            return 0.0
        mx = FUEL_MX_EXTINCTION.get(f, 0.3)
        m = self.theta.dead_fuel_moisture
        if m >= mx:
            return 0.0
        # moisture damping coefficient (Rothermel): drops to 0 at extinction
        rm = m / mx
        eta_m = max(0.0, 1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
        return r0 * eta_m

    def ros(self, row: int, col: int, dr: int, dc: int) -> float:
        """Full eq. 6 ROS [m/min] for spread from (row,col) into (dr,dc)."""
        r0 = self._ros_base(row, col)
        if r0 <= 0:
            return 0.0
        return self.theta.kappa * r0 * self._phi_wind(dr, dc, row, col) * self._phi_slope(row, col, dr, dc)

    def spread_prob(self, row: int, col: int, dr: int, dc: int, diagonal: bool) -> float:
        """Eq. 7 spread probability for one CMA sub-step."""
        r = self.ros(row, col, dr, dc)        # m/min
        if r <= 0:
            return 0.0
        dist = self.raster.delta_m * (math.sqrt(2.0) if diagonal else 1.0)
        return 1.0 - math.exp(-(r * self.dt_cma_min) / dist)

    # -- transition function tau -------------------------------------------
    def _transition(
        self, state: np.ndarray, burn_timer: np.ndarray
    ) -> int:
        """Pure-ish tau: advance one CMA sub-step on the *given* arrays.

        Mutates ``state`` and ``burn_timer`` in place (so it works on the CMA's
        own ``self.state`` *or* on an externally injected grid, e.g. the
        :class:`WildfireCmaAgent` reconstructing ``S`` from the
        ``gis.cell_state`` sensor) and returns the number of new ignitions.
        The wind/slope/fuel physics (``self.theta``, ``self.raster``,
        ``self.rng``) are unchanged -- only the state buffers are parameterised.
        """
        nrows, ncols = self.raster.shape
        burning = np.argwhere(state == BURNING)
        new_ignitions: List[Tuple[int, int]] = []

        for (r, c) in burning:
            for (dr, dc, diag) in _MOORE:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                # Only UNBURNED neighbours can ignite. This already excludes
                # SUPPRESSED (3) cells, so a firefighter's retardant line is a
                # non-ignitable firebreak (v0.3). The explicit SUPPRESSED guard
                # below is documentary -- it is reached only when SUPPRESSED
                # cells exist, so with none present the spread is bit-for-bit
                # identical to v0.2 (the neighbour is UNBURNED either way and no
                # extra RNG is drawn). See tests/test_suppression_block.py.
                if state[nr, nc] == SUPPRESSED:
                    continue
                # CONTAINED ground lines / point-protected assets (v0.4) are
                # likewise non-ignitable; the guard mirrors SUPPRESSED and, with
                # no CONTAINED cell present, is never taken (UNBURNED check below
                # already excludes it), so the no-suppression spread stays
                # bit-for-bit identical to v0.2.
                if state[nr, nc] == CONTAINED:
                    continue
                if state[nr, nc] != UNBURNED:
                    continue
                if not self._burnable(nr, nc):
                    continue
                p = self.spread_prob(r, c, dr, dc, diag)
                if p > 0 and self.rng.random() < p:
                    new_ignitions.append((nr, nc))

        # apply burn-out timers
        burn_timer[state == BURNING] += 1
        burned = (state == BURNING) & (burn_timer >= self.t_burn_steps)
        state[burned] = BURNED_OUT

        for (nr, nc) in new_ignitions:
            if state[nr, nc] == UNBURNED:
                state[nr, nc] = BURNING
                burn_timer[nr, nc] = 0

        return len(new_ignitions)

    def step(self) -> None:
        """Advance the CA one CMA sub-step on ``self.state``: tau(S, R, Theta)."""
        self._transition(self.state, self.burn_timer)
        self.step_count += 1

    def advance(self, minutes: float) -> None:
        """Advance the fire by ``minutes`` of wall-clock time (>= 1 CMA step)."""
        n = max(1, int(round(minutes / self.dt_cma_min)))
        for _ in range(n):
            self.step()

    def advance_state(
        self,
        state: np.ndarray,
        burn_timer: np.ndarray,
        minutes: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance an *injected* fire state by ``minutes`` (>= 1 CMA sub-step).

        Used by the :class:`WildfireCmaAgent`, which owns the authoritative
        ``S`` (read from the ``gis.cell_state`` sensor each step) and the
        per-cell ``burn_timer`` carried across agent steps. Returns the same
        arrays after mutation so callers can diff against a pre-step copy to
        derive the ``gis.cell_mutations`` edit set.
        """
        n = max(1, int(round(minutes / self.dt_cma_min)))
        for _ in range(n):
            self._transition(state, burn_timer)
        return state, burn_timer

    # -- diagnostics -------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        total = self.state.size
        burning = int(np.count_nonzero(self.state == BURNING))
        burned = int(np.count_nonzero(self.state == BURNED_OUT))
        return {
            "step": self.step_count,
            "burning_cells": burning,
            "burned_cells": burned,
            "affected_cells": burning + burned,
            "fraction_burned": (burning + burned) / total,
            "front_size": burning,
        }

    def fire_mask(self) -> np.ndarray:
        """Boolean mask of cells currently or previously on fire."""
        return self.state != UNBURNED
