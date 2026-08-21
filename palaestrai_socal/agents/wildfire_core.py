"""Numpy-only wildfire driver behind the :class:`WildfireCmaMuscle`.

The v0.2 design moves the wildfire *dynamics* out of the environment and into an
agent. :class:`GisWorldEnvironment` is now a passive spatial substrate: it holds
the authoritative cell-state grid ``S`` and merely *applies* the
``gis.cell_mutations`` edits an agent writes. This module is the agent-side fire
brain, kept free of any palaestrAI / pandapower import so it can be unit-tested
with numpy alone (``tests/test_wildfire_agent.py``).

Each environment step the driver:

1. reads the authoritative ``S`` (decoded from the ``gis.cell_state`` sensor),
2. injects the configured ignition point(s) once, at/after ``ignition_step``
   (idempotent: re-setting an already-burning cell is a no-op),
3. advances the GUARDIAN CMA ``tau`` by one environment step on a *copy* of
   ``S`` via :meth:`WildfireCMA.advance_state` (the agent owns the per-cell
   ``burn_timer`` across steps), and
4. diffs the advanced grid against the pre-step grid and returns the changed
   cells as ``(row, col, state, layer)`` mutations for the
   ``gis.cell_mutations`` actuator.

Ignition points are GEOGRAPHIC ``(lon, lat)`` -- an agent parameter, never an
environment actuator -- converted to raster ``(row, col)`` via
``RasterStack.lonlat_to_rc`` at ignition time, exactly as the GUARDIAN
Overseer-Adversary's ``Theta`` prescribes.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_cma.cma import (  # noqa: E402
    BURNING,
    RasterStack,
    Theta,
    WildfireCMA,
)
from wildfire_cma.wind_field import (  # noqa: E402
    perimeter_informed_wind_field,
    reclassify_burned_footprint,
    contain_burnable_footprint,
)

# layer code for fire-written cells (mirrors palaestrai_socal.spaces.LAYER_FIRE,
# duplicated here so this module stays import-light / palaestrai-free).
LAYER_FIRE = 0

# default ignition: dense LA-basin cluster (Eaton-fire-like origin), matching
# the v0.1 environment default.
DEFAULT_IGNITION = (-118.13, 34.19)


class WildfireDriver:
    """Drives the wildfire CMA from an injected cell-state grid.

    Parameters
    ----------
    fuel, dem:
        Co-registered raster layers (2-D arrays of equal shape). The agent
        rebuilds these from the ``gis.fuel_class`` / ``gis.elevation_m`` sensors.
    delta_m:
        Cell size in metres (``gis.cell_size_m`` sensor).
    bounds:
        ``(minlon, minlat, maxlon, maxlat)`` (``gis.bounds`` sensor); used to
        convert ignition ``(lon, lat)`` to raster ``(row, col)``
    ignition_points:
        List of ``(lon, lat)`` geographic ignition coordinates (agent param).
    ignition_rc:
        Optional explicit ``(row, col)`` ignition cells (tests / low-level use).
    ignition_step:
        Environment step at which ignition is injected (default 1 = first step).
    env_step_min:
        Wall-clock minutes advanced per environment step (CMA runs
        ``env_step_min / dt_cma_min`` sub-steps each step).
    perimeter_path:
        (v0.5) Path to GeoJSON perimeter used to build the spatial wind field
        on-the-fly.  Requires ``base_speed``.  If ``wind_field_npz`` is also
        supplied, the npz takes precedence.
    base_speed:
        (v0.5) Base wind speed [m/s] for the perimeter-informed wind field.
    boundary_gain:
        (v0.5) Fractional speed gain toward the deepest interior (default 0.3).
    wind_field_npz:
        (v0.5) Path to a .npz file containing key ``wind_field`` (nrows,ncols,2).
        Overrides on-the-fly build.
    fuel_reclass:
        (v0.5) If True, reclassify non-burnable class-0 cells inside the real
        perimeter to chaparral (class 3) before building the CMA.  Requires
        ``perimeter_path``.  Needed for Palisades; Eaton: False.
    """

    def __init__(
        self,
        fuel: np.ndarray,
        dem: np.ndarray,
        delta_m: float,
        bounds: Tuple[float, float, float, float],
        ignition_points: Optional[Sequence[Tuple[float, float]]] = None,
        ignition_rc: Optional[Sequence[Tuple[int, int]]] = None,
        ignition_step: int = 1,
        env_step_min: float = 60.0,
        dt_cma_min: float = 5.0,
        t_burn_steps: int = 6,
        kappa: float = 1.5,
        dead_fuel_moisture: float = 0.05,
        wind_speed: float = 15.0,
        wind_dir_deg: float = 45.0,
        seed: int = 0,
        # v0.5 spatial wind kwargs (all default OFF => existing behaviour unchanged)
        perimeter_path: Optional[str] = None,
        base_speed: Optional[float] = None,
        boundary_gain: float = 0.3,
        wind_field_npz: Optional[str] = None,
        fuel_reclass: bool = False,
        # v0.5.1: arrest spread at the real perimeter (see contain_burnable_footprint).
        # None/0 => no containment (unbounded growth, pre-v0.5.1 behaviour). A
        # positive value makes cells more than N cells outside the perimeter
        # non-burnable so the calibrated no-FF baseline settles AT the real
        # perimeter and holds a stable extent through the full run.
        containment_margin: Optional[int] = None,
    ):
        fuel_arr = np.asarray(fuel, dtype=np.int16)

        # Compute the real perimeter mask ONCE if any perimeter-driven feature is
        # requested (reclass, containment, or on-the-fly wind field).
        _real_mask = None
        if perimeter_path is not None and (
            fuel_reclass or containment_margin is not None or base_speed is not None
        ):
            # Lazy import to keep this module palaestrai-free; perimeter_validation
            # lives under analysis/ which is on PYTHONPATH via _ROOT insertion above.
            import importlib
            pv = importlib.import_module("analysis.perimeter_validation")
            polys = pv.load_perimeter_polygons(perimeter_path)
            nrows_f, ncols_f = fuel_arr.shape
            _real_mask = pv.rasterize_perimeter(polys, bounds, nrows_f, ncols_f)

        # Optionally reclassify non-burnable cells inside the real perimeter.
        # Needed for Palisades (~13% urban/coastal inside footprint); Eaton: off.
        if fuel_reclass and _real_mask is not None:
            fuel_arr = reclassify_burned_footprint(fuel_arr, _real_mask, target_class=3)

        # Optionally arrest spread a few cells beyond the real perimeter so the
        # calibrated fire fills its footprint then holds (no unbounded growth).
        if containment_margin is not None and _real_mask is not None:
            fuel_arr = contain_burnable_footprint(
                fuel_arr, _real_mask, margin_cells=int(containment_margin)
            )

        self.raster = RasterStack(
            fuel=fuel_arr,
            dem=np.asarray(dem, dtype=float),
            delta_m=float(delta_m),
            bounds=tuple(bounds),
        )
        self.ignition_points: List[Tuple[float, float]] = [
            tuple(p) for p in (ignition_points or [DEFAULT_IGNITION])
        ]
        self.ignition_rc: List[Tuple[int, int]] = [
            tuple(p) for p in (ignition_rc or [])
        ]
        self.ignition_step = int(ignition_step)
        self.env_step_min = float(env_step_min)

        theta = Theta(
            ignition_points=[],   # injection handled explicitly by this driver
            wind_speed=float(wind_speed),
            wind_dir_deg=float(wind_dir_deg),
            dead_fuel_moisture=float(dead_fuel_moisture),
            kappa=float(kappa),
        )
        # Build the CMA with an empty Theta so its constructor does not ignite;
        # we control ignition timing (ignition_step) ourselves.
        self._cma = WildfireCMA(
            self.raster,
            theta,
            dt_cma_min=float(dt_cma_min),
            t_burn_steps=int(t_burn_steps),
            seed=int(seed),
        )

        # v0.5: inject optional per-cell wind field.
        # npz takes precedence; else build on-the-fly from perimeter + scalars.
        if wind_field_npz is not None:
            wf = np.load(wind_field_npz)["wind_field"]
            self._cma.set_wind_field(wf)
        elif perimeter_path is not None and base_speed is not None:
            # Reuse the mask computed above (same grid) if available.
            if _real_mask is not None:
                real_mask = _real_mask
            else:
                import importlib
                pv = importlib.import_module("analysis.perimeter_validation")
                polys = pv.load_perimeter_polygons(perimeter_path)
                nrows_c, ncols_c = self.raster.shape
                real_mask = pv.rasterize_perimeter(polys, bounds, nrows_c, ncols_c)
            wf = perimeter_informed_wind_field(
                real_mask, float(base_speed), float(boundary_gain)
            )
            self._cma.set_wind_field(wf)

        # the driver owns the authoritative burn timer across env steps
        self.burn_timer = np.zeros(self.raster.shape, dtype=np.int16)
        self._step = 0
        self._ignited = False

    # -- episode boundary --------------------------------------------------
    def reset(self) -> None:
        """Re-arm the driver for a fresh episode WITHOUT rebuilding it.

        palaestrAI resets the environment between episodes -- the raster goes
        back to all-UNBURNED -- but the muscle (and therefore this driver)
        survives. Without clearing the ignition latch the fire is injected once,
        in episode 0, and every later episode runs its full length on an empty
        raster. See :meth:`WildfireCmaMuscle.reset`.

        Deliberately *not* reset:

        ``self._cma.rng``
            The stochastic spread stream continues across episodes, so each
            episode draws a different fire. Re-seeding here would replay one
            identical fire N times and remove the environment variation RL
            training depends on. The run as a whole stays reproducible: the
            whole sequence is determined by the single configured ``seed``.

        The static build products (fuel raster, per-cell wind field, perimeter
        mask) are episode-invariant and expensive -- rebuilding the driver would
        re-read the perimeter shapefile every episode -- so they are kept.
        """
        self.burn_timer.fill(0)
        self._step = 0
        self._ignited = False

    # -- helpers -----------------------------------------------------------
    def ignition_cells(self) -> List[Tuple[int, int]]:
        """Resolve configured ignition points to raster ``(row, col)`` cells."""
        cells = [self.raster.lonlat_to_rc(lon, lat)
                 for (lon, lat) in self.ignition_points]
        cells += list(self.ignition_rc)
        return cells

    def set_wind(self, wind_speed: float, wind_dir_deg: float) -> None:
        """Update the CMA wind (driven by the ``gis.wind_field`` sensor).

        When a per-cell wind_field is active it is authoritative; the scalar
        sensor is ignored so calibrated runs are not clobbered by the env wind.
        """
        if self._cma._wind_field is not None:
            # Per-cell field is active; scalar sensor is a no-op for calibrated runs.
            return
        self._cma.theta.wind_speed = float(wind_speed)
        self._cma.theta.wind_dir_deg = float(wind_dir_deg)
        self._cma.theta.clamp()

    # -- main step ---------------------------------------------------------
    def step(
        self,
        cell_state: np.ndarray,
        wind: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Advance the fire one environment step; return cell mutations.

        ``cell_state`` is the authoritative 2-D ``S`` grid read from the
        ``gis.cell_state`` sensor. Returns the list of changed cells as
        ``(row, col, new_state, LAYER_FIRE)`` tuples to write back through the
        ``gis.cell_mutations`` actuator. The returned list is empty when nothing
        changed (e.g. before ignition or after burn-out).
        """
        self._step += 1
        if wind is not None:
            self.set_wind(wind[0], wind[1])

        nr, nc = self.raster.shape
        state = np.asarray(cell_state, dtype=np.int8).reshape(nr, nc).copy()
        before = state.copy()

        # idempotent ignition injection at/after ignition_step
        if not self._ignited and self._step >= self.ignition_step:
            for (r, c) in self.ignition_cells():
                r = int(np.clip(r, 0, nr - 1))
                c = int(np.clip(c, 0, nc - 1))
                if self._cma._burnable(r, c):
                    state[r, c] = BURNING
                    self.burn_timer[r, c] = 0
            self._ignited = True

        # advance the CMA tau on the injected grid (agent owns burn_timer)
        self._cma.advance_state(state, self.burn_timer, self.env_step_min)

        changed = np.argwhere(state != before)
        return [(int(r), int(c), int(state[r, c]), LAYER_FIRE)
                for (r, c) in changed]
