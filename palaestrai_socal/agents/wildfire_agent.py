"""WildfireCmaAgent -- the GUARDIAN wildfire CMA as a palaestrAI agent.

This is the agent half of the v0.2 split: the wildfire dynamics that used to
live inside the monolithic v0.1 environment now run here, reading the GIS
substrate's sensors and writing cell edits back through the
``gis.cell_mutations`` actuator. :class:`GisWorldEnvironment` only *applies*
those edits, so the fire is genuinely agent-driven.

Only the :class:`WildfireCmaMuscle` is custom; the brain/objective are the
palaestrAI dummies (this agent does not learn -- it is a scripted hazard
operator). The numpy-only fire logic lives in
:mod:`palaestrai_socal.agents.wildfire_core` so it is unit-testable without
palaestrai.

Muscle params (experiment YAML ``muscle.params``)
-------------------------------------------------
* ``ignition_points``     list of ``[lon, lat]`` (default: one LA-basin point)
* ``ignition_step``       env step to inject ignition (default 1)
* ``env_step_min``        wall-clock minutes per env step (default 60)
* ``dt_cma_min``          CMA sub-step minutes (default 5)
* ``t_burn_steps``        steps a cell burns before burning out (default 6)
* ``kappa``               global ROS multiplier (default 1.5)
* ``dead_fuel_moisture``  fraction (default 0.05)
* ``wind_speed``/``wind_dir_deg``  fallback wind if no ``gis.wind_field`` sensor
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np

from palaestrai.agent.muscle import Muscle
from palaestrai.agent import ActuatorInformation, SensorInformation

LOG = logging.getLogger("palaestrai_socal.agents.wildfire_agent")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal import spaces  # noqa: E402
from palaestrai_socal.agents.damage_core import (  # noqa: E402
    coerce_to_actuator_space as _coerce,
)
from palaestrai_socal.agents.wildfire_core import (  # noqa: E402
    DEFAULT_IGNITION,
    WildfireDriver,
)


def _suffix_match(uid: str, suffix: str) -> bool:
    """True if a (possibly env-prefixed) uid ends with ``suffix``."""
    return uid == suffix or uid.endswith("." + suffix) or uid.endswith(suffix)


def _find(sensors: List[SensorInformation], suffix: str):
    for s in sensors:
        if _suffix_match(s.uid, suffix):
            return s
    return None


class WildfireCmaMuscle(Muscle):
    """Scripted muscle: advance the wildfire CMA and emit cell mutations."""

    def __init__(
        self,
        ignition_points: Optional[List[List[float]]] = None,
        ignition_step: int = 1,
        env_step_min: float = 60.0,
        dt_cma_min: float = 5.0,
        t_burn_steps: int = 6,
        kappa: float = 1.5,
        dead_fuel_moisture: float = 0.05,
        wind_speed: float = 15.0,
        wind_dir_deg: float = 45.0,
        raster_nrows: Optional[int] = None,
        raster_ncols: Optional[int] = None,
        bounds: Optional[List[float]] = None,
        cell_size_m: Optional[float] = None,
        seed: int = 0,
        # v0.5 spatial wind params (default OFF => existing behaviour unchanged)
        perimeter_path: Optional[str] = None,
        base_speed: Optional[float] = None,
        boundary_gain: float = 0.3,
        wind_field_npz: Optional[str] = None,
        fuel_reclass: bool = False,
        containment_margin: Optional[int] = None,
    ):
        super().__init__()
        # Static geometry fallbacks. palaestrAI's per-agent flat memory
        # (``_MuscleMemory._infos_to_df``) requires every subscribed sensor to
        # flatten to the SAME length, so this agent subscribes only to the
        # equal-length full-grid sensors (fuel/dem/cell_state). The small static
        # scalars (grid_shape/bounds/cell_size) are supplied here as params and
        # match the GisWorldEnvironment exactly (default bounds == SOCAL_BOUNDS;
        # delta_m via the same ``_approx_cell_size_m``). See docs/AGENTS.md.
        self._geo = dict(
            raster_nrows=raster_nrows,
            raster_ncols=raster_ncols,
            bounds=tuple(bounds) if bounds is not None else None,
            cell_size_m=cell_size_m,
        )
        self._cfg = dict(
            ignition_points=[tuple(p) for p in (ignition_points or [list(DEFAULT_IGNITION)])],
            ignition_step=int(ignition_step),
            env_step_min=float(env_step_min),
            dt_cma_min=float(dt_cma_min),
            t_burn_steps=int(t_burn_steps),
            kappa=float(kappa),
            dead_fuel_moisture=float(dead_fuel_moisture),
            wind_speed=float(wind_speed),
            wind_dir_deg=float(wind_dir_deg),
            seed=int(seed),
            # v0.5 spatial wind
            perimeter_path=perimeter_path,
            base_speed=base_speed,
            boundary_gain=float(boundary_gain),
            wind_field_npz=wind_field_npz,
            fuel_reclass=bool(fuel_reclass),
            containment_margin=(int(containment_margin) if containment_margin is not None else None),
        )
        self._driver: Optional[WildfireDriver] = None

    # -- driver bootstrap from sensors -------------------------------------
    def _ensure_driver(self, sensors: List[SensorInformation]) -> bool:
        if self._driver is not None:
            return True
        fuel_s = _find(sensors, "gis.fuel_class")
        dem_s = _find(sensors, "gis.elevation_m")
        if fuel_s is None or dem_s is None:
            LOG.warning("wildfire muscle missing fuel/dem sensors; cannot build")
            return False

        # grid shape: prefer the sensor, else the params (which == config).
        shape_s = _find(sensors, "gis.grid_shape")
        if shape_s is not None:
            nr, nc = (int(x) for x in np.asarray(shape_s.value).ravel()[:2])
        elif self._geo["raster_nrows"] and self._geo["raster_ncols"]:
            nr, nc = int(self._geo["raster_nrows"]), int(self._geo["raster_ncols"])
        else:  # last resort: infer a square grid from the flat fuel sensor
            n = int(np.asarray(fuel_s.value).size)
            nr = nc = int(round(n ** 0.5))

        # bounds: prefer the sensor, else the param, else SOCAL_BOUNDS default.
        bounds_s = _find(sensors, "gis.bounds")
        if bounds_s is not None:
            bounds = tuple(float(x) for x in np.asarray(bounds_s.value).ravel()[:4])
        elif self._geo["bounds"] is not None:
            bounds = tuple(float(x) for x in self._geo["bounds"])
        else:
            from wildfire_cma.gis import SOCAL_BOUNDS
            bounds = tuple(float(x) for x in SOCAL_BOUNDS)

        # cell size: prefer the sensor, else the param, else compute it the SAME
        # way the env does (``_approx_cell_size_m``) so spread distance matches.
        size_s = _find(sensors, "gis.cell_size_m")
        if size_s is not None:
            delta_m = float(np.asarray(size_s.value).ravel()[0])
        elif self._geo["cell_size_m"] is not None:
            delta_m = float(self._geo["cell_size_m"])
        else:
            from wildfire_cma.gis import _approx_cell_size_m
            delta_m = float(_approx_cell_size_m(bounds, nr, nc))

        fuel = np.asarray(fuel_s.value, dtype=float).reshape(nr, nc)
        dem = np.asarray(dem_s.value, dtype=float).reshape(nr, nc)
        self._driver = WildfireDriver(
            fuel=fuel, dem=dem, delta_m=delta_m, bounds=bounds, **self._cfg
        )
        LOG.info("wildfire driver built: grid=%dx%d delta=%.0fm", nr, nc, delta_m)
        return True

    # -- inference ---------------------------------------------------------
    def propose_actions(
        self,
        sensors: List[SensorInformation],
        actuators_available: List[ActuatorInformation],
    ) -> Tuple[List[ActuatorInformation], Any]:
        muts: List[Tuple[int, int, int, int]] = []
        if self._ensure_driver(sensors):
            cs = _find(sensors, "gis.cell_state")
            if cs is not None:
                nr, nc = self._driver.raster.shape
                grid = np.asarray(cs.value, dtype=float).reshape(nr, nc)
                wind = None
                wf = _find(sensors, "gis.wind_field")
                if wf is not None:
                    w = np.asarray(wf.value, dtype=float).ravel()
                    if w.size >= 2:
                        wind = (float(w[0]), float(w[1]))
                muts = self._driver.step(grid, wind=wind)

        vec = spaces.encode_mutations(muts, cap=spaces.CAP)
        for act in actuators_available:
            if _suffix_match(act.uid, "gis.cell_mutations"):
                # Cast to the actuator's own space dtype/shape so the Box
                # containment check passes regardless of float32/float64.
                act(_coerce(vec, act))
        return actuators_available, len(muts)

    def reset(self):
        """Re-arm the fire for the next episode (palaestrAI episode boundary).

        The environment is reset for us, but this muscle is kept alive across
        episodes, and the ignition latch lives on the driver. Without this the
        fire ignites in episode 0 only and every subsequent episode is 60 steps
        of an unburnt raster -- silently, since an empty episode looks exactly
        like a perfectly-suppressed one in the store.

        The driver itself is preserved (see :meth:`WildfireDriver.reset`): its
        raster and wind field do not change between episodes.
        """
        if self._driver is not None:
            self._driver.reset()

    def update(self, update: Any):
        pass

    def prepare_model(self):
        pass
