"""FirefighterAgent -- the aero tanker responder as a palaestrAI agent (v0.3).

This is the third hazard/responder agent in the v0.2 multi-agent split (after
:class:`~palaestrai_socal.agents.wildfire_agent.WildfireCmaMuscle` and
:class:`~palaestrai_socal.agents.damage_agent.DamageMapperMuscle`). It models a
fleet of ``n_planes`` Large Air Tankers (Erickson MD-87 class): each env step it
reads the GIS fire field + wind, decides how much retardant line the fleet can
lay (grounding itself in high wind), and writes a contiguous ``SUPPRESSED`` line
just ahead of the fire head through the already-exposed ``gis.cell_mutations``
actuator (state ``SUPPRESSED`` on ``LAYER_SUPPRESSION``).

Like the other two agents it is **scripted** (palaestrAI dummy brain + dummy
objective); only the muscle is custom. The numpy-only decision logic lives in
:mod:`palaestrai_socal.agents.firefighter_core` so it is unit-testable without
palaestrai. The single operational parameter is ``n_planes``; every other number
is a documented constant in ``firefighter_core``. The structure mirrors
``wildfire_agent.py`` (``_find`` / ``_suffix_match`` / ``_ensure_geo``) so adding
more responder *types* later is a matter of new params, not new plumbing.

Muscle params (experiment YAML ``muscle.params``)
-------------------------------------------------
* ``n_planes``        number of tankers in service (the one operational knob)
* ``env_step_min``    wall-clock minutes per env step (default 60) -> drop count
* ``wind_speed`` / ``wind_dir_deg``  fallback wind if no ``gis.wind_field``
  sensor is subscribed (matches the GisWorldEnvironment config, like the
  wildfire muscle); used for the grounding decision and downwind targeting
* ``raster_nrows`` / ``raster_ncols`` / ``bounds`` / ``cell_size_m``  static
  geometry fallbacks (== gis_world config), so this agent can subscribe to only
  equal-length grid sensors (cell_state, fuel_class). See docs/AGENTS.md.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from palaestrai.agent.muscle import Muscle
from palaestrai.agent import ActuatorInformation, SensorInformation

LOG = logging.getLogger("palaestrai_socal.agents.firefighter_agent")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal import spaces  # noqa: E402
from palaestrai_socal.agents.damage_core import (  # noqa: E402
    coerce_to_actuator_space as _coerce,
)
from palaestrai_socal.agents.firefighter_core import (  # noqa: E402
    GROUND_WIND_MS,
    drops_this_step,
    is_grounded,
    line_km_this_step,
    retardant_budget,
    select_retardant_line,
)


def _suffix_match(uid: str, suffix: str) -> bool:
    """True if a (possibly env-prefixed) uid ends with ``suffix``."""
    return uid == suffix or uid.endswith("." + suffix) or uid.endswith(suffix)


def _find(sensors: List[SensorInformation], suffix: str):
    for s in sensors:
        if _suffix_match(s.uid, suffix):
            return s
    return None


class FirefighterMuscle(Muscle):
    """Scripted muscle: lay a retardant firebreak ahead of the fire head."""

    def __init__(
        self,
        n_planes: int = 1,
        env_step_min: float = 60.0,
        wind_speed: float = 15.0,
        wind_dir_deg: float = 45.0,
        raster_nrows: Optional[int] = None,
        raster_ncols: Optional[int] = None,
        bounds: Optional[List[float]] = None,
        cell_size_m: Optional[float] = None,
    ):
        super().__init__()
        self._n_planes = int(n_planes)
        self._env_step_min = float(env_step_min)
        # fallback wind (matches the env config) for the grounding decision and
        # downwind targeting when no gis.wind_field sensor is subscribed.
        self._wind_speed = float(wind_speed)
        self._wind_dir_deg = float(wind_dir_deg)
        self._geo = dict(
            raster_nrows=raster_nrows,
            raster_ncols=raster_ncols,
            bounds=tuple(bounds) if bounds is not None else None,
            cell_size_m=cell_size_m,
        )
        self._shape: Optional[Tuple[int, int]] = None
        self._cell_size_m: Optional[float] = None
        self._line_km_cumulative = 0.0

    # -- geometry bootstrap from sensors / params --------------------------
    def _ensure_geo(self, sensors: List[SensorInformation]) -> bool:
        if self._shape is not None and self._cell_size_m is not None:
            return True
        # grid shape: prefer the sensor, else the params (== env config).
        shape_s = _find(sensors, "gis.grid_shape")
        if shape_s is not None:
            nr, nc = (int(x) for x in np.asarray(shape_s.value).ravel()[:2])
        elif self._geo["raster_nrows"] and self._geo["raster_ncols"]:
            nr = int(self._geo["raster_nrows"])
            nc = int(self._geo["raster_ncols"])
        else:
            cs = _find(sensors, "gis.cell_state")
            if cs is None:
                LOG.warning("firefighter missing grid shape (no sensor/param)")
                return False
            n = int(np.asarray(cs.value).size)
            nr = nc = int(round(n ** 0.5))

        # cell size: prefer the sensor, else the param, else compute from bounds.
        size_s = _find(sensors, "gis.cell_size_m")
        if size_s is not None:
            delta_m = float(np.asarray(size_s.value).ravel()[0])
        elif self._geo["cell_size_m"] is not None:
            delta_m = float(self._geo["cell_size_m"])
        else:
            bounds_s = _find(sensors, "gis.bounds")
            if bounds_s is not None:
                bounds = tuple(float(x)
                               for x in np.asarray(bounds_s.value).ravel()[:4])
            elif self._geo["bounds"] is not None:
                bounds = tuple(float(x) for x in self._geo["bounds"])
            else:
                from wildfire_cma.gis import SOCAL_BOUNDS
                bounds = tuple(float(x) for x in SOCAL_BOUNDS)
            from wildfire_cma.gis import _approx_cell_size_m
            delta_m = float(_approx_cell_size_m(bounds, nr, nc))

        self._shape = (nr, nc)
        self._cell_size_m = delta_m
        LOG.info("firefighter geo: grid=%dx%d delta=%.0fm n_planes=%d",
                 nr, nc, delta_m, self._n_planes)
        return True

    # -- inference ---------------------------------------------------------
    def propose_actions(
        self,
        sensors: List[SensorInformation],
        actuators_available: List[ActuatorInformation],
    ) -> Tuple[List[ActuatorInformation], Any]:
        muts: List[Tuple[int, int, int, int]] = []
        # wind: prefer the live sensor, else the configured fallback.
        wind_speed, wind_dir = self._wind_speed, self._wind_dir_deg
        wf = _find(sensors, "gis.wind_field")
        if wf is not None:
            w = np.asarray(wf.value, dtype=float).ravel()
            if w.size >= 2:
                wind_speed, wind_dir = float(w[0]), float(w[1])

        cs = _find(sensors, "gis.cell_state")
        if cs is not None and self._ensure_geo(sensors):
            nr, nc = self._shape
            grid = np.asarray(cs.value, dtype=np.int8).reshape(nr, nc)
            fuel = None
            fs = _find(sensors, "gis.fuel_class")
            if fs is not None:
                fuel = np.asarray(fs.value, dtype=float).reshape(nr, nc)
            budget = retardant_budget(
                self._n_planes, wind_speed, self._env_step_min, self._cell_size_m)
            cells = select_retardant_line(grid, fuel, wind_dir, budget)
            muts = [(r, c, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)
                    for (r, c) in cells]

        # write the SUPPRESSED line (dtype-coerced, like the fire/damage agents).
        vec = spaces.encode_mutations(muts, cap=spaces.CAP)
        for act in actuators_available:
            if _suffix_match(act.uid, "gis.cell_mutations"):
                act(_coerce(vec, act))

        self._line_km_cumulative += line_km_this_step(
            self._n_planes, self._env_step_min, wind_speed)
        telemetry: Dict[str, float] = {
            "planes_in_service": float(self._n_planes),
            "drops_this_step": drops_this_step(
                self._n_planes, self._env_step_min, wind_speed),
            "retardant_cells": float(len(muts)),
            "grounded": 1.0 if is_grounded(wind_speed) else 0.0,
            "line_km_cumulative": float(self._line_km_cumulative),
        }
        self._last_telemetry = telemetry
        return actuators_available, telemetry

    def update(self, update: Any):
        pass

    def prepare_model(self):
        pass
