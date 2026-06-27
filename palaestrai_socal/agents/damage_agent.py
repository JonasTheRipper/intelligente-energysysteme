"""DamageMapperAgent -- the wildfire->grid damage mapper as a palaestrAI agent.

This is the third actor in the v0.2 split (after :class:`GisWorldEnvironment` and
:class:`~palaestrai_socal.agents.wildfire_agent.WildfireCmaMuscle`). It reads the
GIS substrate's ``gis.cell_state`` sensor and, each step, drives the REAL MIDAS
powergrid's ``...load-<bus>-<idx>.p_mw`` actuators to ``0`` on every bus the fire
has reached -- a **load-shed trip** (see :mod:`damage_core` for why this replaces
v0.1's ``in_service=False`` bus trips under mosaik).

Only the :class:`DamageMapperMuscle` is custom; brain/objective are palaestrAI
dummies (this agent does not learn -- it is a scripted hazard-consequence
operator). The numpy-only mapping logic lives in
:mod:`palaestrai_socal.agents.damage_core` so it is unit-testable without
palaestrai / pandapower.

Muscle params (experiment YAML ``muscle.params``)
-------------------------------------------------
* ``grid_json``  path to the pandapower JSON whose bus ``geo`` + indices match
  the MIDAS powergrid (default: the rescaled SoCal MIDAS grid).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from palaestrai.agent.muscle import Muscle
from palaestrai.agent import ActuatorInformation, SensorInformation

LOG = logging.getLogger("palaestrai_socal.agents.damage_agent")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal.agents.damage_core import (  # noqa: E402
    DamageMapperDriver,
    coerce_to_actuator_space as _coerce,
    load_actuator_bus as _load_bus,
)

DEFAULT_GRID_JSON = os.path.join(
    _ROOT, "midas_socal", "socal_grid_midas_rescaled.json"
)


def _suffix_match(uid: str, suffix: str) -> bool:
    return uid == suffix or uid.endswith("." + suffix) or uid.endswith(suffix)


def _find(sensors: List[SensorInformation], suffix: str):
    for s in sensors:
        if _suffix_match(s.uid, suffix):
            return s
    return None


class DamageMapperMuscle(Muscle):
    """Scripted muscle: shed load on fire-affected buses."""

    def __init__(
        self,
        grid_json: Optional[str] = None,
        raster_nrows: Optional[int] = None,
        raster_ncols: Optional[int] = None,
        bounds: Optional[List[float]] = None,
    ):
        super().__init__()
        self._grid_json = grid_json or DEFAULT_GRID_JSON
        # Static geometry fallbacks: this agent subscribes only to the single
        # ``gis.cell_state`` sensor (palaestrAI's flat per-agent memory cannot
        # store sensors of differing length together), so grid_shape/bounds are
        # supplied as params matching the GisWorldEnvironment. See docs/AGENTS.md.
        self._geo = dict(
            raster_nrows=raster_nrows,
            raster_ncols=raster_ncols,
            bounds=tuple(bounds) if bounds is not None else None,
        )
        self._driver: Optional[DamageMapperDriver] = None
        self._bus_lonlat: Optional[Dict[int, Tuple[float, float]]] = None

    # -- bus geo (loaded once from the pandapower JSON) ---------------------
    def _bus_geo(self) -> Dict[int, Tuple[float, float]]:
        if self._bus_lonlat is not None:
            return self._bus_lonlat
        import pandapower as pp
        from wildfire_cma.damage import _bus_lonlat

        net = pp.from_json(self._grid_json)
        out: Dict[int, Tuple[float, float]] = {}
        for b in net.bus.index:
            ll = _bus_lonlat(net, b)
            if ll is not None:
                out[int(b)] = ll
        self._bus_lonlat = out
        LOG.info("damage mapper: loaded geo for %d buses", len(out))
        return out

    def _ensure_driver(self, sensors: List[SensorInformation]) -> bool:
        if self._driver is not None:
            return True
        # grid shape: prefer the sensor, else the params (== env config).
        shape_s = _find(sensors, "gis.grid_shape")
        if shape_s is not None:
            nr, nc = (int(x) for x in np.asarray(shape_s.value).ravel()[:2])
        elif self._geo["raster_nrows"] and self._geo["raster_ncols"]:
            nr, nc = int(self._geo["raster_nrows"]), int(self._geo["raster_ncols"])
        else:
            LOG.warning("damage mapper missing grid shape (no sensor/param)")
            return False

        # bounds: prefer the sensor, else the param, else SOCAL_BOUNDS default.
        bounds_s = _find(sensors, "gis.bounds")
        if bounds_s is not None:
            bounds = tuple(float(x) for x in np.asarray(bounds_s.value).ravel()[:4])
        elif self._geo["bounds"] is not None:
            bounds = tuple(float(x) for x in self._geo["bounds"])
        else:
            from wildfire_cma.gis import SOCAL_BOUNDS
            bounds = tuple(float(x) for x in SOCAL_BOUNDS)

        self._driver = DamageMapperDriver(self._bus_geo(), bounds, (nr, nc))
        LOG.info("damage mapper driver built: %d co-registered buses",
                 len(self._driver.bus_cell))
        return True

    # -- inference ---------------------------------------------------------
    def propose_actions(
        self,
        sensors: List[SensorInformation],
        actuators_available: List[ActuatorInformation],
    ) -> Tuple[List[ActuatorInformation], Any]:
        n_shed = 0
        if self._ensure_driver(sensors):
            cs = _find(sensors, "gis.cell_state")
            if cs is not None:
                nr, nc = self._driver.nrows, self._driver.ncols
                grid = np.asarray(cs.value, dtype=float).reshape(nr, nc)
                # only evaluate the buses this agent actually controls
                controllable = {
                    _load_bus(a.uid) for a in actuators_available
                }
                controllable.discard(None)
                shed = self._driver.evaluate(grid, buses=controllable)
                for act in actuators_available:
                    bus = _load_bus(act.uid)
                    if bus is not None and bus in shed:
                        # Cast to the actuator's own space dtype/shape so the
                        # Box(dtype=np.float32) containment check passes (a
                        # python float / np.float64 fails -- see damage_core).
                        act(_coerce(0.0, act))
                        n_shed += 1
        return actuators_available, n_shed

    def update(self, update: Any):
        pass

    def prepare_model(self):
        pass
