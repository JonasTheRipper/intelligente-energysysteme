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
from palaestrai_socal.agents.firefighting import (  # noqa: E402
    IncidentCommand,
    build_resources,
)
from palaestrai_socal.agents.firefighting.planner import (  # noqa: E402
    value_raster_from_buses,
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
        n_helos: int = 0,
        n_crews: int = 0,
        n_dozers: int = 0,
        n_engines: int = 0,
        doctrine: str = "auto",
        protect_assets: bool = False,
        grid_json: Optional[str] = None,
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
        # v0.4 multi-resource knobs (all optional; default to the v0.3 fleet of
        # tankers only, so an experiment that sets only n_planes is unchanged).
        self._n_helos = int(n_helos)
        self._n_crews = int(n_crews)
        self._n_dozers = int(n_dozers)
        self._n_engines = int(n_engines)
        self._doctrine = str(doctrine)
        self._protect_assets = bool(protect_assets)
        self._grid_json = grid_json
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
        # the deterministic fleet-mix allocator (DESIGN §4). Tankers-only +
        # indirect/auto doctrine reproduces v0.3's retardant line exactly.
        self._command = IncidentCommand(
            resources=build_resources(
                n_planes=self._n_planes, n_helos=self._n_helos,
                n_crews=self._n_crews, n_dozers=self._n_dozers,
                n_engines=self._n_engines),
            doctrine=self._doctrine,
            protect_assets=self._protect_assets,
        )
        # lazily-built per-cell grid-asset value raster for triage / point
        # protection (only when protect_assets is on). None until first built;
        # a graceful no-op if the grid JSON / pandapower are unavailable.
        self._value_raster: Optional[np.ndarray] = None
        self._value_raster_tried = False
        self._last_telemetry: Dict[str, float] = {}

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

    # -- grid-asset value raster (triage / point protection) ---------------
    def _mean_slope_deg(self, sensors: List[SensorInformation]) -> float:
        """Representative slope [deg] for ground-resource productivity.

        Prefers a ``gis.slope`` sensor; else derives a coarse mean slope from a
        ``gis.elevation_m`` sensor; else 0 (flat) so ground crews/dozers are not
        penalised when no terrain sensor is subscribed (graceful fallback).
        """
        sl = _find(sensors, "gis.slope")
        if sl is not None:
            v = np.asarray(sl.value, dtype=float).ravel()
            if v.size:
                return float(np.nanmean(np.abs(v)))
        el = _find(sensors, "gis.elevation_m")
        if el is not None and self._shape is not None and self._cell_size_m:
            nr, nc = self._shape
            dem = np.asarray(el.value, dtype=float).reshape(nr, nc)
            gy, gx = np.gradient(dem, float(self._cell_size_m))
            grade = np.sqrt(gy * gy + gx * gx)
            return float(np.degrees(np.arctan(np.nanmean(grade))))
        return 0.0

    def _ensure_value_raster(self) -> Optional[np.ndarray]:
        """Build the per-cell grid-asset value raster once (lazy).

        Reuses the DamageMapper bus->cell registration (the inverse of its
        de-energisation) and weights each cell by the served load attached to
        the bus there, so triage / point protection favour grid-critical ground
        (DESIGN §5). Any failure (no grid JSON, no pandapower) degrades to None
        -- triage simply stays off.
        """
        if self._value_raster_tried:
            return self._value_raster
        self._value_raster_tried = True
        if not self._protect_assets or self._shape is None:
            return None
        try:
            import pandapower as pp
            from wildfire_cma.damage import _bus_lonlat
            from palaestrai_socal.agents.damage_core import DamageMapperDriver
            from palaestrai_socal.agents.damage_agent import DEFAULT_GRID_JSON

            grid_json = self._grid_json or DEFAULT_GRID_JSON
            net = pp.from_json(grid_json)
            bus_lonlat = {}
            bus_value = {}
            for b in net.bus.index:
                ll = _bus_lonlat(net, b)
                if ll is not None:
                    bus_lonlat[int(b)] = ll
            for _, row in net.load.iterrows():
                b = int(row["bus"])
                bus_value[b] = bus_value.get(b, 0.0) + float(row.get("p_mw", 0.0))
            bounds = self._geo["bounds"]
            if bounds is None:
                from wildfire_cma.gis import SOCAL_BOUNDS
                bounds = SOCAL_BOUNDS
            driver = DamageMapperDriver(bus_lonlat, bounds, self._shape)
            self._value_raster = value_raster_from_buses(
                self._shape, driver.bus_cell, bus_value)
            LOG.info("firefighter value raster: %d asset cells, max %.1f MW",
                     int(np.count_nonzero(self._value_raster)),
                     float(self._value_raster.max() if self._value_raster.size
                           else 0.0))
        except Exception as exc:                       # pragma: no cover
            LOG.warning("firefighter value raster unavailable (%s); "
                        "triage off", exc)
            self._value_raster = None
        return self._value_raster

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
            slope_deg = self._mean_slope_deg(sensors)
            value_raster = self._ensure_value_raster()
            # delegate the multi-resource decision to the incident commander;
            # tankers-only + indirect/auto reproduces v0.3's retardant line.
            muts = self._command.propose(
                grid, fuel, wind_speed, wind_dir,
                slope_deg=slope_deg, value_raster=value_raster,
                step_min=self._env_step_min, cell_m=self._cell_size_m)

        # write the suppression/containment edits (dtype-coerced, like the
        # fire/damage agents).
        vec = spaces.encode_mutations(muts, cap=spaces.CAP)
        for act in actuators_available:
            if _suffix_match(act.uid, "gis.cell_mutations"):
                act(_coerce(vec, act))

        self._line_km_cumulative += line_km_this_step(
            self._n_planes, self._env_step_min, wind_speed)
        n_supp = sum(1 for m in muts if m[2] == spaces.SUPPRESSED)
        n_cont = sum(1 for m in muts if m[2] == spaces.CONTAINED)
        # Telemetry is recorded on the muscle instance for inspection, but is
        # NOT returned on the brain/data channel: returning a dict there made
        # palaestrAI's DummyBrain do ``dict + int`` each turn (a non-fatal
        # TypeError logged every step). Per DESIGN §6 we return None on that
        # channel; resource state remains reconstructable from the stored
        # SUPPRESSED/CONTAINED cell-state diff (analysis/plane_icons.py).
        self._last_telemetry = {
            "planes_in_service": float(self._n_planes),
            "helos_in_service": float(self._n_helos),
            "crews_in_service": float(self._n_crews),
            "dozers_in_service": float(self._n_dozers),
            "engines_in_service": float(self._n_engines),
            "drops_this_step": drops_this_step(
                self._n_planes, self._env_step_min, wind_speed),
            "suppressed_cells": float(n_supp),
            "contained_cells": float(n_cont),
            "mutation_cells": float(len(muts)),
            "grounded": 1.0 if is_grounded(wind_speed) else 0.0,
            "line_km_cumulative": float(self._line_km_cumulative),
        }
        return actuators_available, None

    def update(self, update: Any):
        pass

    def prepare_model(self):
        pass
