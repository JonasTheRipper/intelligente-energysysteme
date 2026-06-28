"""GisWorldEnvironment -- the spatial hazard substrate (palaestrAI env).

v0.2 splits the v0.1 monolith into two palaestrAI environments:

* this ``GisWorldEnvironment`` -- the **passive spatial state** ``S`` (the GIS
  raster: fuel, elevation, wind, and the per-cell hazard ``cell_state`` grid);
* a real ``palaestrai_mosaik.MosaikEnvironment`` -- the **power grid** stepped
  by MIDAS/mosaik (see :mod:`palaestrai_socal.midas_grid_env`).

The wildfire dynamics no longer live in the environment. Instead the
:class:`~palaestrai_socal.agents.wildfire_agent.WildfireCmaAgent` reads the GIS
sensors and writes cell edits back through the ``gis.cell_mutations`` actuator;
this environment merely *applies* those mutations to ``S`` and re-publishes the
spatial telemetry. That keeps the env a clean, hazard-agnostic substrate that a
future FirefighterAgent or flood hazard can also write to (via the reserved
``SUPPRESSED`` / ``FLOODED`` state codes and the ``layer`` field).

Sensor namespace (env uid ``gis_world`` -> ``gis_world.gis.<name>``):
    gis.grid_shape   (2,)  [nrows, ncols]            static
    gis.bounds       (4,)  [minlon, minlat, maxlon, maxlat]  static
    gis.cell_size_m  (1,)  metres                     static
    gis.fuel_class   (N,)  flattened fuel-class ids   static
    gis.elevation_m  (N,)  flattened DEM [m]          static
    gis.cell_state   (N,)  flattened hazard state     dynamic
    gis.front_cells  padded-set of (row,col,state,layer) burning cells  dynamic
    gis.wind_field   (2,)  [speed m/s, dir deg]       dynamic
    gis.front_size   (1,)  count of burning cells     dynamic
    gis.affected_cells (1,) burning + burned-out      dynamic

Actuators:
    gis.cell_mutations  padded-set of (row,col,state,layer) edits
    gis.wind_override   (2,) [speed, dir]; <0 entries mean "keep default"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("palaestrai_socal.gis_world_env")

from palaestrai.environment.environment import Environment
from palaestrai.environment.environment_baseline import EnvironmentBaseline
from palaestrai.environment.environment_state import EnvironmentState
from palaestrai.agent import SensorInformation, ActuatorInformation, RewardInformation

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal import spaces  # noqa: E402
from palaestrai_socal.agents.firefighter_core import (  # noqa: E402
    SUPPRESS_PERSIST_STEPS, age_suppressed,
)
from wildfire_cma.cma import BURNING, BURNED_OUT, UNBURNED  # noqa: E402
from wildfire_cma.gis import (  # noqa: E402
    SOCAL_BOUNDS, synthetic_socal, socal_from_srtm,
)

# how many burning cells to advertise in the gis.front_cells padded set
FRONT_CAP = 8192


class GisWorldEnvironment(Environment):
    """The GIS spatial substrate as a palaestrAI environment."""

    def __init__(self, uid: str, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)
        # the conductor flattens the YAML ``params:`` block into kwargs, but
        # programmatic callers pass ``params={...}``; support both.
        if isinstance(kwargs.get("params"), dict):
            p = dict(kwargs["params"])
        else:
            p = dict(kwargs)

        self.raster_nrows = int(p.get("raster_nrows", 600))
        self.raster_ncols = int(p.get("raster_ncols", 760))
        self.bounds = tuple(p.get("bounds", SOCAL_BOUNDS))
        self.seed = int(p.get("seed", 0))
        self.use_real_dem = bool(p.get("use_real_dem", True))
        self.env_step_min = float(p.get("env_step_min", 60.0))
        self.max_steps = int(p.get("max_steps", 120))
        self.default_wind = (
            float(p.get("wind_speed", 15.0)),
            float(p.get("wind_dir_deg", 225.0)),
        )

        self._raster = None
        self._state: Optional[np.ndarray] = None
        # per-cell SUPPRESSED-line persistence timer (v0.3): how many consecutive
        # env steps a cell has held retardant, used to age the line back to
        # UNBURNED after SUPPRESS_PERSIST_STEPS (see firefighter_core.age_suppressed).
        self._suppress_age: Optional[np.ndarray] = None
        self._wind: Tuple[float, float] = self.default_wind
        self._step = 0

    # -- terrain -----------------------------------------------------------
    def _build_raster(self):
        if self.use_real_dem:
            try:
                return socal_from_srtm(
                    nrows=self.raster_nrows, ncols=self.raster_ncols,
                    bounds=self.bounds, seed=self.seed or 7,
                )
            except FileNotFoundError as exc:
                LOG.warning("Real DEM unavailable (%s); synthetic", exc)
        return synthetic_socal(
            nrows=self.raster_nrows, ncols=self.raster_ncols,
            bounds=self.bounds, seed=self.seed or 7,
        )

    # -- spaces ------------------------------------------------------------
    @property
    def _ncell(self) -> int:
        return self.raster_nrows * self.raster_ncols

    def _sensor_list(self) -> List[SensorInformation]:
        r = self._raster
        nr, nc = r.shape
        front = np.argwhere(self._state == BURNING)
        front_muts = [(int(rr), int(cc), BURNING, spaces.LAYER_FIRE)
                      for (rr, cc) in front[:FRONT_CAP]]
        front_size = int((self._state == BURNING).sum())
        affected = int(((self._state == BURNING) |
                        (self._state == BURNED_OUT)).sum())
        out = [
            SensorInformation(
                value=np.array([nr, nc], dtype=np.float64),
                space=spaces.vector_box(0.0, 1.0e6, 2), uid="gis.grid_shape"),
            SensorInformation(
                value=np.array(self.bounds, dtype=np.float64),
                space=spaces.vector_box(-180.0, 180.0, 4), uid="gis.bounds"),
            SensorInformation(
                value=np.array([r.delta_m], dtype=np.float64),
                space=spaces.scalar_box(0.0, 1.0e5), uid="gis.cell_size_m"),
            SensorInformation(
                value=r.fuel.astype(np.float64).ravel(),
                space=spaces.vector_box(0.0, 64.0, self._ncell),
                uid="gis.fuel_class"),
            SensorInformation(
                value=r.dem.astype(np.float64).ravel(),
                space=spaces.vector_box(-500.0, 9000.0, self._ncell),
                uid="gis.elevation_m"),
            SensorInformation(
                value=self._state.astype(np.float64).ravel(),
                space=spaces.vector_box(0.0, 16.0, self._ncell),
                uid="gis.cell_state"),
            SensorInformation(
                value=spaces.encode_mutations(front_muts, cap=FRONT_CAP),
                space=spaces.mutation_space(FRONT_CAP), uid="gis.front_cells"),
            SensorInformation(
                value=np.array(self._wind, dtype=np.float64),
                space=spaces.vector_box(0.0, 360.0, 2), uid="gis.wind_field"),
            SensorInformation(
                value=np.array([front_size], dtype=np.float64),
                space=spaces.scalar_box(0.0, 1.0e7), uid="gis.front_size"),
            SensorInformation(
                value=np.array([affected], dtype=np.float64),
                space=spaces.scalar_box(0.0, 1.0e7), uid="gis.affected_cells"),
        ]
        return out

    def _actuator_list(self) -> List[ActuatorInformation]:
        return [
            ActuatorInformation(
                value=spaces.encode_mutations([], cap=spaces.CAP),
                space=spaces.mutation_space(spaces.CAP),
                uid="gis.cell_mutations"),
            ActuatorInformation(
                value=np.array([-1.0, -1.0], dtype=np.float64),
                space=spaces.vector_box(-1.0, 360.0, 2),
                uid="gis.wind_override"),
        ]

    # -- lifecycle ---------------------------------------------------------
    def start_environment(self) -> EnvironmentBaseline:
        LOG.info("starting GisWorldEnvironment %s", self.uid)
        self._raster = self._build_raster()
        self._state = np.full(self._raster.shape, UNBURNED, dtype=np.int8)
        self._suppress_age = np.zeros(self._raster.shape, dtype=np.int16)
        self._wind = self.default_wind
        self._step = 0
        self.sensors = self._sensor_list()
        self.actuators = self._actuator_list()
        return EnvironmentBaseline(
            sensors_available=self.sensors,
            actuators_available=self.actuators,
            static_world_model={
                "bounds": list(self.bounds),
                "grid_shape": [self.raster_nrows, self.raster_ncols],
                "cell_size_m": float(self._raster.delta_m),
            },
        )

    # -- step --------------------------------------------------------------
    def _apply_mutations(self, actuators: List[ActuatorInformation]) -> int:
        """Apply all agents' cell edits to ``S`` with deterministic arbitration.

        v0.2 applied mutations last-writer-wins. v0.3 collects every proposed
        ``gis.cell_mutations`` edit across all actuators (the fire and the
        firefighter both write this actuator) and resolves each cell by fixed
        priority (BURNED_OUT > SUPPRESSED > BURNING > UNBURNED) via
        :func:`spaces.arbitrate_mutations`, so suppression holds against
        same-step spread and the result is independent of agent turn order.
        """
        all_muts: List[Tuple[int, int, int, int]] = []
        for act in actuators or []:
            if act.uid.endswith("gis.cell_mutations") or act.uid == "gis.cell_mutations":
                all_muts.extend(
                    spaces.decode_mutations(np.asarray(act.value).ravel(),
                                            cap=spaces.CAP))
            elif act.uid.endswith("gis.wind_override") or act.uid == "gis.wind_override":
                v = np.asarray(act.value, dtype=np.float64).ravel()
                spd = float(v[0]) if v.size > 0 else -1.0
                ddeg = float(v[1]) if v.size > 1 else -1.0
                if spd >= 0.0:
                    self._wind = (spd, self._wind[1])
                if ddeg >= 0.0:
                    self._wind = (self._wind[0], ddeg % 360.0)
        new_state = spaces.arbitrate_mutations(self._state, all_muts)
        n_applied = int(np.count_nonzero(new_state != self._state))
        self._state = new_state
        return n_applied

    def update(self, actuators: List[ActuatorInformation]) -> EnvironmentState:
        self._step += 1
        self._apply_mutations(actuators)
        # Age the firefighter's retardant lines: a cell SUPPRESSED for
        # SUPPRESS_PERSIST_STEPS env steps reverts to UNBURNED (retardant
        # breakdown). The env owns this timer, exactly as the fire agent owns
        # the burn timer, so the reversion is recorded as ordinary state.
        age_suppressed(self._state, self._suppress_age, SUPPRESS_PERSIST_STEPS)
        self.sensors = self._sensor_list()

        front_size = int((self._state == BURNING).sum())
        affected = int(((self._state == BURNING) |
                        (self._state == BURNED_OUT)).sum())

        world_state: Dict[str, object] = {
            "kind": "gis_world",
            "step": self._step,
            "sim_minutes": self._step * self.env_step_min,
            "bounds": list(self.bounds),
            "grid_shape": [self.raster_nrows, self.raster_ncols],
            "cell_size_m": float(self._raster.delta_m),
            "wind_speed_m_per_s": float(self._wind[0]),
            "wind_dir_deg": float(self._wind[1]),
            "front_size": front_size,
            "affected_cells": affected,
            "cell_state": spaces.encode_grid(self._state, dtype="int8"),
        }
        # static layers only on the first step (DEM is large) so the timelapse
        # can rebuild the basemap entirely from the store without a live env.
        if self._step == 1:
            world_state["fuel_class"] = spaces.encode_grid(
                self._raster.fuel, dtype="int16")
            world_state["elevation_m"] = spaces.encode_grid(
                self._raster.dem, dtype="float32")

        rewards = [RewardInformation(
            value=np.array([0.0], dtype=np.float64),
            space=spaces.scalar_box(-1.0, 1.0), uid="gis_reward")]
        done = self._step >= self.max_steps
        return EnvironmentState(
            sensor_information=self.sensors,
            rewards=rewards,
            done=done,
            world_state=world_state,
        )
