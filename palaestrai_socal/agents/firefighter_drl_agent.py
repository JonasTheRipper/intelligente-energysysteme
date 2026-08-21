"""LearningFirefighterMuscle -- the v0.7 Deep-RL firefighter rollout worker.

This is the online inference half of the v0.7 DRL firefighter. Unlike the
scripted :class:`~palaestrai_socal.agents.firefighter_agent.FirefighterMuscle`
(which allocates its fleet with a fixed doctrine every step), this muscle asks a
*learned* hARL SAC/CQL policy which **doctrine** to run each env step, then
hands that doctrine to the very same deterministic
:class:`~palaestrai_socal.agents.firefighting.planner.IncidentCommand` machinery
so the physical suppression edits stay identical to the scripted teacher's.

Why subclass :class:`harl.SACMuscle`?
-------------------------------------
palaestrAI transfers the trained actor from the brain to the muscle through the
``BrainDumper`` (``sac_actor`` dump) via :meth:`prepare_model` / ``_load_model``.
By subclassing the hARL muscle we inherit that plumbing untouched -- we only
override :meth:`propose_actions` to (1) *present* the compact 17-dim observation
(:func:`firefighter_drl.extract_obs`) instead of the raw grid sensors the SAC
brain would otherwise flatten, and (2) *translate* the policy's Discrete(4)
doctrine index into an :class:`IncidentCommand` allocation, writing the result
through the existing ``gis.cell_mutations`` actuator (never letting the base
class map raw network output onto that actuator directly).

Train/serve parity
-------------------
The observation built here is the exact vector the offline harvester
(:mod:`palaestrai_socal.agents.harvest_teacher_transitions`) wrote into the CQL
replay buffer, because both call :func:`firefighter_drl.extract_obs`. The brain
(:class:`palaestrai_socal.agents.firefighter_drl_brain.FirefighterSacBrain`)
builds its networks at ``OBS_DIM`` / ``N_TACTICS``, so the actor loaded here
consumes the 17-dim obs and emits a doctrine index in ``0..3``.

Muscle params (experiment YAML ``muscle.params``)
-------------------------------------------------
* ``n_planes`` / ``n_helos`` / ``n_crews`` / ``n_dozers`` / ``n_engines``
  fleet mix (drives the resource-availability gate + IncidentCommand budgets).
* ``start_steps`` uniform-random doctrine steps before the policy is used
  (exploration warm-up; ignored in test mode).
* ``base_served_mw`` SAIDI/served-load denominator for obs feature 10.
* ``saidi_scale`` SAIDI normalisation for obs features 11-12.
* ``env_step_min`` / ``wind_speed`` / ``wind_dir_deg`` env-config fallbacks.
* ``protect_assets`` / ``grid_json`` triage value-raster config (feature 3 /
  ACT_TRIAGE); triage degrades to a no-op when no value raster is available.
* ``raster_nrows`` / ``raster_ncols`` / ``bounds`` / ``cell_size_m`` geometry
  fallbacks (== gis_world config), like the scripted firefighter.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np

from palaestrai.agent import ActuatorInformation, SensorInformation
from palaestrai.types import Mode

from harl import SACMuscle
from harl.sac.action_type import ActionType

LOG = logging.getLogger("palaestrai_socal.agents.firefighter_drl_agent")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal import spaces  # noqa: E402
from palaestrai_socal.agents import _memory_compat  # noqa: E402
from palaestrai_socal.agents import firefighter_drl as drl  # noqa: E402
from palaestrai_socal.agents.damage_core import (  # noqa: E402
    coerce_to_actuator_space as _coerce,
)
from palaestrai_socal.agents.firefighter_core import is_grounded  # noqa: E402
from palaestrai_socal.agents.firefighting import (  # noqa: E402
    IncidentCommand,
    build_resources,
)
from palaestrai_socal.agents.firefighting.planner import (  # noqa: E402
    value_raster_from_buses,
)

# This muscle runs in the RolloutWorker process, whose _remember() tabulates the
# firefighter's ragged grid+load sensor mix into a DataFrame. Patch before any
# Memory is touched.
_memory_compat.install()


def _suffix_match(uid: str, suffix: str) -> bool:
    """True if a (possibly env-prefixed) uid ends with ``suffix``."""
    return uid == suffix or uid.endswith("." + suffix) or uid.endswith(suffix)


def _find(sensors: List[SensorInformation], suffix: str):
    for s in sensors:
        if _suffix_match(s.uid, suffix):
            return s
    return None


# Discrete(4) doctrine id -> (IncidentCommand doctrine, protect_assets, needs).
# ``needs`` is the resource class the doctrine relies on; if unavailable this
# step (grounded tankers / no ground resources / no engines), the muscle falls
# back to a no-op so the policy never "spends" a doctrine it cannot execute.
_DOCTRINE_MAP = {
    drl.ACT_NOOP: ("auto", False, "none"),
    drl.ACT_INDIRECT: ("indirect", False, "line"),
    drl.ACT_DIRECT: ("direct", False, "line"),
    drl.ACT_TRIAGE: ("auto", True, "engines"),
}


class LearningFirefighterMuscle(SACMuscle):
    """SAC/CQL muscle that picks a suppression *doctrine* each env step."""

    def __init__(
        self,
        n_planes: int = 3,
        n_helos: int = 0,
        n_crews: int = 0,
        n_dozers: int = 0,
        n_engines: int = 0,
        start_steps: int = 200,
        base_served_mw: float = drl.BASE_SERVED_MW,
        saidi_scale: float = drl.SAIDI_SCALE,
        protect_assets: bool = False,
        grid_json: Optional[str] = None,
        env_step_min: float = 60.0,
        max_steps: int = 60,
        wind_speed: float = 15.0,
        wind_dir_deg: float = 45.0,
        raster_nrows: Optional[int] = None,
        raster_ncols: Optional[int] = None,
        bounds: Optional[List[float]] = None,
        cell_size_m: Optional[float] = None,
    ):
        super().__init__(start_steps=int(start_steps))
        self._n_planes = int(n_planes)
        self._n_helos = int(n_helos)
        self._n_crews = int(n_crews)
        self._n_dozers = int(n_dozers)
        self._n_engines = int(n_engines)
        self._base_served_mw = float(base_served_mw)
        self._saidi_scale = float(saidi_scale)
        self._protect_assets = bool(protect_assets)
        self._grid_json = grid_json
        self._env_step_min = float(env_step_min)
        self._max_steps = int(max_steps)
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
        self._value_raster: Optional[np.ndarray] = None
        self._value_raster_tried = False
        self._prev_saidi = 0.0
        self._cum_customer_min = 0.0
        self._step_i = 0
        # Warm-up exploration previously used the GLOBAL numpy RNG, so
        # replicates of the same experiment file were not reproducible:
        # re-running with the same `seed:` drew different warm-up actions. With
        # start_steps=200 of a 1,200-step phase that is a sixth of training.
        # Seed a private Generator from the muscle's own uid (palaestrAI mints
        # it per rollout worker from the run seed), so a replicate reproduces
        # and different run seeds differ deliberately rather than by accident.
        self._explore_rng = np.random.default_rng(
            abs(hash((str(getattr(self, "uid", "firefighter")), int(start_steps))))
            % (2 ** 32)
        )
        # one IncidentCommand per doctrine intent; rebuilt lazily so a doctrine
        # forces the requested attack while keeping the configured fleet mix.
        self._commanders: dict = {}

    # -- geometry bootstrap (mirrors the scripted firefighter) -------------
    def _ensure_geo(self, sensors: List[SensorInformation]) -> bool:
        if self._shape is not None and self._cell_size_m is not None:
            return True
        shape_s = _find(sensors, "gis.grid_shape")
        if shape_s is not None:
            nr, nc = (int(x) for x in np.asarray(shape_s.value).ravel()[:2])
        elif self._geo["raster_nrows"] and self._geo["raster_ncols"]:
            nr = int(self._geo["raster_nrows"])
            nc = int(self._geo["raster_ncols"])
        else:
            cs = _find(sensors, "gis.cell_state")
            if cs is None:
                LOG.warning("drl firefighter missing grid shape")
                return False
            n = int(np.asarray(cs.value).size)
            nr = nc = int(round(n ** 0.5))

        size_s = _find(sensors, "gis.cell_size_m")
        if size_s is not None:
            delta_m = float(np.asarray(size_s.value).ravel()[0])
        elif self._geo["cell_size_m"] is not None:
            delta_m = float(self._geo["cell_size_m"])
        else:
            bounds_s = _find(sensors, "gis.bounds")
            if bounds_s is not None:
                bounds = tuple(
                    float(x) for x in np.asarray(bounds_s.value).ravel()[:4]
                )
            elif self._geo["bounds"] is not None:
                bounds = tuple(float(x) for x in self._geo["bounds"])
            else:
                from wildfire_cma.gis import SOCAL_BOUNDS

                bounds = tuple(float(x) for x in SOCAL_BOUNDS)
            from wildfire_cma.gis import _approx_cell_size_m

            delta_m = float(_approx_cell_size_m(bounds, nr, nc))

        self._shape = (nr, nc)
        self._cell_size_m = delta_m
        LOG.info(
            "drl firefighter geo: grid=%dx%d delta=%.0fm fleet=%d/%d/%d/%d/%d",
            nr, nc, delta_m, self._n_planes, self._n_helos,
            self._n_crews, self._n_dozers, self._n_engines,
        )
        return True

    def _commander_for(self, doctrine: str, protect: bool) -> IncidentCommand:
        key = (doctrine, protect)
        cmd = self._commanders.get(key)
        if cmd is None:
            cmd = IncidentCommand(
                resources=build_resources(
                    n_planes=self._n_planes, n_helos=self._n_helos,
                    n_crews=self._n_crews, n_dozers=self._n_dozers,
                    n_engines=self._n_engines,
                ),
                doctrine=doctrine,
                protect_assets=protect,
            )
            self._commanders[key] = cmd
        return cmd

    def _mean_slope_deg(self, sensors: List[SensorInformation]) -> float:
        el = _find(sensors, "gis.elevation_m")
        if el is not None and self._shape is not None and self._cell_size_m:
            nr, nc = self._shape
            dem = np.asarray(el.value, dtype=float).reshape(nr, nc)
            return drl.mean_slope_deg(dem, self._cell_size_m)
        return 0.0

    def _ensure_value_raster(self) -> Optional[np.ndarray]:
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
            bus_value: dict = {}
            for b in net.bus.index:
                ll = _bus_lonlat(net, b)
                if ll is not None:
                    bus_lonlat[int(b)] = ll
            for _, row in net.load.iterrows():
                b = int(row["bus"])
                bus_value[b] = bus_value.get(b, 0.0) + float(
                    row.get("p_mw", 0.0)
                )
            bounds = self._geo["bounds"]
            if bounds is None:
                from wildfire_cma.gis import SOCAL_BOUNDS

                bounds = SOCAL_BOUNDS
            driver = DamageMapperDriver(bus_lonlat, bounds, self._shape)
            self._value_raster = value_raster_from_buses(
                self._shape, driver.bus_cell, bus_value
            )
        except Exception as exc:  # pragma: no cover
            LOG.warning(
                "drl firefighter value raster unavailable (%s); triage off",
                exc,
            )
            self._value_raster = None
        return self._value_raster

    # -- served MW / SAIDI from the subscribed grid-load sensors -----------
    def _served_mw(self, sensors: List[SensorInformation]) -> Optional[float]:
        total = 0.0
        found = False
        for s in sensors:
            uid = getattr(s, "uid", "") or ""
            if uid.endswith(".p_mw") and "-load-" in uid:
                v = np.asarray(s.value, dtype=float).ravel()
                if v.size:
                    total += float(v[0])
                    found = True
        return total if found else None

    # -- doctrine index -> resource availability gate ----------------------
    def _doctrine_executable(self, need: str, wind_speed: float) -> bool:
        if need == "none":
            return True
        grounded = is_grounded(float(wind_speed))
        if need == "line":
            # tankers/helos (air) grounded in high wind; crews/dozers are not.
            air = (self._n_planes + self._n_helos) > 0 and not grounded
            ground = (self._n_crews + self._n_dozers) > 0
            return air or ground
        if need == "engines":
            return self._n_engines > 0
        return True

    # -- policy doctrine index --------------------------------------------
    def _policy_doctrine(self, obs: np.ndarray) -> int:
        """Return a Discrete(4) doctrine id from the learned policy or warm-up.

        During TRAIN warm-up (``_actions_proposed < start_steps``) or before a
        model has been transferred, pick a uniform-random doctrine. Otherwise
        query the SAC actor (deterministic in eval / TEST mode).
        """
        warmup = (
            self._actions_proposed < self._start_steps
            and self.mode == Mode.TRAIN
        )
        if warmup or self._model is None:
            return int(self._explore_rng.integers(drl.N_TACTICS))
        import torch as T

        assert self._model.action_type == ActionType.DISCRETE, (
            "LearningFirefighterMuscle requires a Discrete SAC actor"
        )
        obs_t = T.tensor(obs, dtype=T.float, device=self._device)
        a = self._model.act(obs_t, self._mode != Mode.TRAIN)
        idx = int(np.asarray(a).ravel()[0])
        return int(np.clip(idx, 0, drl.N_TACTICS - 1))

    # -- inference ---------------------------------------------------------
    def propose_actions(
        self,
        sensors: List[SensorInformation],
        actuators_available: List[ActuatorInformation],
    ) -> Tuple[List[ActuatorInformation], Any]:
        self._actions_proposed += 1
        self._step_i += 1

        # wind: prefer the live sensor, else the configured fallback.
        wind_speed, wind_dir = self._wind_speed, self._wind_dir_deg
        wf = _find(sensors, "gis.wind_field")
        if wf is not None:
            w = np.asarray(wf.value, dtype=float).ravel()
            if w.size >= 2:
                wind_speed, wind_dir = float(w[0]), float(w[1])

        grid = None
        fuel = None
        dem = None
        if self._ensure_geo(sensors):
            nr, nc = self._shape
            cs = _find(sensors, "gis.cell_state")
            if cs is not None:
                grid = np.asarray(cs.value, dtype=np.int8).reshape(nr, nc)
            fs = _find(sensors, "gis.fuel_class")
            if fs is not None:
                fuel = np.asarray(fs.value, dtype=float).reshape(nr, nc)
            el = _find(sensors, "gis.elevation_m")
            if el is not None:
                dem = np.asarray(el.value, dtype=float).reshape(nr, nc)

        # served MW / cumulative SAIDI from the grid-load sensors (feature 10-12).
        served = self._served_mw(sensors)
        base = self._base_served_mw if self._base_served_mw > 0 else 1.0
        total_customers = max(1.0, base) * drl.CUSTOMERS_PER_MW
        if served is not None:
            disconnected = float(np.clip(base - served, 0.0, base))
            self._cum_customer_min += (
                disconnected * drl.CUSTOMERS_PER_MW * self._env_step_min
            )
        saidi = (
            self._cum_customer_min / total_customers
            if total_customers > 0
            else 0.0
        )
        served_val = served if served is not None else base

        avail = drl.resource_availability(
            n_planes=self._n_planes, n_helos=self._n_helos,
            n_crews=self._n_crews, n_dozers=self._n_dozers,
            n_engines=self._n_engines, wind_speed=wind_speed,
        )

        # build the 17-dim observation (identical call to the offline harvester).
        if grid is None:
            grid = np.zeros(self._shape or (1, 1), dtype=np.int8)
        obs = drl.extract_obs(
            state=grid, fuel=fuel, dem=dem,
            cell_size_m=self._cell_size_m,
            wind_speed=wind_speed, wind_dir_deg=wind_dir,
            served_mw=served_val, base_served_mw=base,
            saidi=saidi, prev_saidi=self._prev_saidi,
            step=self._step_i,
            max_steps=max(1, self._max_steps),
            saidi_scale=self._saidi_scale, **avail,
        )
        self._prev_saidi = saidi

        # policy -> doctrine index -> resource gate.
        act_id = self._policy_doctrine(obs)
        doctrine, protect, need = _DOCTRINE_MAP[act_id]
        if act_id == drl.ACT_NOOP or not self._doctrine_executable(
            need, wind_speed
        ):
            muts: List[Tuple[int, int, int, int]] = []
        else:
            slope_deg = self._mean_slope_deg(sensors)
            value_raster = (
                self._ensure_value_raster()
                if act_id == drl.ACT_TRIAGE
                else None
            )
            cmd = self._commander_for(doctrine, protect)
            muts = cmd.propose(
                grid, fuel, wind_speed, wind_dir,
                slope_deg=slope_deg, value_raster=value_raster,
                step_min=self._env_step_min, cell_m=self._cell_size_m,
            )

        # write the encoded mutation vector onto the cell_mutations actuator.
        vec = spaces.encode_mutations(muts, cap=spaces.CAP)
        for act in actuators_available:
            if _suffix_match(act.uid, "gis.cell_mutations"):
                act(_coerce(vec, act))

        # Return the (obs, action) pair the SAC brain remembers for training.
        # obs is the 17-dim vector (NOT the raw sensor flatten), action is the
        # scalar doctrine index -- matching the brain's Box(17)/Discrete(4) nets.
        return actuators_available, (
            obs.astype(np.float64),
            np.array([act_id], dtype=np.float64),
        )

    def __repr__(self) -> str:
        return (
            f"LearningFirefighterMuscle(uid={self.uid}, "
            f"start_steps={self._start_steps}, "
            f"fleet={self._n_planes}/{self._n_helos}/{self._n_crews}/"
            f"{self._n_dozers}/{self._n_engines})"
        )
