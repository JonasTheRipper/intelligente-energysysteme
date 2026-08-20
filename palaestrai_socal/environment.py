"""palaestrAI environment wrapping the SoCal grid + wildfire CMA.

This turns the Southern California MIDAS co-simulation environment into a
first-class :class:`palaestrai.environment.environment.Environment` so it can be
driven by the arsenAI / palaestrAI experiment runner.

GUARDIAN framing
----------------
The wildfire is modelled as a **Constrained Mutation operator** of the grid
(the GUARDIAN four-tuple ``(S, tau, D, Theta)``), *not* as a learning agent.
The Overseer-Adversary chooses the parameter vector ``Theta`` (ignition
point(s), wind, fuel moisture, global ROS multiplier ``kappa``); the cellular
automaton ``tau`` advances the fire state ``S``; the damage mapper ``D`` mutates
the pandapower grid ``G_t = G (+) dG_t``; a power flow is solved on the mutated
topology; and the disconnected-customer count drives the reward.

Interface
---------
* **Sensors** (observation): grid + fire telemetry --
  ``min_bus_vm_pu``, ``mean_bus_vm_pu``, ``customers_connected``,
  ``customers_disconnected``, ``saidi_minutes``, ``fire_front_cells``,
  ``fire_affected_cells``, ``failed_buses``, ``failed_lines``,
  ``wind_speed_m_per_s``, ``wind_dir_deg``, ``grid_served_mw``,
  ``pf_converged``.
* **Actuators** (the Overseer-Adversary action = ``Theta``):
  ``ignition_lon``, ``ignition_lat``, ``kappa``, ``dead_fuel_moisture``,
  ``wind_speed``, ``wind_dir_deg``. The wind actuators default to the
  NOAA-sourced values but may be overridden by the adversary.
* **Reward**: ``customers_disconnected`` (the adversary maximises grid harm).

The environment is self-contained and offline-capable: it builds a synthetic
SoCal raster when no LANDFIRE/3DEP rasters are supplied, and reads NOAA weather
from the same CSV the MIDAS ``weather`` simulator consumes.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("palaestrai_socal.environment")

# -- palaestrAI imports -----------------------------------------------------
from palaestrai.environment.environment import Environment
from palaestrai.environment.environment_baseline import EnvironmentBaseline
from palaestrai.environment.environment_state import EnvironmentState
from palaestrai.agent import (
    SensorInformation,
    ActuatorInformation,
    RewardInformation,
)
from palaestrai.types import Box

# -- wildfire CMA -----------------------------------------------------------
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_cma.cma import (  # noqa: E402
    BURNED_OUT,
    BURNING,
    RasterStack,
    Theta,
    WildfireCMA,
)
from wildfire_cma.damage import DamageMapper  # noqa: E402
from wildfire_cma.gis import (  # noqa: E402
    SOCAL_BOUNDS, synthetic_socal, socal_from_srtm,
)

# Reuse the proven SoCal dispatch / convergence recipe ("config D").
sys.path.insert(0, os.path.join(_ROOT, "socal_grid"))
import dispatch_and_run as dar  # noqa: E402


# Average customers (meters) served per MW of peak load in CA investor-owned
# utilities (rough planning figure ~ 1 customer per 5 kW peak -> 200/MW). Used
# to translate disconnected load MW into a customer count for SAIDI-style KPIs.
CUSTOMERS_PER_MW = 200.0


def _load_grid(grid_json: str):
    import pandapower as pp

    return pp.from_json(grid_json)


class SoCalWildfireEnvironment(Environment):
    """SoCal grid + wildfire CMA as a palaestrAI environment.

    Parameters (via ``params`` kwarg from the experiment file)
    ----------------------------------------------------------
    grid_json:
        path to the SoCal pandapower JSON (default: bundled model).
    weather_csv:
        NOAA weather CSV (MIDAS schema) used to seed wind sensors/actuators.
    raster_nrows, raster_ncols:
        synthetic raster resolution when no real rasters are supplied.
    dt_cma_min:
        CMA timestep in minutes.
    env_step_min:
        wall-clock minutes advanced per environment ``update`` (the CMA runs
        ``env_step_min / dt_cma_min`` sub-steps each environment step).
    clearance_m:
        radiant-heat line-failure clearance buffer.
    max_steps:
        episode length in environment steps (e.g. 5 days at 60-min steps = 120).
    seed:
        RNG seed for the stochastic ignition draws.
    """

    def __init__(self, uid: str, *args, **kwargs):
        super().__init__(uid, *args, **kwargs)
        p = dict(kwargs.get("params", {}) or {})

        self.grid_json = p.get(
            "grid_json",
            os.path.join(_ROOT, "socal_grid", "socal_grid.json"),
        )
        self.weather_csv = p.get(
            "weather_csv",
            os.path.expanduser(
                "~/.config/midas/midas_data/socal_noaa_weather.csv"
            ),
        )
        self.raster_nrows = int(p.get("raster_nrows", 600))
        self.raster_ncols = int(p.get("raster_ncols", 760))
        self.dt_cma_min = float(p.get("dt_cma_min", 5.0))
        self.env_step_min = float(p.get("env_step_min", 60.0))
        self.clearance_m = float(p.get("clearance_m", 120.0))
        self.t_burn_steps = int(p.get("t_burn_steps", 6))
        self.max_steps = int(p.get("max_steps", 120))
        self.seed = int(p.get("seed", 0))
        # use the real SRTM GL3 DEM mosaic if cached (data/dem/*.npz); falls
        # back to the deterministic synthetic terrain when unavailable.
        self.use_real_dem = bool(p.get("use_real_dem", True))

        # default ignition: dense LA-basin cluster (Eaton-fire-like origin)
        self.default_ignition = tuple(
            p.get("default_ignition", (-118.13, 34.19))
        )

        # populated in start_environment()
        self._net = None
        self._net0 = None  # pristine copy
        self._raster: Optional[RasterStack] = None
        self._cma: Optional[WildfireCMA] = None
        self._damage: Optional[DamageMapper] = None
        self._weather = None
        self._step = 0
        self._cum_customer_minutes = 0.0
        self._total_customers = 0.0
        self._peak_load_mw = 0.0

    # -- terrain ------------------------------------------------------------
    def _build_raster(self):
        """Build the fuel+DEM raster: real SRTM GL3 if cached, else synthetic."""
        if self.use_real_dem:
            try:
                r = socal_from_srtm(
                    nrows=self.raster_nrows, ncols=self.raster_ncols,
                    bounds=SOCAL_BOUNDS, seed=self.seed or 7,
                )
                LOG.info("Using REAL SRTM GL3 terrain (elev %.0f..%.0f m)",
                         float(r.dem.min()), float(r.dem.max()))
                return r
            except FileNotFoundError as exc:
                # Deliberate graceful degradation here (unlike GisWorldEnvironment,
                # which hard-fails): the README documents that this driver runs
                # with or without the git-ignored DEM binary. Log it loudly --
                # the synthetic raster has a different fuel map and a different
                # set of class-9 house cells, so results are NOT comparable.
                LOG.warning(
                    "Real DEM unavailable (%s); falling back to the SYNTHETIC "
                    "raster -- fuel map and house cells differ from a real-DEM "
                    "run, so KPIs from the two are not comparable", exc,
                )
        r = synthetic_socal(
            nrows=self.raster_nrows, ncols=self.raster_ncols, seed=self.seed or 7
        )
        LOG.info("Raster source: %s", getattr(r, "source", "unknown"))
        return r

    # -- weather ------------------------------------------------------------
    def _load_weather(self):
        try:
            import pandas as pd

            if os.path.exists(self.weather_csv):
                df = pd.read_csv(self.weather_csv)
                return df
        except Exception as e:  # pragma: no cover
            LOG.warning("could not load NOAA weather %s: %s", self.weather_csv, e)
        return None

    def _wind_at(self, step: int) -> Tuple[float, float]:
        """Return (wind_speed_m_per_s, wind_dir_deg) for an env step."""
        if self._weather is None or len(self._weather) == 0:
            # deterministic Santa-Ana fallback
            return 15.0, 45.0
        # env_step_min relative to hourly NOAA rows
        hour = int((step * self.env_step_min) // 60) % len(self._weather)
        row = self._weather.iloc[hour]
        spd = float(row.get("wind_v_m_per_s", 15.0))
        ddeg = float(row.get("wind_dir_deg", 45.0))
        return spd, ddeg

    # -- spaces -------------------------------------------------------------
    def _sensor_specs(self):
        # (uid, low, high)
        return [
            ("min_bus_vm_pu", 0.0, 2.0),
            ("mean_bus_vm_pu", 0.0, 2.0),
            ("customers_connected", 0.0, 5.0e7),
            ("customers_disconnected", 0.0, 5.0e7),
            ("saidi_minutes", 0.0, 1.0e6),
            ("fire_front_cells", 0.0, 1.0e6),
            ("fire_affected_cells", 0.0, 1.0e6),
            ("failed_buses", 0.0, 1.0e4),
            ("failed_lines", 0.0, 1.0e4),
            ("wind_speed_m_per_s", 0.0, 60.0),
            ("wind_dir_deg", 0.0, 360.0),
            ("grid_served_mw", 0.0, 1.0e5),
            ("pf_converged", 0.0, 1.0),
        ]

    def _actuator_specs(self):
        minlon, minlat, maxlon, maxlat = SOCAL_BOUNDS
        return [
            ("ignition_lon", minlon, maxlon),
            ("ignition_lat", minlat, maxlat),
            ("kappa", 1.0, 8.0),
            ("dead_fuel_moisture", 0.01, 0.40),
            ("wind_speed", 0.0, 60.0),
            ("wind_dir_deg", 0.0, 360.0),
        ]

    def _make_sensors(self, values: dict) -> List[SensorInformation]:
        out = []
        for (uid, lo, hi) in self._sensor_specs():
            v = float(values.get(uid, 0.0))
            v = float(np.clip(v, lo, hi))  # keep telemetry inside its space
            out.append(
                SensorInformation(
                    value=np.array([v], dtype=np.float64),
                    space=Box(low=lo, high=hi, shape=(1,), dtype=np.float64),
                    uid=uid,
                )
            )
        return out

    def _make_actuators(self) -> List[ActuatorInformation]:
        out = []
        spd, ddeg = self._wind_at(0)
        defaults = {
            "ignition_lon": self.default_ignition[0],
            "ignition_lat": self.default_ignition[1],
            "kappa": 1.5,
            "dead_fuel_moisture": 0.05,
            "wind_speed": spd,
            "wind_dir_deg": ddeg,
        }
        for (uid, lo, hi) in self._actuator_specs():
            v = float(np.clip(defaults[uid], lo, hi))
            out.append(
                ActuatorInformation(
                    value=np.array([v], dtype=np.float64),
                    space=Box(low=lo, high=hi, shape=(1,), dtype=np.float64),
                    uid=uid,
                )
            )
        return out

    # -- lifecycle ----------------------------------------------------------
    def start_environment(self) -> EnvironmentBaseline:
        LOG.info("starting SoCalWildfireEnvironment %s", self.uid)
        import pandapower as pp

        # Build a *converging, balanced* baseline grid using the proven SoCal
        # dispatch recipe ("config D"): strengthen the equivalent network,
        # co-locate generation at each load bus, and solve with Iwamoto-NR
        # load continuation. The wildfire then mutates THIS dispatched grid.
        self._net = _load_grid(self.grid_json)
        ok = dar.run(self._net, verbose=False)
        if not ok or not self._net.get("converged", False):
            LOG.warning("baseline dispatch did not converge; KPIs may saturate")
        self._base_served_mw = (
            float(self._net.res_load.p_mw[self._net.load.in_service].sum())
            if len(self._net.res_load) else 0.0
        )

        self._raster = self._build_raster()
        self._weather = self._load_weather()
        self._step = 0
        self._cum_customer_minutes = 0.0

        # total customers ~ (served) load * customers/MW, using the balanced
        # baseline dispatch as the reference "all customers connected" state.
        self._peak_load_mw = float(self._net.load[self._net.load.in_service].p_mw.sum())
        self._total_customers = max(1.0, self._base_served_mw) * CUSTOMERS_PER_MW

        # CMA seeded with default Theta (overridden by the adversary on update)
        spd, ddeg = self._wind_at(0)
        theta0 = Theta(
            ignition_points=[tuple(self.default_ignition)],
            wind_speed=spd,
            wind_dir_deg=ddeg,
            dead_fuel_moisture=0.05,
            kappa=1.5,
        )
        self._cma = WildfireCMA(
            self._raster,
            theta0,
            dt_cma_min=self.dt_cma_min,
            t_burn_steps=self.t_burn_steps,
            seed=self.seed,
        )
        self._damage = DamageMapper(
            self._net, self._raster, clearance_m=self.clearance_m
        )

        self.sensors = self._make_sensors({})
        self.actuators = self._make_actuators()
        return EnvironmentBaseline(
            sensors_available=self.sensors,
            actuators_available=self.actuators,
            static_world_model={
                "grid_json": self.grid_json,
                "bounds": list(SOCAL_BOUNDS),
                "peak_load_mw": self._peak_load_mw,
                "total_customers": self._total_customers,
            },
        )

    # -- step ---------------------------------------------------------------
    def _apply_theta(self, actuators: List[ActuatorInformation]) -> Theta:
        a = {}
        for act in actuators or []:
            v = act.value
            try:
                v = float(np.asarray(v).ravel()[0])
            except Exception:
                v = float(v)
            a[act.uid] = v
        spd_def, ddeg_def = self._wind_at(self._step)
        return Theta(
            ignition_points=[(
                a.get("ignition_lon", self.default_ignition[0]),
                a.get("ignition_lat", self.default_ignition[1]),
            )],
            wind_speed=a.get("wind_speed", spd_def),
            wind_dir_deg=a.get("wind_dir_deg", ddeg_def),
            dead_fuel_moisture=a.get("dead_fuel_moisture", 0.05),
            kappa=a.get("kappa", 1.5),
        ).clamp()

    def _runpp(self) -> Tuple[bool, float, float, float]:
        """Solve the (mutated) grid. Returns (converged, min_vm, mean_vm, served_mw).

        Warm-starts from the previous converged state (``init='results'``) since
        the wildfire removes assets from an already-balanced dispatch; falls
        back to a fresh dc-init Iwamoto-NR if the topology change is severe
        enough to break the warm start (e.g. an islanded sub-network).
        """
        import pandapower as pp

        converged = False
        have_results = len(self._net.res_bus) > 0
        attempts = (
            [dict(algorithm="iwamoto_nr", init="results", max_iteration=100)]
            if have_results else []
        ) + [
            dict(algorithm="iwamoto_nr", init="dc", max_iteration=150),
            dict(algorithm="nr", init="dc", max_iteration=150),
        ]
        for kw in attempts:
            try:
                pp.runpp(self._net, calculate_voltage_angles=True,
                         enforce_q_lims=False, **kw)
                if bool(self._net["converged"]):
                    converged = True
                    break
            except Exception as e:
                LOG.debug("runpp %s failed: %s", kw.get("init"), e)
        if converged and len(self._net.res_bus):
            in_serv = self._net.bus.in_service.values
            vm = self._net.res_bus.vm_pu.values[in_serv]
            vm = vm[~np.isnan(vm)]
            min_vm = float(np.min(vm)) if len(vm) else 0.0
            mean_vm = float(np.mean(vm)) if len(vm) else 0.0
            served = float(
                self._net.res_load.p_mw[self._net.load.in_service].sum()
            ) if len(self._net.res_load) else 0.0
        else:
            min_vm = mean_vm = served = 0.0
        return converged, min_vm, mean_vm, served

    def update(self, actuators: List[ActuatorInformation]) -> EnvironmentState:
        self._step += 1

        # 1) Overseer-Adversary sets Theta; reseed the CMA's control vector.
        theta = self._apply_theta(actuators)
        self._cma.theta = theta
        # (re)ignite any new ignition points from Theta (idempotent on existing)
        for (lon, lat) in theta.ignition_points:
            self._cma.ignite_lonlat(lon, lat)

        # 2) advance the cellular automaton tau by one environment step
        self._cma.advance(minutes=self.env_step_min)

        # 3) damage mapper D: mutate the grid G_t = G (+) dG_t
        ds = self._damage.evaluate(self._cma)
        self._damage.apply(self._net)

        # 4) power flow on the mutated topology
        converged, min_vm, mean_vm, served_mw = self._runpp()

        # 5) KPIs: disconnected load -> customers; accumulate SAIDI minutes.
        # The fire disconnects load two ways: (a) loads taken out of service on
        # burned buses, and (b) load that can no longer be served because the
        # network islanded. We measure the served-load shortfall against the
        # balanced baseline dispatch. If the post-mutation PF fails to converge
        # (severe islanding), we fall back to the out-of-service load directly so
        # the KPI stays meaningful instead of saturating at 100%.
        oos_load_mw = float(
            self._net.load.p_mw[~self._net.load.in_service].sum()
        ) if len(self._net.load) else 0.0
        if converged:
            disconnected_mw = max(0.0, self._base_served_mw - served_mw)
        else:
            # PF did not converge on the mutated topology: attribute the
            # de-energised load to the assets the fire removed.
            disconnected_mw = max(oos_load_mw, 0.0)
        disconnected_mw = min(disconnected_mw, self._base_served_mw)
        customers_disconnected = disconnected_mw * CUSTOMERS_PER_MW
        customers_connected = max(0.0, self._total_customers - customers_disconnected)
        self._cum_customer_minutes += customers_disconnected * self.env_step_min
        saidi_minutes = (
            self._cum_customer_minutes / self._total_customers
            if self._total_customers > 0
            else 0.0
        )

        fstats = self._cma.stats()
        sensor_values = {
            "min_bus_vm_pu": min_vm,
            "mean_bus_vm_pu": mean_vm,
            "customers_connected": customers_connected,
            "customers_disconnected": customers_disconnected,
            "saidi_minutes": saidi_minutes,
            "fire_front_cells": fstats.get("front_size", 0),
            "fire_affected_cells": fstats.get("affected_cells", 0),
            "failed_buses": len(ds.failed_buses),
            "failed_lines": len(ds.failed_lines),
            "wind_speed_m_per_s": theta.wind_speed,
            "wind_dir_deg": theta.wind_dir_deg,
            "grid_served_mw": served_mw,
            "pf_converged": 1.0 if converged else 0.0,
        }
        self.sensors = self._make_sensors(sensor_values)

        # reward: the adversary maximises disconnected customers
        rewards = [
            RewardInformation(
                value=np.array(
                    [float(np.clip(customers_disconnected, 0.0, 5.0e7))],
                    dtype=np.float64,
                ),
                space=Box(low=0.0, high=5.0e7, shape=(1,), dtype=np.float64),
                uid="customers_disconnected",
            )
        ]

        done = self._step >= self.max_steps

        return EnvironmentState(
            sensor_information=self.sensors,
            rewards=rewards,
            done=done,
            world_state=sensor_values,
        )
