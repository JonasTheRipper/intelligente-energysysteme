"""SoCal MIDAS power-grid environment (REAL palaestrai_mosaik co-simulation).

This is the v0.2 replacement for the v0.1 hand-rolled pandapower ``_runpp()``
environment. The power grid is now computed by a **real MIDAS scenario stepped
by mosaik** inside the palaestrAI experiment, exactly as the user required.

``SocalMidasGridEnvironment`` is a thin subclass of
``palaestrai_mosaik.MosaikEnvironment`` that injects the MIDAS wiring so the
experiment YAML only has to supply simple params:

* ``module``           = ``midas_palaestrai.descriptor:Descriptor``
* ``description_func`` = ``describe``   (-> sensors, actuators, world_state)
* ``instance_func``    = ``get_world``  (async -> mosaik World, entities)

The Descriptor runs ``midas.api.run(name, params, config, no_build=True,
no_run=True)`` with ``params["with_arl"]=True``, which builds the SoCal mosaik
World and auto-exposes every powergrid element as a palaestrAI sensor/actuator
(uids ``Powergrid-0.0-<element>.<attr>``). The DamageMapperAgent later drives
``...-load-<eid>-<idx>.p_mw`` actuators to shed fire-affected load.

The conductor instantiates environments as ``Class(**params, uid=, seed=,
broker_uri=)`` (the YAML ``params:`` block is flattened into kwargs), so this
``__init__`` reads from kwargs and forwards the MIDAS keyword-only args to the
MosaikEnvironment base.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Any, Dict, List, Optional

from palaestrai_mosaik import MosaikEnvironment

LOG = logging.getLogger("palaestrai_socal.midas_grid_env")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_SCENARIO = os.path.join(_ROOT, "midas_socal", "socal.yml")
DEFAULT_GRIDFILE = os.path.join(_ROOT, "midas_socal", "socal_grid_midas.json")
DEFAULT_SCENARIO_NAME = "socal"
DEFAULT_START_DATE = "2025-01-07 00:00:00+0000"


def _extract_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Support both nested ``params={...}`` and conductor-flattened kwargs."""
    if isinstance(kwargs.get("params"), dict):
        return dict(kwargs["params"])
    return dict(kwargs)


class _GridHealthReward:
    """Continuous grid-health reward for the SoCal MIDAS environment.

    ``MosaikEnvironment.update`` requires a callable ``self.reward``. The native
    ``NoExtGridHealthReward`` wraps the ``Powergrid-0.Grid-0.health`` value in a
    ``Discrete(2)`` space, but midas-powergrid reports health as a *float* (e.g.
    ``1.0079``), which raises ``Value not contained in Discrete(2)``. This reward
    reads the same ``...health`` sensor and returns it in a continuous ``Box`` so
    it round-trips for any float health value.
    """

    def __init__(self, health_uid: str = "Powergrid-0.Grid-0.health"):
        self._uid = health_uid

    def __call__(self, sensors, actuators=None, *args, **kwargs):
        import numpy as np
        from palaestrai.agent import RewardInformation
        from palaestrai.types import Box

        value = 0.0
        for s in sensors:
            if s.uid == self._uid or s.uid.endswith(".health"):
                try:
                    value = float(np.asarray(s.value).ravel()[0])
                except (TypeError, ValueError, IndexError):
                    value = float(s.value)
                break
        return [
            RewardInformation(
                np.array([value], dtype=np.float64),
                Box(low=-1.0e9, high=1.0e9, shape=(1,), dtype=np.float64),
                "grid_health_reward",
            )
        ]


# ---------------------------------------------------------------------------
# ARL sensor/actuator whitelist
# ---------------------------------------------------------------------------
# MIDAS exposes EVERY attribute of every grid element to palaestrAI: for this
# net, 66,223 sensors and 11,250 actuators. Each becomes its own ARLSensor /
# ARLActuator mosaik entity with its own dataflow edge, and every environment
# step palaestrAI rebuilds all 66,223 SensorInformation objects, pickles them
# (7.5 MB) and ships them onward. The agents subscribe to about twenty.
#
# Measured on a 13-step run of the M1 smoke experiment (startup excluded):
#     as shipped                          38.3 s/step
#     + StoreDumpTrimmer                  12.5 s/step
#     + this whitelist (~1,900 sensors)    1.85 s/step
#
# The filter is applied where palaestrAI builds ``env.sensors``, BEFORE
# ``_start_mosaik`` wires the entities, so the removed ones are never created
# at all -- this is not just a store trim.
#
# IMPORTANT: filtering here also removes those sensors from the STORE, so the
# patterns must cover everything the analysis tier reads:
#   *-load-*.p_mw    served MW / SAIDI, and the damage mapper's actuators
#   *-bus-*.vm_pu    vmin/vmean grid-metrics panels
#   *-line-*.p_from_mw  real intertie flow (else a labelled proxy is used)
#   *Grid-0.health   the environment's own reward
# Leave ``keep_sensors`` unset to restore the unfiltered behaviour exactly.
_PATCHED_FLAG = "_socal_whitelisted"


def _install_whitelist() -> None:
    """Wrap ``palaestrai_mosaik.util.load_sensors_and_actuators`` (idempotent)."""
    from palaestrai_mosaik import util as _mutil

    if getattr(_mutil.load_sensors_and_actuators, _PATCHED_FLAG, False):
        return
    _original = _mutil.load_sensors_and_actuators

    def _filtered(env, description_fnc):
        world_state = _original(env, description_fnc)
        pats = getattr(env, "_keep_sensor_patterns", None)
        apats = getattr(env, "_keep_actuator_patterns", None)
        if pats:
            before = len(env.sensors)
            env.sensors = [
                s for s in env.sensors
                if any(fnmatch.fnmatch(s.uid, pat) for pat in pats)
            ]
            LOG.info(
                "%s: ARL sensor whitelist %d -> %d (%.1f%% removed)",
                env.uid, before, len(env.sensors),
                100.0 * (1 - len(env.sensors) / max(1, before)),
            )
            if not env.sensors:
                raise ValueError(
                    f"keep_sensors matched no sensor for {env.uid}; patterns "
                    f"{list(pats)} would leave the environment with none."
                )
        if apats:
            before = len(env.actuators)
            env.actuators = [
                a for a in env.actuators
                if any(fnmatch.fnmatch(a.uid, pat) for pat in apats)
            ]
            LOG.info(
                "%s: ARL actuator whitelist %d -> %d",
                env.uid, before, len(env.actuators),
            )
            if not env.actuators:
                raise ValueError(
                    f"keep_actuators matched no actuator for {env.uid}; "
                    f"patterns {list(apats)} would leave the environment with "
                    "none, and palaestrAI requires at least one."
                )
        return world_state

    _filtered._socal_whitelisted = True
    _mutil.load_sensors_and_actuators = _filtered


# Installed at MODULE IMPORT, not in __init__, and this matters: the
# environment is constructed by the EnvironmentConductor in one process, then
# attached to the SimulationController, which is where start_environment() --
# and therefore load_sensors_and_actuators() -- actually runs. A patch applied
# in __init__ lands in the wrong process and silently does nothing (verified:
# the stored dump still carried all 10,686 grid sensors). Any process that
# unpickles a SocalMidasGridEnvironment must import this module to resolve the
# class, so patching here reaches all of them. The wrapper is a no-op for
# environments without keep patterns, so this is safe to install unconditionally.
_install_whitelist()


class SocalMidasGridEnvironment(MosaikEnvironment):
    """SoCal MIDAS/mosaik power grid as a palaestrAI environment."""

    def __init__(self, uid: str, **kwargs):
        p = _extract_params(kwargs)

        # fnmatch patterns; unset => no filtering (behaviour unchanged)
        keep_sensors: Optional[List[str]] = p.pop("keep_sensors", None)
        keep_actuators: Optional[List[str]] = p.pop("keep_actuators", None)
        kwargs.pop("keep_sensors", None)
        kwargs.pop("keep_actuators", None)

        seed = int(kwargs.get("seed", p.get("seed", 0)))
        step_size = int(p.get("step_size", 3600))      # seconds (60 min)
        max_steps = int(p.get("max_steps", 120))
        end_sec = int(p.get("end", max_steps * step_size))
        # arl_sync_freq must be < end; one sync per environment step.
        arl_sync_freq = int(p.get("arl_sync_freq", step_size))
        start_date = str(p.get("start_date", DEFAULT_START_DATE))

        scenario_name = str(p.get("scenario_name", DEFAULT_SCENARIO_NAME))
        scenario_cfg = p.get("scenario", DEFAULT_SCENARIO)
        config_list = scenario_cfg if isinstance(scenario_cfg, list) else [scenario_cfg]

        midas_params: Dict[str, Any] = {
            "name": scenario_name,
            "config": config_list,
            "silent": bool(p.get("silent", True)),
        }
        # optional offline guard: skip the (cached) MIDAS data download
        if "skip_download" in p:
            midas_params["skip_download"] = bool(p["skip_download"])

        LOG.info(
            "SocalMidasGridEnvironment %s: scenario=%s end=%ss step=%ss",
            uid, scenario_name, end_sec, step_size,
        )

        super().__init__(
            uid,
            seed=seed,
            module="midas_palaestrai.descriptor:Descriptor",
            description_func="describe",
            instance_func="get_world",
            arl_sync_freq=arl_sync_freq,
            end=end_sec,
            start_date=start_date,
            silent=bool(p.get("silent", True)),
            simulation_timeout=int(p.get("simulation_timeout", 120)),
            params=midas_params,
        )

        # MosaikEnvironment.update() unconditionally calls ``self.reward(...)``;
        # the base leaves it None unless the experiment file supplies a
        # ``reward:`` block. Default to a continuous grid-health reward so the
        # env is self-contained. A ``reward:`` key in the YAML still overrides
        # this (the conductor sets ``environment.reward`` after construction).
        # These travel with the pickled instance to the process that runs
        # start_environment(); the module-level patch above reads them there.
        self._keep_sensor_patterns = keep_sensors
        self._keep_actuator_patterns = keep_actuators

        self.reward = _GridHealthReward()
