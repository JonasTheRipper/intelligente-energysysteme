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

import logging
import os
from typing import Any, Dict

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


class SocalMidasGridEnvironment(MosaikEnvironment):
    """SoCal MIDAS/mosaik power grid as a palaestrAI environment."""

    def __init__(self, uid: str, **kwargs):
        p = _extract_params(kwargs)

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
        self.reward = _GridHealthReward()
