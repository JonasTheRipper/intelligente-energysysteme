"""SaidiObjective -- the DRL firefighter's utility function (v0.7).

The Deep-RL firefighter *minimises* cumulative SAIDI (System Average
Interruption Duration Index). palaestrAI maximises the objective's
``internal_reward``, so we return the **negative** SAIDI accrued this step::

    reward = -delta_saidi / SAIDI_SCALE        (<= 0)

Each step the agent is charged the SAIDI minutes that accrued during the step;
holding the fire (and thus keeping load energised) accrues zero, so a policy
that maximises return is exactly a policy that minimises total SAIDI.

Deriving SAIDI from the agent's Memory
--------------------------------------
The firefighter subscribes to the power-grid load sensors (``*-load-*.p_mw``).
We recover *served MW* this step by summing them (identical to
:class:`midas_socal.grid_kpis.GridKpiReducer`), then convert the shortfall vs a
configured ``base_served_mw`` into the SAIDI increment using the same planning
constant the environment uses (``CUSTOMERS_PER_MW = 200``). This keeps the
online reward consistent with the offline teacher transitions harvested from
the store (which use the identical formula), so the CQL bootstrap and the SAC
fine-tuning optimise the same quantity.

The reducer is stateful only in that it needs the *previous* step's served MW
to charge a per-step delta; the running SAIDI itself cancels in the delta, so
we track just the cumulative customer-minutes to report a normalised level.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from palaestrai.agent.objective import Objective

# same planning figure the environment / reducer use to turn MW -> customers.
CUSTOMERS_PER_MW = 200.0

SAIDI_SCALE = 60.0
BASE_SERVED_MW = 1.0


def _served_mw_from_readings(readings) -> Optional[float]:
    """Sum every ``*-load-*.p_mw`` sensor reading -> served MW (or None)."""
    total = 0.0
    found = False
    for r in readings:
        uid = getattr(r, "uid", None) or getattr(r, "sensor_id", None) or ""
        if uid.endswith(".p_mw") and "-load-" in uid:
            try:
                val = np.asarray(r.value, dtype=float).ravel()
                if val.size:
                    total += float(val[0])
                    found = True
            except (TypeError, ValueError):
                pass
    return total if found else None


class SaidiObjective(Objective):
    """Return ``-delta_saidi / scale`` from the agent's grid-load sensors.

    Parameters (via ``params``)
    ---------------------------
    scale:
        SAIDI normalisation (default 60). Divides the per-step SAIDI delta so
        the reward magnitude is O(1) for the SAC/CQL critics.
    base_served_mw:
        Baseline served load (MW) used as the SAIDI denominator reference. The
        environment's grid serves ``base_served_mw`` when no load is shed;
        default 1.0 matches the normalised-load testbed grid.
    dt_min:
        Environment step length in minutes (SAIDI is customer-*minutes*).
        Default 60 (the testbed's ``env_step_min``).
    """

    def __init__(self, params: Optional[dict] = None):
        params = {} if params is None else params
        super().__init__(params=params)
        self._scale = float(params.get("scale", SAIDI_SCALE))
        self._base_served_mw = float(
            params.get("base_served_mw", BASE_SERVED_MW)
        )
        self._dt_min = float(params.get("dt_min", 60.0))
        self._total_customers = (
            max(1.0, self._base_served_mw) * CUSTOMERS_PER_MW
        )

    def internal_reward(self, memory, **kwargs) -> float:
        tail = memory.tail(1)
        readings = list(getattr(tail, "sensor_readings", []) or [])
        # Memory.tail may nest readings one level deep depending on version.
        if readings and isinstance(readings[0], (list, tuple)):
            flat: List = []
            for grp in readings:
                flat.extend(grp)
            readings = flat
        served = _served_mw_from_readings(readings)
        if served is None:
            # no grid-load sensors this step -> nothing to charge.
            return 0.0

        base = self._base_served_mw if self._base_served_mw > 0 else 1.0
        disconnected_mw = float(np.clip(base - served, 0.0, base))
        customers_disconnected = disconnected_mw * CUSTOMERS_PER_MW
        # SAIDI minutes accrued THIS step (customer-minutes / total customers).
        delta_saidi = (
            customers_disconnected * self._dt_min
        ) / self._total_customers
        reward = -float(delta_saidi) / (self._scale if self._scale else 1.0)
        # reward is <= 0 by construction; clip tiny positive fp noise to 0.
        return min(0.0, reward)
