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

``memory.tail(1).sensor_readings`` is a ``pd.DataFrame`` in real palaestrAI
runs (and a plain list in the unit tests' fake tails), so the lookup handles
both. For the DRL firefighter it is specifically the *one-row object-cell*
frame produced by :mod:`palaestrai_socal.agents._memory_compat`, because that
agent mixes grid rasters with the scalar power sensors.

The reducer is stateful only in that it needs the *previous* step's served MW
to charge a per-step delta; the running SAIDI itself cancels in the delta, so
we track just the cumulative customer-minutes to report a normalised level.
"""
from __future__ import annotations

from typing import List, Optional

import logging

import numpy as np
import pandas as pd

from palaestrai.agent.objective import Objective

LOG = logging.getLogger("palaestrai_socal.agents.saidi_objective")

# same planning figure the environment / reducer use to turn MW -> customers.
CUSTOMERS_PER_MW = 200.0

SAIDI_SCALE = 60.0
BASE_SERVED_MW = 1.0


def _is_load_uid(uid) -> bool:
    """True for the grid-load power sensors we sum (``*-load-*.p_mw``)."""
    uid = str(uid)
    return uid.endswith(".p_mw") and "-load-" in uid


def _first_scalar(value) -> Optional[float]:
    """First usable float in a reading, or None if there is none.

    Accepts everything the two Memory shapes can hand us: a plain number, a
    0-d array, the ``(1,)`` array a scalar sensor carries, or a longer vector.
    Non-numeric and non-finite cells (a uid missing from a concatenated frame
    reads back as NaN) are reported as absent rather than poisoning the sum.
    """
    try:
        flat = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if not flat.size:
        return None
    scalar = float(flat[0])
    return scalar if np.isfinite(scalar) else None


def _served_mw_from_readings(readings) -> Optional[float]:
    """Sum every ``*-load-*.p_mw`` sensor reading -> served MW (or None)."""
    total = 0.0
    found = False
    for r in readings:
        uid = getattr(r, "uid", None) or getattr(r, "sensor_id", None) or ""
        if not _is_load_uid(uid):
            continue
        value = _first_scalar(getattr(r, "value", None))
        if value is not None:
            total += value
            found = True
    return total if found else None


def _served_mw_from_frame(frame: pd.DataFrame) -> Optional[float]:
    """Sum the ``*-load-*.p_mw`` columns of a Memory frame -> served MW.

    Handles both shapes ``memory.tail(1).sensor_readings`` can take: the
    ordinary equal-length frame, whose cells are plain numbers, and the
    one-row object-cell frame the ragged-safe shim
    (:mod:`palaestrai_socal.agents._memory_compat`) produces when the agent
    also subscribes to grid rasters, whose cells hold whole arrays. The most
    recent row is used, so this stays correct if more than one is present.
    """
    if frame.empty:
        return None
    total = 0.0
    found = False
    for uid in frame.columns:
        if not _is_load_uid(uid):
            continue
        value = _first_scalar(frame[uid].iloc[-1])
        if value is not None:
            total += value
            found = True
    return total if found else None


def _served_mw(sensor_readings) -> Optional[float]:
    """Served MW from either Memory representation of one step's readings.

    Deliberately dispatches on type instead of testing truthiness: a real
    ``pd.DataFrame`` raises ``ValueError: The truth value of a DataFrame is
    ambiguous`` on ``bool()``, and iterating one yields column *names*.
    """
    if sensor_readings is None:
        return None
    if isinstance(sensor_readings, pd.DataFrame):
        return _served_mw_from_frame(sensor_readings)

    readings: List = list(sensor_readings)
    # Memory.tail may nest readings one level deep depending on version.
    if readings and isinstance(readings[0], (list, tuple)):
        readings = [r for group in readings for r in group]
    return _served_mw_from_readings(readings)


class SaidiObjective(Objective):
    """Return ``-delta_saidi / scale`` from the agent's grid-load sensors.

    Construction
    ------------
    Both call styles are supported, because palaestrAI's
    :func:`palaestrai.util.dynaloader.load_with_params` unpacks a YAML
    ``params:`` block as **keyword arguments** (``Class(**params)``) rather
    than handing the dict over as a single positional argument::

        SaidiObjective(scale=60.0, base_served_mw=1.0, dt_min=60.0)  # loader
        SaidiObjective(params={"scale": 60.0})                       # direct

    Keys present in an explicit ``params`` dict take precedence over the
    keyword arguments; unknown keys are passed through to the base class.

    Parameters
    ----------
    scale:
        SAIDI normalisation (default 60). Divides the per-step SAIDI delta so
        the reward magnitude is O(1) for the SAC/CQL critics.
    base_served_mw:
        Baseline served load (MW) used as the SAIDI denominator reference. The
        environment's grid serves ``base_served_mw`` when no load is shed;
        default 1.0 matches the normalised-load testbed grid. Pass ``"auto"``
        (or ``null``) to latch it from the first observation instead -- the
        robust choice, since the correct constant depends on exactly which load
        sensors the agent subscribes to.
    dt_min:
        Environment step length in minutes (SAIDI is customer-*minutes*).
        Default 60 (the testbed's ``env_step_min``).
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        *,
        scale: float = SAIDI_SCALE,
        base_served_mw: float = BASE_SERVED_MW,
        dt_min: float = 60.0,
    ):
        settings = {
            "scale": scale,
            "base_served_mw": base_served_mw,
            "dt_min": dt_min,
        }
        if params:
            settings.update(params)
        super().__init__(params=settings)

        self._scale = float(settings["scale"])
        self._dt_min = float(settings["dt_min"])

        # ``base_served_mw: auto`` (or null) defers the baseline to the first
        # observation, matching how analysis.store_readers.read_run derives it
        # (``base_served = served_by_step[0]``). A hard-coded constant has to
        # equal the sum of exactly the load sensors THIS agent subscribes to,
        # which is a coupling nothing checks: too low and the charge is pinned
        # at zero, too high and every step is charged a phantom outage. Both
        # failures are silent. Measured on the Eaton scenario, a nameplate
        # estimate of 241.5 MW against an actual 232.237 MW produced a constant
        # -19.18 reward on a step where nothing had happened.
        base_setting = settings["base_served_mw"]
        self._auto_base = base_setting is None or (
            isinstance(base_setting, str) and base_setting.strip().lower() == "auto"
        )
        self._base_served_mw = 0.0 if self._auto_base else float(base_setting)
        self._total_customers = (
            max(1.0, self._base_served_mw) * CUSTOMERS_PER_MW
        )

    def _resolve_base(self, served: float) -> None:
        """Latch the baseline from the first observation (auto mode only)."""
        self._base_served_mw = float(served)
        self._total_customers = max(1.0, self._base_served_mw) * CUSTOMERS_PER_MW
        self._auto_base = False
        LOG.info(
            "SaidiObjective: baseline served load latched at %.3f MW from the "
            "first observation",
            self._base_served_mw,
        )

    def internal_reward(self, memory, **kwargs) -> float:
        tail = memory.tail(1)
        served = _served_mw(getattr(tail, "sensor_readings", None))
        if served is None:
            # no grid-load sensors this step -> nothing to charge.
            return 0.0

        if self._auto_base:
            # The first observation defines "everything energised": charge
            # nothing for it, then price every later shortfall against it.
            self._resolve_base(served)
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
