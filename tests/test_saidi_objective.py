"""Unit tests for the DRL firefighter's SaidiObjective (numpy-only).

The objective returns ``-delta_saidi / scale`` (<= 0) from the agent's
grid-load (``*-load-*.p_mw``) sensor readings in the latest Memory step. These
tests drive it with a tiny fake Memory/reading shim (no palaestrai Memory
required) to pin: the sign/zero-at-full-service property, linear scaling in the
shortfall, the ``scale`` / ``base_served_mw`` / ``dt_min`` params, that only
``-load-*.p_mw`` sensors are summed, and the graceful zero when no load sensors
are present.
"""

import numpy as np

from palaestrai_socal.agents.saidi_objective import (
    CUSTOMERS_PER_MW, SaidiObjective,
)


class _Reading:
    def __init__(self, uid, value):
        self.uid = uid
        self.value = value


class _Tail:
    def __init__(self, readings):
        self.sensor_readings = readings


class _Memory:
    """Minimal stand-in for palaestrai's agent Memory: only ``tail(1)``."""

    def __init__(self, readings):
        self._readings = readings

    def tail(self, _n=1):
        return _Tail(self._readings)


def _loads(*mw):
    return [
        _Reading(f"Powergrid-0.0-load-{i}-{i}.p_mw", np.array([v]))
        for i, v in enumerate(mw)
    ]


# -- sign / zero-at-full-service ------------------------------------------
def test_zero_reward_when_fully_served():
    obj = SaidiObjective(params={"base_served_mw": 1.0})
    r = obj.internal_reward(_Memory(_loads(1.0)))
    assert r == 0.0


def test_reward_negative_when_load_shed():
    obj = SaidiObjective(params={"base_served_mw": 1.0})
    r = obj.internal_reward(_Memory(_loads(0.5)))
    assert r < 0.0


def test_reward_is_never_positive():
    obj = SaidiObjective(params={"base_served_mw": 1.0})
    # served ABOVE base (fp overshoot) must still clip to 0, not go positive.
    r = obj.internal_reward(_Memory(_loads(1.5)))
    assert r <= 0.0


# -- magnitude / scaling ---------------------------------------------------
def test_reward_matches_saidi_formula():
    scale, base, dt = 60.0, 1.0, 60.0
    obj = SaidiObjective(
        params={"scale": scale, "base_served_mw": base, "dt_min": dt}
    )
    served = 0.25
    r = obj.internal_reward(_Memory(_loads(served)))
    disconnected = base - served
    total_customers = base * CUSTOMERS_PER_MW
    delta_saidi = (disconnected * CUSTOMERS_PER_MW * dt) / total_customers
    expected = -delta_saidi / scale
    assert r == np.float32(expected) or abs(r - expected) < 1e-9


def test_scale_divides_reward():
    kw = {"base_served_mw": 1.0, "dt_min": 60.0}
    r1 = SaidiObjective(params={**kw, "scale": 60.0}).internal_reward(
        _Memory(_loads(0.5))
    )
    r2 = SaidiObjective(params={**kw, "scale": 120.0}).internal_reward(
        _Memory(_loads(0.5))
    )
    assert abs(r1 - 2.0 * r2) < 1e-9


# -- sensor selection / robustness ----------------------------------------
def test_only_load_sensors_summed():
    obj = SaidiObjective(params={"base_served_mw": 1.0})
    readings = _loads(0.4, 0.4) + [
        _Reading("Powergrid-0.0-bus-0.vm_pu", np.array([1.02])),
        _Reading("Powergrid-0.Grid-0.health", np.array([1.0])),
    ]
    # served = 0.8 (only the two -load- sensors), shortfall 0.2 -> r < 0.
    r = obj.internal_reward(_Memory(readings))
    assert r < 0.0
    # a full 1.0 across two loads -> zero reward, ignoring non-load sensors.
    r_full = obj.internal_reward(_Memory(_loads(0.5, 0.5)))
    assert r_full == 0.0


def test_no_load_sensors_returns_zero():
    obj = SaidiObjective(params={"base_served_mw": 1.0})
    readings = [_Reading("Powergrid-0.0-bus-0.vm_pu", np.array([1.0]))]
    assert obj.internal_reward(_Memory(readings)) == 0.0
