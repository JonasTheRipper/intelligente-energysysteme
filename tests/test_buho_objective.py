"""Unit tests for BurnedHousesObjective and MooObjective.

Both read their inputs out of ``memory.tail(1)``, so they are driven here with
a small fake Memory shim covering the shapes a real tail can take. As in
``tests/test_saidi_objective.py``, ``palaestrai.agent.objective.Objective`` is
stubbed when palaestrai is absent so these stay in the fast ``unit`` CI stage.

The properties pinned here are the ones that make the objectives *composable*:
every axis is a charge (<= 0), the house charge is an increment rather than a
level, and MooObjective normalises before weighting so a SAIDI-sized term is
not swamped by a house-sized one.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

try:  # pragma: no cover - availability probe
    import palaestrai.agent.objective  # noqa: F401
except ImportError:
    _agent_mod = types.ModuleType("palaestrai.agent")
    _objective_mod = types.ModuleType("palaestrai.agent.objective")

    class _StubObjective:
        def __init__(self, params=None):
            self.params = params

    _objective_mod.Objective = _StubObjective
    _agent_mod.objective = _objective_mod
    sys.modules.setdefault("palaestrai", types.ModuleType("palaestrai"))
    sys.modules["palaestrai.agent"] = _agent_mod
    sys.modules["palaestrai.agent.objective"] = _objective_mod

from palaestrai_socal.agents import objective_support as osup  # noqa: E402
from palaestrai_socal.agents.buho_objective import (  # noqa: E402
    BurnedHousesObjective,
)

pytestmark = pytest.mark.unit


class _Info:
    def __init__(self, uid, value):
        self.uid = uid
        self.value = value


class _Tail:
    def __init__(self, readings):
        self.sensor_readings = readings


class _Memory:
    """Minimal stand-in for palaestrai's agent Memory: only ``tail(1)``."""

    def __init__(self, readings=None):
        self._tail = _Tail(readings if readings is not None else [])

    def tail(self, n=1):
        return self._tail

    def set(self, readings):
        self._tail = _Tail(readings)


def _house_readings(total, burned_step, burned_total=None, prefix="gis_world."):
    out = [
        _Info(f"{prefix}gis.houses_total", np.array([float(total)])),
        _Info(f"{prefix}gis.houses_burned_this_step", np.array([float(burned_step)])),
    ]
    if burned_total is not None:
        out.append(
            _Info(f"{prefix}gis.houses_burned_total", np.array([float(burned_total)]))
        )
    return out


# --------------------------------------------------------------------------
# objective_support.read_scalar
# --------------------------------------------------------------------------
def test_read_scalar_matches_env_prefixed_uid():
    tail = _Tail([_Info("gis_world.gis.houses_total", np.array([42.0]))])
    assert osup.read_scalar(tail, "gis.houses_total") == 42.0
    assert osup.read_scalar(tail, "gis.houses_burned_total") is None


def test_read_scalar_handles_a_dataframe_tail():
    """Stock palaestrAI hands over a frame, not a list of readings."""
    tail = _Tail(pd.DataFrame({"gis_world.gis.houses_total": [7.0]}))
    assert osup.read_scalar(tail, "gis.houses_total") == 7.0


def test_read_scalar_handles_the_ragged_object_cell_frame():
    """The shape _memory_compat produces once rasters and scalars mix."""
    frame = pd.DataFrame(
        {"gis_world.gis.houses_total": [np.array([9.0])]}, dtype=object
    )
    assert osup.read_scalar(frame_tail := _Tail(frame), "gis.houses_total") == 9.0
    assert osup.read_scalar(frame_tail, "gis.nope") is None


def test_read_scalar_treats_nan_as_absent():
    tail = _Tail([_Info("gis.houses_total", np.array([np.nan]))])
    assert osup.read_scalar(tail, "gis.houses_total") is None


# --------------------------------------------------------------------------
# BurnedHousesObjective
# --------------------------------------------------------------------------
def test_reward_is_zero_when_nothing_burns():
    obj = BurnedHousesObjective()
    assert obj.internal_reward(_Memory(_house_readings(1000, 0))) == 0.0


def test_reward_is_a_negative_charge_proportional_to_the_loss():
    obj = BurnedHousesObjective(scale=0.02)
    # 10 of 1000 houses = 1% of the settlement, half of the 2% "bad step"
    r = obj.internal_reward(_Memory(_house_readings(1000, 10)))
    assert r == pytest.approx(-0.5)
    # twice the loss, twice the charge -- linear, no clipping in range
    r2 = obj.internal_reward(_Memory(_house_readings(1000, 20)))
    assert r2 == pytest.approx(-1.0)


def test_reward_charges_the_increment_not_the_running_total():
    """The distinguishing property vs a survival level.

    Holding a burned footprint steady must cost nothing: the agent is charged
    where the loss happens, so a do-nothing policy cannot accumulate reward for
    houses that were never threatened.
    """
    obj = BurnedHousesObjective(scale=0.02)
    mem = _Memory(_house_readings(1000, 10, burned_total=10))
    first = obj.internal_reward(mem)
    mem.set(_house_readings(1000, 0, burned_total=10))
    second = obj.internal_reward(mem)
    assert first < 0.0
    assert second == 0.0


def test_survival_fraction_tracks_the_cumulative_sensor():
    obj = BurnedHousesObjective()
    assert obj.survival_fraction() == 1.0          # before any telemetry
    obj.internal_reward(_Memory(_house_readings(200, 5, burned_total=50)))
    assert obj.survival_fraction() == pytest.approx(0.75)


def test_reward_is_zero_when_the_raster_holds_no_houses():
    """A fine grid may contain no settlement; that must not divide by zero."""
    obj = BurnedHousesObjective()
    assert obj.internal_reward(_Memory(_house_readings(0, 0))) == 0.0


def test_reward_is_zero_and_warns_once_without_the_sensors(caplog):
    obj = BurnedHousesObjective()
    mem = _Memory([_Info("socal_grid.Powergrid-0.0-load-1-1.p_mw", np.array([1.0]))])
    with caplog.at_level("WARNING"):
        assert obj.internal_reward(mem) == 0.0
        assert obj.internal_reward(mem) == 0.0
    assert sum("BurnedHousesObjective" in r.message for r in caplog.records) == 1


def test_reward_never_goes_positive():
    """Guards the composability contract: every axis is a charge."""
    obj = BurnedHousesObjective()
    for burned in (0, 1, 10, 10_000):
        assert obj.internal_reward(_Memory(_house_readings(100, burned))) <= 0.0


# --------------------------------------------------------------------------
# MooObjective
# --------------------------------------------------------------------------
def _moo():
    from palaestrai_socal.agents.moo_objective import MooObjective

    return MooObjective


def test_moo_rejects_degenerate_weights():
    MooObjective = _moo()
    with pytest.raises(ValueError):
        MooObjective(alpha=-0.5, beta=1.0)
    with pytest.raises(ValueError):
        MooObjective(alpha=0.0, beta=0.0)


def test_moo_default_weights_sum_to_one():
    """A silent 2/3 rescale would shift any objective-based stop threshold."""
    obj = _moo()()
    assert obj.alpha + obj.beta == pytest.approx(1.0)


def test_moo_normalisation_keeps_the_saidi_axis_visible():
    """The whole point of the normalisers, at CALIBRATED magnitudes.

    Both numbers below are taken from the Eaton scenario rather than invented:

    * served 0.99945 of ``base_served_mw`` reproduces the documented
      no-firefighting per-step SAIDI charge of ~-5.5e-4 (the 60-step baseline
      accrues ~1.97 SAIDI). Raw, that is ~1e-4.
    * 2 of 101 house cells is ~2% of the Eaton window's settlement in one
      hour -- a bad step, i.e. the ``scale`` the house axis is defined against.

    Unnormalised those differ by ~1000:1 and equal weights would be a lie.
    After normalisation the two weighted contributions land within an order of
    magnitude of each other, which is what makes alpha/beta mean something.
    """
    obj = _moo()(alpha=0.5, beta=0.5)
    readings = _house_readings(101, 2)
    readings.append(
        _Info("socal_grid.Powergrid-0.0-load-1-1.p_mw", np.array([0.99945]))
    )
    total = obj.internal_reward(_Memory(readings))

    terms = obj.last_terms
    assert terms["saidi"]["raw"] < 0.0
    assert abs(terms["saidi"]["normalised"]) > abs(terms["saidi"]["raw"])
    ratio = abs(terms["houses"]["weighted"]) / abs(terms["saidi"]["weighted"])
    assert 0.1 < ratio < 10.0, f"axes still incommensurable (ratio {ratio:.1f})"
    assert total == pytest.approx(sum(t["weighted"] for t in terms.values()))


def test_moo_total_is_never_positive_and_is_zero_when_nothing_happens():
    obj = _moo()()
    assert obj.internal_reward(_Memory(_house_readings(1000, 0))) == 0.0


def test_moo_reuses_its_subobjectives():
    """Rebuilding per call would reset BUHO's reporting state every step."""
    obj = _moo()()
    first = obj.burned_houses_objective
    obj.internal_reward(_Memory(_house_readings(1000, 5, burned_total=5)))
    assert obj.burned_houses_objective is first
    assert obj.burned_houses_objective.survival_fraction() == pytest.approx(0.995)


def test_moo_running_shares_report_axis_balance():
    obj = _moo()()
    obj.internal_reward(_Memory(_house_readings(1000, 10)))
    shares = obj.running_shares()
    assert shares["houses"] == pytest.approx(1.0)   # only the house axis fired
    assert sum(shares.values()) == pytest.approx(1.0)
