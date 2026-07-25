"""Unit tests for the DRL firefighter's SaidiObjective.

The objective returns ``-delta_saidi / scale`` (<= 0) from the agent's
grid-load (``*-load-*.p_mw``) sensor readings in the latest Memory step. Most
tests drive it with a tiny fake Memory/reading shim (no palaestrai Memory
required) to pin: the sign/zero-at-full-service property, linear scaling in the
shortfall, the ``scale`` / ``base_served_mw`` / ``dt_min`` params, that only
``-load-*.p_mw`` sensors are summed, and the graceful zero when no load sensors
are present.

Those fakes hand ``tail.sensor_readings`` over as a plain *list*, which is not
what palaestrAI does: a real ``Memory.tail(1)`` returns a ``pd.DataFrame``. The
final section therefore drives the objective through the genuine
``palaestrai.agent.memory.Memory`` so the DataFrame path is covered too, in
both shapes it can take -- the ordinary equal-length frame, and the one-row
object-cell frame :mod:`palaestrai_socal.agents._memory_compat` produces once
the agent also subscribes to grid rasters.
"""

import numpy as np
import pandas as pd
import pytest

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


# -- palaestrAI loader construction contract -------------------------------
def test_kwargs_construction_matches_loader():
    """``load_with_params(module, params)`` calls ``Class(**params)``.

    A YAML ``params:`` block is therefore unpacked as **keyword arguments**
    rather than handed over as a single dict, so the objective must accept both
    shapes and build an identical reward function from either. Constructing
    with kwargs used to raise ``TypeError: __init__() got an unexpected keyword
    argument 'scale'``, which killed every phase at agent setup.
    """
    # verbatim from experiment_eaton_firefighting.yml's objective params.
    params = {"scale": 60.0, "base_served_mw": 1.0, "dt_min": 60.0}
    shed = _Memory(_loads(0.5))

    from_loader = SaidiObjective(**params)     # what the loader does
    from_dict = SaidiObjective(params=params)  # direct construction

    # guard against a vacuous 0.0 == 0.0 comparison.
    assert from_loader.internal_reward(shed) < 0.0
    assert from_loader.internal_reward(shed) == from_dict.internal_reward(shed)

    # every keyword must actually be wired through, not silently ignored.
    tuned = {"scale": 120.0, "base_served_mw": 2.0, "dt_min": 30.0}
    assert (
        SaidiObjective(**tuned).internal_reward(shed)
        == SaidiObjective(params=tuned).internal_reward(shed)
        != from_loader.internal_reward(shed)
    )

    # and the no-argument form still yields the documented defaults.
    assert (
        SaidiObjective().internal_reward(shed)
        == SaidiObjective(params={}).internal_reward(shed)
        == from_loader.internal_reward(shed)
    )


# -- real Memory DataFrame path --------------------------------------------
# The objective used to do ``list(tail.sensor_readings or [])``, which raises
# "The truth value of a DataFrame is ambiguous" on every real step -- and would
# have iterated column *names* even without the ``or``. Only the list-based
# fakes above ever worked, so these pin the shape palaestrAI actually delivers.

# scale 60, base 1.0 MW, dt 60 min -> one full MW shed costs exactly 1.0.
_FRAME_PARAMS = {"scale": 60.0, "base_served_mw": 1.0, "dt_min": 60.0}


@pytest.fixture
def numpy_nan_alias(monkeypatch):
    """palaestrAI 3.5.9 reads ``np.NAN``, which numpy 2 removed.

    Production pins numpy 1.26, so restore the alias rather than skip the real
    ``Memory.__getitem__`` code path on newer numpy.
    """
    monkeypatch.setattr(np, "NAN", np.nan, raising=False)


def _real_memory_with_loads(*mw, rasters=False):
    """A genuine palaestrAI ``Memory`` holding exactly one full step.

    Built through the real ``Memory.append``, so ``tail(1).sensor_readings`` is
    the actual DataFrame palaestrAI hands the objective at runtime rather than
    a hand-rolled stand-in. ``rasters=True`` adds the grid sensors that make
    the columns ragged, which is the DRL firefighter's real subscription.
    """
    memory_mod = pytest.importorskip(
        "palaestrai.agent.memory",
        reason="palaestrAI not installed; no real Memory to frame",
    )
    sensor_mod = pytest.importorskip("palaestrai.agent.sensor_information")
    types_mod = pytest.importorskip("palaestrai.types")
    sensor_information = sensor_mod.SensorInformation
    box = types_mod.Box

    def _sensor(uid, value):
        # a real SensorInformation requires a space and asserts the value fits.
        return sensor_information(
            value=value,
            space=box(
                low=np.full(value.shape, -1e6, dtype=np.float64),
                high=np.full(value.shape, 1e6, dtype=np.float64),
            ),
            uid=uid,
        )

    readings = [
        _sensor(f"Powergrid-0.0-load-{i}-{i}.p_mw", np.array([v]))
        for i, v in enumerate(mw)
    ]
    if rasters:
        readings += [
            _sensor("gis.cell_state", np.zeros(23660, dtype=np.float64)),
            _sensor("gis.wind_field", np.array([15.0, 45.0])),
        ]

    memory = memory_mod.Memory()
    memory.append(
        "firefighter",
        sensor_readings=readings,
        actuator_setpoints=[],
        rewards=[_sensor("reward", np.array([0.0]))],
        done=False,
        observations=np.zeros(17),
        actions=np.zeros(1),
        objective=np.array([0.0]),
    )
    return memory


def test_internal_reward_from_real_dataframe_tail_full_service(
    numpy_nan_alias,
):
    """Fully served load through a real Memory frame costs nothing."""
    memory = _real_memory_with_loads(0.6, 0.4)

    frame = memory.tail(1).sensor_readings
    assert isinstance(frame, pd.DataFrame), "not exercising the DataFrame path"

    assert SaidiObjective(**_FRAME_PARAMS).internal_reward(memory) == 0.0


def test_internal_reward_from_real_dataframe_tail_load_shed(numpy_nan_alias):
    """Shedding load through a real Memory frame charges the SAIDI delta."""
    memory = _real_memory_with_loads(0.25)

    frame = memory.tail(1).sensor_readings
    assert isinstance(frame, pd.DataFrame)
    # ordinary equal-length frame: plain numeric cells, no arrays.
    assert not any(
        isinstance(frame[col].iloc[-1], np.ndarray) for col in frame.columns
    )

    # 0.75 MW shed * 200 customers/MW * 60 min / 200 customers = 45 SAIDI min,
    # over scale 60 -> exactly -0.75.
    reward = SaidiObjective(**_FRAME_PARAMS).internal_reward(memory)
    assert reward == pytest.approx(-0.75)


def test_internal_reward_from_ragged_dataframe_tail(numpy_nan_alias):
    """The firefighter's raster+load mix reaches the objective intact.

    Grid rasters make the columns ragged, so ``_memory_compat`` returns a
    one-row frame of object cells holding whole arrays. The scalar power
    sensors must survive that as ``(1,)`` arrays -- an earlier ``.at``-based
    fill flattened them to 0-d scalars -- and the rasters must not be summed
    as if they were load.
    """
    pytest.importorskip("palaestrai.agent.memory")
    from palaestrai_socal.agents import _memory_compat

    _memory_compat.install()

    memory = _real_memory_with_loads(0.3, 0.2, rasters=True)
    frame = memory.tail(1).sensor_readings

    # confirm we really are on the ragged object-cell path, not the plain one.
    assert len(frame) == 1
    assert isinstance(frame["gis.cell_state"].iloc[0], np.ndarray)
    assert frame["gis.cell_state"].iloc[0].shape == (23660,)
    assert frame["Powergrid-0.0-load-0-0.p_mw"].iloc[0].shape == (1,)

    # served 0.5 MW -> same 0.5 MW shortfall as a plain frame would give.
    reward = SaidiObjective(**_FRAME_PARAMS).internal_reward(memory)
    assert reward == pytest.approx(-0.5)
    assert reward == SaidiObjective(**_FRAME_PARAMS).internal_reward(
        _real_memory_with_loads(0.3, 0.2)
    )
