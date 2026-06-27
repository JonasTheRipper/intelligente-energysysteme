"""Unit tests for the damage agent's numpy-only driver (DamageMapperDriver).

These exercise the agent-side damage mapper in isolation (no palaestrai / no
pandapower): bus geo->cell co-registration, fire-affected bus detection, the
load-shed latch (monotonic), the controllable-bus subset filter, and the load
actuator-uid bus parser. The thin :class:`DamageMapperMuscle` palaestrai adapter
is covered by the slow two-env smoke run, not here.
"""

import numpy as np
import pytest

from palaestrai_socal.agents.damage_core import (
    DamageMapperDriver, BURNING, BURNED_OUT, UNBURNED,
    load_actuator_bus as _load_bus,
)

# raster: 0..10 lon, 0..10 lat, 10x10 cells -> 1 deg per cell.
BOUNDS = (0.0, 0.0, 10.0, 10.0)
SHAPE = (10, 10)


def _bus_geo():
    # bus 0 at (lon=5, lat=5) -> centre; bus 1 at (lon=1, lat=9) -> NW corner;
    # bus 2 at (lon=9, lat=1) -> SE corner; bus 3 outside the raster bounds.
    return {0: (5.0, 5.0), 1: (1.0, 9.0), 2: (9.0, 1.0), 3: (20.0, 20.0)}


def _grid(*burning_rc):
    g = np.full(SHAPE, UNBURNED, dtype=np.int8)
    for (r, c) in burning_rc:
        g[r, c] = BURNING
    return g


def test_coregistration_skips_out_of_bounds_bus():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    assert 3 not in d.bus_cell                 # outside raster -> dropped
    assert set(d.bus_cell) == {0, 1, 2}


def test_bus_cell_mapping_matches_lonlat_convention():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    # span divisor is (n-1)=9 (matches RasterStack.lonlat_to_rc). row 0 = north.
    # bus 1 (lon=1,lat=9): col=int(0.1*9)=0, row=int(0.1*9)=0.
    assert d.bus_cell[1] == (0, 0)
    # bus 2 (lon=9,lat=1): col=int(0.9*9)=8, row=int(0.9*9)=8.
    assert d.bus_cell[2] == (8, 8)


def test_fire_on_bus_cell_sheds_it():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    r, c = d.bus_cell[0]
    shed = d.evaluate(_grid((r, c)))
    assert 0 in shed
    assert 1 not in shed and 2 not in shed


def test_burned_out_also_counts_as_affected():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    r, c = d.bus_cell[1]
    g = np.full(SHAPE, UNBURNED, dtype=np.int8)
    g[r, c] = BURNED_OUT
    assert 1 in d.evaluate(g)


def test_shed_is_latched_monotonic():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    r0, c0 = d.bus_cell[0]
    d.evaluate(_grid((r0, c0)))            # bus 0 burns
    # next step: bus-0 cell back to UNBURNED (e.g. decoded differently) but the
    # latch keeps it shed; bus 1 now burns and is added.
    r1, c1 = d.bus_cell[1]
    shed = d.evaluate(_grid((r1, c1)))
    assert {0, 1} <= shed                  # 0 stayed shed, 1 added
    assert d.shed_buses == {0, 1}


def test_controllable_subset_filters_evaluation():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    r0, c0 = d.bus_cell[0]
    r1, c1 = d.bus_cell[1]
    # both bus-0 and bus-1 cells burn, but only bus 1 is controllable here.
    shed = d.evaluate(_grid((r0, c0), (r1, c1)), buses={1})
    assert shed == {1}


def test_no_fire_sheds_nothing():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    assert d.evaluate(np.full(SHAPE, UNBURNED, dtype=np.int8)) == set()


@pytest.mark.parametrize("uid,bus", [
    ("Powergrid-0.0-load-0-0.p_mw", 0),
    ("Powergrid-0.0-load-1-1.p_mw", 1),
    ("socal_grid.Powergrid-0.0-load-123-45.p_mw", 123),
    ("Powergrid-0.0-load-7-7.q_mvar", None),     # not a p_mw actuator
    ("Powergrid-0.0-bus-3.vm_pu", None),         # not a load actuator
])
def test_load_bus_parser(uid, bus):
    assert _load_bus(uid) == bus


def test_reset_clears_latch():
    d = DamageMapperDriver(_bus_geo(), BOUNDS, SHAPE)
    r, c = d.bus_cell[0]
    d.evaluate(_grid((r, c)))
    assert d.shed_buses == {0}
    d.reset()
    assert d.shed_buses == set()


# -- dtype-safe actuator coercion (regression for Eaton OutOfActionSpaceError) --
# The Box containment check in palaestrAI wraps the written value into an
# np.ndarray inferring its dtype: a python float / np.float64 written to a
# Box(dtype=np.float32) actuator FAILS containment and raises
# OutOfActionSpaceError. coerce_to_actuator_space() must cast to the actuator's
# own space dtype + shape so containment always holds.
#
# These touch the real palaestrAI Box/ActuatorInformation, so they are skipped
# in the lightweight CI ``unit`` stage (which does not install palaestrai) and
# run in the manual ``system`` stage / locally.
_HAS_PALAESTRAI = True
try:  # pragma: no cover - availability probe
    import palaestrai.types  # noqa: F401
    import palaestrai.agent  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_PALAESTRAI = False

_needs_palaestrai = pytest.mark.skipif(
    not _HAS_PALAESTRAI, reason="palaestrai not installed (lightweight stage)")


@_needs_palaestrai
def test_coerce_to_actuator_space_scalar_float32():
    from palaestrai.types import Box
    from palaestrai.agent import ActuatorInformation
    import palaestrai.agent.util.space_value_utils as svu
    from palaestrai_socal.agents.damage_core import coerce_to_actuator_space

    box = Box(low=0.0, high=2.8107593, shape=(), dtype=np.float32)
    act = ActuatorInformation(space=box, uid="Powergrid-0.0-load-5-0.p_mw")
    val = coerce_to_actuator_space(0.0, act)          # python float in
    assert val.dtype == np.float32
    assert svu._space_contains(box, val)              # the actual crash check
    # a raw python float / np.float64 would FAIL the same check:
    assert not svu._space_contains(box, 0.0)
    assert not svu._space_contains(box, np.float64(0.0))


@_needs_palaestrai
def test_coerce_to_actuator_space_vector_float32():
    from palaestrai.types import Box
    from palaestrai.agent import ActuatorInformation
    import palaestrai.agent.util.space_value_utils as svu
    from palaestrai_socal.agents.damage_core import coerce_to_actuator_space

    box = Box(low=0.0, high=1000.0, shape=(40,), dtype=np.float32)
    act = ActuatorInformation(space=box, uid="gis.cell_mutations")
    vec = np.arange(40, dtype=np.float64)             # float64 vector in
    val = coerce_to_actuator_space(vec, act)
    assert val.dtype == np.float32 and val.shape == (40,)
    assert svu._space_contains(box, val)


@_needs_palaestrai
def test_coerce_to_actuator_space_preserves_float64():
    from palaestrai.types import Box
    from palaestrai.agent import ActuatorInformation
    import palaestrai.agent.util.space_value_utils as svu
    from palaestrai_socal.agents.damage_core import coerce_to_actuator_space

    box = Box(low=0.0, high=10.0, shape=(), dtype=np.float64)
    act = ActuatorInformation(space=box, uid="x")
    val = coerce_to_actuator_space(0.0, act)
    assert val.dtype == np.float64
    assert svu._space_contains(box, val)
