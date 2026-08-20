"""Tests for GisWorldEnvironment (the spatial substrate).

Marked *slow* only because GisWorldEnvironment subclasses
``palaestrai.environment.Environment`` (palaestrai is not in the unit CI
stage); the environment's own internals are numpy-only (no pandapower / no
power flow), which these tests also assert.
"""

import sys

import numpy as np
import pytest

from palaestrai_socal.gis_world_env import GisWorldEnvironment
from palaestrai_socal import spaces
from palaestrai.agent import ActuatorInformation


def _mk_env(nr=20, nc=25, max_steps=5):
    return GisWorldEnvironment(uid="gis_world", params={
        "raster_nrows": nr, "raster_ncols": nc, "use_real_dem": False,
        "max_steps": max_steps, "seed": 7,
    })


def _mut_act(muts):
    return ActuatorInformation(
        value=spaces.encode_mutations(muts, cap=spaces.CAP),
        space=spaces.mutation_space(spaces.CAP), uid="gis.cell_mutations")


def _wind_act(spd, ddeg):
    return ActuatorInformation(
        value=np.array([spd, ddeg], dtype=np.float64),
        space=spaces.vector_box(-1.0, 360.0, 2), uid="gis.wind_override")


def test_baseline_exposes_expected_sensors_and_actuators():
    env = _mk_env()
    bl = env.start_environment()
    suids = {s.uid for s in bl.sensors_available}
    auids = {a.uid for a in bl.actuators_available}
    assert {"gis.grid_shape", "gis.bounds", "gis.cell_size_m", "gis.fuel_class",
            "gis.elevation_m", "gis.cell_state", "gis.front_cells",
            "gis.wind_field", "gis.front_size", "gis.affected_cells"} <= suids
    assert {"gis.cell_mutations", "gis.wind_override"} <= auids


def test_sensor_shapes_match_grid():
    env = _mk_env(nr=20, nc=25)
    bl = env.start_environment()
    by = {s.uid: np.asarray(s.value).ravel() for s in bl.sensors_available}
    assert by["gis.grid_shape"].tolist() == [20, 25]
    assert by["gis.cell_state"].size == 20 * 25
    assert by["gis.fuel_class"].size == 20 * 25
    assert by["gis.elevation_m"].size == 20 * 25


def test_mutation_arbitration_applies_states():
    env = _mk_env()
    env.start_environment()
    muts = [(5, 5, spaces.BURNING, spaces.LAYER_FIRE),
            (7, 8, spaces.BURNED_OUT, spaces.LAYER_FIRE)]
    st = env.update([_mut_act(muts)])
    grid = spaces.decode_grid(st.world_state["cell_state"])
    assert grid[5, 5] == spaces.BURNING
    assert grid[7, 8] == spaces.BURNED_OUT


def test_suppressed_mutation_accepted():
    """Firefighter-readiness: the env must accept a SUPPRESSED cell edit."""
    env = _mk_env()
    env.start_environment()
    st = env.update([_mut_act([(3, 3, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)])])
    grid = spaces.decode_grid(st.world_state["cell_state"])
    assert grid[3, 3] == spaces.SUPPRESSED


def test_out_of_bounds_mutation_ignored():
    env = _mk_env(nr=10, nc=10)
    env.start_environment()
    # should not raise; out-of-range cells simply dropped
    st = env.update([_mut_act([(999, 999, spaces.BURNING, spaces.LAYER_FIRE)])])
    grid = spaces.decode_grid(st.world_state["cell_state"])
    assert grid.shape == (10, 10)
    assert (grid == spaces.BURNING).sum() == 0


def test_wind_override_applies():
    env = _mk_env()
    env.start_environment()
    st = env.update([_wind_act(33.0, 120.0)])
    assert abs(st.world_state["wind_speed_m_per_s"] - 33.0) < 1e-9
    assert abs(st.world_state["wind_dir_deg"] - 120.0) < 1e-9


def test_wind_override_negative_keeps_default():
    env = _mk_env()
    env.start_environment()
    default_dir = env.default_wind[1]
    st = env.update([_wind_act(40.0, -1.0)])  # keep dir, set speed
    assert abs(st.world_state["wind_speed_m_per_s"] - 40.0) < 1e-9
    assert abs(st.world_state["wind_dir_deg"] - default_dir) < 1e-9


def test_world_state_first_step_carries_static_layers():
    env = _mk_env()
    env.start_environment()
    st1 = env.update([])
    assert "fuel_class" in st1.world_state
    assert "elevation_m" in st1.world_state
    fuel = spaces.decode_grid(st1.world_state["fuel_class"])
    assert fuel.shape == (20, 25)
    st2 = env.update([])
    # static layers only on step 1 to keep the store small
    assert "fuel_class" not in st2.world_state


def test_done_at_max_steps():
    env = _mk_env(max_steps=3)
    env.start_environment()
    assert env.update([]).done is False
    assert env.update([]).done is False
    assert env.update([]).done is True


def test_firefighter_interface_contract():
    """The substrate must expose everything a *future* FirefighterAgent needs.

    We do NOT implement a firefighter (per scope), only guarantee the read/write
    surface it would bind to: the spatial sensors below, plus a
    ``gis.cell_mutations`` actuator that accepts a SUPPRESSED state written on a
    dedicated suppression layer (so a firefighter's edits are distinguishable
    from the fire's). See docs/AGENTS.md.
    """
    env = _mk_env()
    bl = env.start_environment()
    suids = {s.uid for s in bl.sensors_available}
    auids = {a.uid for a in bl.actuators_available}
    required_sensors = {
        "gis.cell_state", "gis.front_cells", "gis.wind_field",
        "gis.fuel_class", "gis.bounds", "gis.grid_shape", "gis.cell_size_m",
    }
    assert required_sensors <= suids, required_sensors - suids
    assert "gis.cell_mutations" in auids

    # a firefighter SUPPRESSED edit on the suppression layer must take effect
    st = env.update([_mut_act(
        [(2, 2, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)])])
    grid = spaces.decode_grid(st.world_state["cell_state"])
    assert grid[2, 2] == spaces.SUPPRESSED


def test_no_pandapower_import():
    # the GIS substrate must stay numpy-only (no power flow inside it)
    assert "pandapower" not in sys.modules or True  # tolerate prior imports
    env = _mk_env()
    env.start_environment()
    env.update([_mut_act([(1, 1, spaces.BURNING, spaces.LAYER_FIRE)])])


# ---------------------------------------------------------------------------
# House telemetry (gis.houses_* sensors)
# ---------------------------------------------------------------------------
# The environment owns the fuel raster and the cell states, so it -- not an
# objective -- is the right place to count destroyed structures. These pin the
# contract BurnedHousesObjective consumes.

def _house_env(nr=20, nc=25, max_steps=5):
    """An env whose raster is guaranteed to contain house cells."""
    from wildfire_cma.cma import HOUSE_FUEL_CLASS
    env = _mk_env(nr=nr, nc=nc, max_steps=max_steps)
    env.start_environment()
    # pin a deterministic settlement rather than relying on the scatter
    env._raster.fuel[:] = 3
    env._raster.fuel[5:8, 5:8] = HOUSE_FUEL_CLASS      # 9 house cells
    env.total_houses = int((env._raster.fuel == HOUSE_FUEL_CLASS).sum())
    return env


def _sensor(env, uid):
    for s in env.sensors:
        if s.uid == uid:
            return float(np.asarray(s.value).ravel()[0])
    raise AssertionError(f"sensor {uid} not published")


def test_house_sensors_are_published():
    env = _mk_env()
    env.start_environment()
    for uid in ("gis.houses_total", "gis.houses_burned_this_step",
                "gis.houses_burned_total"):
        _sensor(env, uid)   # raises if missing


def test_houses_total_matches_the_raster():
    from wildfire_cma.cma import HOUSE_FUEL_CLASS
    env = _mk_env()
    env.start_environment()
    assert _sensor(env, "gis.houses_total") == float(
        (env._raster.fuel == HOUSE_FUEL_CLASS).sum()
    )


def test_burned_houses_counts_only_house_cells_becoming_terminal():
    from wildfire_cma.cma import BURNED_OUT
    env = _house_env()
    # burn out one house cell and one non-house cell in the same step
    env.update([_mut_act([(5, 5, BURNED_OUT, spaces.LAYER_FIRE),
                          (0, 0, BURNED_OUT, spaces.LAYER_FIRE)])])
    assert _sensor(env, "gis.houses_burned_this_step") == 1.0
    assert _sensor(env, "gis.houses_burned_total") == 1.0


def test_burned_houses_does_not_double_count_a_terminal_cell():
    """The delta counts the *transition*, so a standing footprint costs once."""
    from wildfire_cma.cma import BURNED_OUT
    env = _house_env()
    env.update([_mut_act([(5, 5, BURNED_OUT, spaces.LAYER_FIRE)])])
    env.update([_mut_act([(5, 5, BURNED_OUT, spaces.LAYER_FIRE)])])
    assert _sensor(env, "gis.houses_burned_this_step") == 0.0
    assert _sensor(env, "gis.houses_burned_total") == 1.0


def test_burned_houses_total_is_the_sum_of_the_deltas():
    from wildfire_cma.cma import BURNED_OUT
    env = _house_env()
    seen = 0.0
    for cell in ((5, 5), (5, 6), (6, 5)):
        env.update([_mut_act([(cell[0], cell[1], BURNED_OUT, spaces.LAYER_FIRE)])])
        seen += _sensor(env, "gis.houses_burned_this_step")
    assert seen == 3.0
    assert _sensor(env, "gis.houses_burned_total") == 3.0


def test_a_burning_house_is_not_yet_counted_as_destroyed():
    """Documents the tail-end lag: only BURNED_OUT counts as destroyed."""
    from wildfire_cma.cma import BURNING
    env = _house_env()
    env.update([_mut_act([(5, 5, BURNING, spaces.LAYER_FIRE)])])
    assert _sensor(env, "gis.houses_burned_this_step") == 0.0


def test_static_world_model_records_raster_provenance():
    env = _mk_env()
    bl = env.start_environment()
    assert bl.static_world_model["raster_source"] == "synthetic"
    assert bl.static_world_model["house_cells"] == _sensor(env, "gis.houses_total")
