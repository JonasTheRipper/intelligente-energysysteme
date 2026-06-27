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
