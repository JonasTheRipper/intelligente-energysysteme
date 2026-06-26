"""System test: the SoCal wildfire palaestrAI environment runs end-to-end.

Verifies:
  1. The environment subclasses palaestrai.Environment and instantiates.
  2. ``start_environment()`` returns an EnvironmentBaseline with the expected
     sensor / actuator spaces.
  3. ``update(actuators)`` advances the fire CMA, mutates the grid, solves a
     power flow, and returns an EnvironmentState whose reward (disconnected
     customers) grows as the fire spreads into the grid.
  4. The episode terminates after ``max_steps``.

Run: python3 tests/test_palaestrai_env.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from palaestrai.agent import ActuatorInformation  # noqa: E402
from palaestrai.environment.environment import Environment  # noqa: E402
from palaestrai.types import Box  # noqa: E402

from palaestrai_socal.environment import SoCalWildfireEnvironment  # noqa: E402


def _make_env(max_steps=6):
    return SoCalWildfireEnvironment(
        uid="socal-wildfire-test",
        params={
            "raster_nrows": 600,
            "raster_ncols": 760,
            "dt_cma_min": 5.0,
            "env_step_min": 60.0,
            "clearance_m": 120.0,
            # ignite over the dense LA-basin cluster used by the damage test
            "default_ignition": (-118.246, 33.804),
            "max_steps": max_steps,
            "seed": 1,
        },
    )


def test_subclass_and_construct():
    env = _make_env()
    assert isinstance(env, Environment)
    print("constructed", env.uid)


def test_start_environment_baseline():
    env = _make_env()
    baseline = env.start_environment()
    sensors = baseline.sensors_available
    actuators = baseline.actuators_available
    s_uids = {s.uid for s in sensors}
    a_uids = {a.uid for a in actuators}
    print("sensors:", sorted(s_uids))
    print("actuators:", sorted(a_uids))
    assert {"min_bus_vm_pu", "customers_disconnected", "fire_front_cells",
            "failed_buses", "pf_converged"} <= s_uids
    assert {"ignition_lon", "ignition_lat", "kappa", "dead_fuel_moisture",
            "wind_speed", "wind_dir_deg"} <= a_uids
    swm = baseline.static_world_model
    print("peak_load_mw:", round(swm["peak_load_mw"]),
          "total_customers:", round(swm["total_customers"]))
    assert swm["peak_load_mw"] > 1000


def _adversary_actuators(env, lon, lat, kappa=2.0, fm=0.04,
                         wind=18.0, wdir=45.0):
    spec = {
        "ignition_lon": lon, "ignition_lat": lat, "kappa": kappa,
        "dead_fuel_moisture": fm, "wind_speed": wind, "wind_dir_deg": wdir,
    }
    out = []
    for uid, val in spec.items():
        out.append(ActuatorInformation(
            value=np.array([float(val)], dtype=np.float64),
            space=Box(low=-1e9, high=1e9, shape=(1,), dtype=np.float64),
            uid=uid,
        ))
    return out


def test_update_spreads_and_damages_grid():
    env = _make_env(max_steps=8)
    env.start_environment()
    acts = _adversary_actuators(env, -118.246, 33.804, kappa=2.5,
                                fm=0.04, wind=20.0, wdir=45.0)

    last = None
    rewards = []
    fronts = []
    for step in range(8):
        state = env.update(acts)
        ws = state.world_state
        rewards.append(ws["customers_disconnected"])
        fronts.append(ws["fire_affected_cells"])
        last = state
        print(
            f"step {step+1}: front={ws['fire_affected_cells']:>5} "
            f"failed_bus={ws['failed_buses']:>3} failed_line={ws['failed_lines']:>3} "
            f"disc_cust={ws['customers_disconnected']:>12,.0f} "
            f"served_mw={ws['grid_served_mw']:>10,.1f} "
            f"conv={int(ws['pf_converged'])}"
        )

    # fire grows monotonically (CA states never un-burn)
    assert fronts[-1] >= fronts[0]
    assert fronts[-1] > 5, "fire should spread over 8 hours"
    # the grid must eventually lose assets
    assert last.world_state["failed_buses"] > 0
    # reward (disconnected customers) is non-negative and should appear once the
    # fire reaches the grid
    assert max(rewards) > 0, "fire reaching the LA cluster must disconnect load"
    # reward object plumbing
    assert last.rewards and last.rewards[0].uid == "customers_disconnected"


def test_episode_terminates():
    env = _make_env(max_steps=3)
    env.start_environment()
    acts = _adversary_actuators(env, -118.246, 33.804)
    dones = [env.update(acts).done for _ in range(3)]
    print("done flags:", dones)
    assert dones[-1] is True
    assert dones[0] is False


if __name__ == "__main__":
    for fn in [
        test_subclass_and_construct,
        test_start_environment_baseline,
        test_update_spreads_and_damages_grid,
        test_episode_terminates,
    ]:
        print(f"\n=== {fn.__name__} ===")
        fn()
        print(f"PASS {fn.__name__}")
    print("\nALL PALAESTRAI ENV TESTS PASSED")
