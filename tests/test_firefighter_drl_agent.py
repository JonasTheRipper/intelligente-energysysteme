"""Adapter tests for the v0.7 LearningFirefighterMuscle (SAC/CQL inference).

These touch the real palaestrAI ``SensorInformation`` / ``ActuatorInformation``
types and the hARL ``SACMuscle`` base, so (like ``test_firefighter_agent.py``)
they are skipped in the lightweight CI ``unit`` stage and run in the manual
``system`` stage / locally.

With no model transferred (``_model is None``) the muscle is in warm-up: it
picks a uniform-random doctrine. That is enough to pin the wiring WITHOUT a
trained actor: the muscle builds the exact 17-dim contract observation, maps the
Discrete(4) doctrine through ``_DOCTRINE_MAP`` onto IncidentCommand, gates
doctrines it cannot execute, emits a dtype-correct ``gis.cell_mutations`` vector,
recovers served MW from the ``-load-*.p_mw`` sensors, and returns the
``(obs, [action])`` pair the SAC brain remembers.
"""

import numpy as np
import pytest

_HAS_STACK = True
try:  # pragma: no cover - availability probe
    import palaestrai.agent  # noqa: F401
    import palaestrai.types  # noqa: F401
    import harl  # noqa: F401
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_STACK = False

_needs_stack = pytest.mark.skipif(
    not _HAS_STACK, reason="palaestrai/harl/torch not installed (light stage)"
)

NR, NC = 20, 20
BOUNDS = [-118.1771, 34.1469, -117.9981, 34.2528]
CELL_M = 50.0


def _sensors(state, *, served=(1.0,), with_geo=True):
    from palaestrai.agent import SensorInformation
    from palaestrai_socal import spaces

    fuel = np.full((NR, NC), 3, dtype=np.float64)
    dem = np.zeros((NR, NC), dtype=np.float64)
    out = [
        SensorInformation(
            value=state.astype(np.float64).ravel(),
            space=spaces.vector_box(0.0, 16.0, NR * NC),
            uid="gis.cell_state",
        ),
        SensorInformation(
            value=fuel.ravel(),
            space=spaces.vector_box(0.0, 64.0, NR * NC),
            uid="gis.fuel_class",
        ),
        SensorInformation(
            value=dem.ravel(),
            space=spaces.vector_box(-500.0, 5000.0, NR * NC),
            uid="gis.elevation_m",
        ),
    ]
    if with_geo:
        out += [
            SensorInformation(
                value=np.array([NR, NC], dtype=np.float64),
                space=spaces.vector_box(0.0, 4096.0, 2),
                uid="gis.grid_shape",
            ),
            SensorInformation(
                value=np.array([CELL_M], dtype=np.float64),
                space=spaces.vector_box(0.0, 5000.0, 1),
                uid="gis.cell_size_m",
            ),
        ]
    for i, mw in enumerate(served):
        out.append(
            SensorInformation(
                value=np.array([mw]),
                space=spaces.vector_box(0.0, 10.0, 1),
                uid=f"Powergrid-0.0-load-{i}-{i}.p_mw",
            )
        )
    return out


def _actuator():
    from palaestrai.agent import ActuatorInformation
    from palaestrai_socal import spaces

    return ActuatorInformation(
        value=spaces.encode_mutations([], cap=spaces.CAP),
        space=spaces.mutation_space(spaces.CAP),
        uid="gis.cell_mutations",
    )


def _muscle(**over):
    from palaestrai_socal.agents.firefighter_drl_agent import (
        LearningFirefighterMuscle,
    )

    kw = dict(
        n_planes=3, n_helos=2, n_crews=4, n_dozers=2, n_engines=3,
        start_steps=10_000,  # force warm-up (random doctrine) in TRAIN
        env_step_min=60.0, max_steps=60, wind_speed=8.0, wind_dir_deg=45.0,
        raster_nrows=NR, raster_ncols=NC, bounds=BOUNDS, cell_size_m=CELL_M,
    )
    kw.update(over)
    return LearningFirefighterMuscle(**kw)


def _front():
    from palaestrai_socal import spaces

    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    S[10, 8:13] = spaces.BURNING
    return S


# -- doctrine map ----------------------------------------------------------
@_needs_stack
def test_doctrine_map_covers_all_actions():
    from palaestrai_socal.agents import firefighter_drl as drl
    from palaestrai_socal.agents.firefighter_drl_agent import _DOCTRINE_MAP

    assert set(_DOCTRINE_MAP) == {
        drl.ACT_NOOP, drl.ACT_INDIRECT, drl.ACT_DIRECT, drl.ACT_TRIAGE
    }
    # triage carries the engines gate + protect_assets on.
    doc, protect, need = _DOCTRINE_MAP[drl.ACT_TRIAGE]
    assert protect is True and need == "engines"


# -- return contract -------------------------------------------------------
@_needs_stack
def test_propose_returns_obs_action_pair():
    from palaestrai_socal.agents import firefighter_drl as drl

    m = _muscle()
    acts, brain = m.propose_actions(_sensors(_front()), [_actuator()])
    obs, action = brain
    assert obs.shape == (drl.OBS_DIM,)
    assert obs.dtype == np.float64          # cast for the brain buffer
    assert action.shape == (1,)
    assert 0 <= int(action[0]) < drl.N_TACTICS


@_needs_stack
def test_emits_dtype_correct_mutations_actuator():
    m = _muscle()
    a = _actuator()
    acts, _ = m.propose_actions(_sensors(_front()), [a])
    out = acts[0]
    assert np.asarray(out.value).dtype == out.space.dtype


# -- warm-up doctrine is a valid discrete index ----------------------------
@_needs_stack
def test_warmup_doctrine_in_range_and_deterministic_seed():
    from palaestrai_socal.agents import firefighter_drl as drl

    np.random.seed(0)
    m = _muscle()
    ids = set()
    for _ in range(20):
        _, (_, action) = m.propose_actions(_sensors(_front()), [_actuator()])
        ids.add(int(action[0]))
    assert ids and all(0 <= i < drl.N_TACTICS for i in ids)


# -- resource gate ---------------------------------------------------------
@_needs_stack
def test_doctrine_executable_gate():
    m = _muscle(n_planes=0, n_helos=0, n_crews=0, n_dozers=0, n_engines=0)
    # no fleet at all -> line/engines doctrines are not executable.
    assert m._doctrine_executable("none", 8.0) is True
    assert m._doctrine_executable("line", 8.0) is False
    assert m._doctrine_executable("engines", 8.0) is False
    # air-only fleet (no ground crews/dozers/engines): a 'line' is flyable in
    # low wind but NOT once the aircraft are wind-grounded.
    m2 = _muscle(n_planes=3, n_helos=0, n_crews=0, n_dozers=0, n_engines=2)
    assert m2._doctrine_executable("line", 8.0) is True      # air ok in low wind
    assert m2._doctrine_executable("line", 25.0) is False    # grounded, no ground
    assert m2._doctrine_executable("engines", 8.0) is True


@_needs_stack
def test_ground_crews_survive_high_wind_gate():
    # crews/dozers are not wind-grounded -> a 'line' doctrine stays executable
    # in high wind when ground resources exist even though tankers are grounded.
    m = _muscle(n_planes=3, n_helos=0, n_crews=4, n_dozers=2, n_engines=0)
    assert m._doctrine_executable("line", 25.0) is True


# -- served MW recovery ----------------------------------------------------
@_needs_stack
def test_served_mw_sums_load_sensors_only():
    m = _muscle()
    s = _sensors(_front(), served=(0.3, 0.3, 0.2))
    assert m._served_mw(s) == pytest.approx(0.8)
    # no load sensors -> None (muscle falls back to base served).
    from palaestrai.agent import SensorInformation
    from palaestrai_socal import spaces

    only_state = [
        SensorInformation(
            value=_front().astype(np.float64).ravel(),
            space=spaces.vector_box(0.0, 16.0, NR * NC),
            uid="gis.cell_state",
        )
    ]
    assert m._served_mw(only_state) is None
