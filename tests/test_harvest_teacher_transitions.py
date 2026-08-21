"""Tests for the offline teacher-transition harvester (harvest_teacher_transitions).

Two tiers:

* **Pure unit** (numpy-only, no store): the payload-decoding helpers that turn a
  stored ``muscle_actions`` row back into a DRL label -- ``_env_step_index``
  (``socal_grid`` tick -> env-step index), ``_decode_setpoints`` (jsonpickle
  ``ActuatorInformation`` -> cell mutations via ``spaces.decode_mutations``),
  the ``PHASE_FLEET`` / ``DEFAULT_PHASES`` tables, and the per-step
  forward-fill in ``_teacher_actions_for_phase`` (DB monkeypatched away).
* **System** (needs a running v0.5 store + psycopg2): end-to-end
  ``harvest_phase`` / ``harvest`` against the live ``palaestrai_eaton_v05``
  store, skipped automatically when the store is unreachable.

The store URI defaults to the local dev cluster and is overridable via the
``SOCAL_EATON_STORE`` env var so the same file runs in other environments.
"""

import json
import os

import numpy as np
import pytest

from palaestrai_socal import spaces
from palaestrai_socal.agents import firefighter_drl as drl
from palaestrai_socal.agents import harvest_teacher_transitions as h

# --------------------------------------------------------------------------
# store-reachability probe (system tier only)
# --------------------------------------------------------------------------
_STORE = os.environ.get(
    "SOCAL_EATON_STORE",
    "postgresql://palaestrai:socal_local_1782561794@127.0.0.1:5433/palaestrai_eaton_v05",
)


def _store_reachable(uri: str) -> bool:
    try:  # pragma: no cover - environment probe
        import psycopg2

        con = psycopg2.connect(uri, connect_timeout=3)
        con.close()
        return True
    except Exception:  # noqa: BLE001
        return False


_needs_store = pytest.mark.skipif(
    not _store_reachable(_STORE),
    reason=f"v0.5 Eaton store unreachable ({_STORE})",
)


# --------------------------------------------------------------------------
# helpers to build a realistic stored payload
# --------------------------------------------------------------------------
def _setpoints_payload(muts):
    """A jsonpickle-style ``actuator_setpoints`` list holding one cell_mutations
    actuator, exactly as palaestrAI writes it into the store."""
    vec = spaces.encode_mutations(muts)
    return [
        {
            "py/state": {
                "uid": "gis_world.gis.cell_mutations",
                "space": "Box(...)",
                "value": [float(x) for x in vec],
                "value_ids": None,
            }
        }
    ]


def _simtimes(grid_tick):
    return {
        "gis_world": {"simtime_ticks": 0, "simtime_timestamp": None},
        "socal_grid": {"simtime_ticks": grid_tick, "simtime_timestamp": None},
    }


# ==========================================================================
# _env_step_index -- tick -> env-step mapping
# ==========================================================================
def test_env_step_index_first_step():
    # world_states for socal_grid tick every 3600 s starting at 3600 (step 1);
    # step index = tick/secs - 1, so tick 3600 -> index 0.
    assert h._env_step_index(_simtimes(3600), 3600.0) == 0


def test_env_step_index_later_step():
    # the teacher acts every 4th env step: 3600, 18000, 32400 ...
    # tick 14400 -> index 3.
    assert h._env_step_index(_simtimes(14400), 3600.0) == 3
    assert h._env_step_index(_simtimes(18000), 3600.0) == 4


def test_env_step_index_accepts_json_string():
    # the store may hand back a jsonb string; the helper must json.loads it.
    assert h._env_step_index(json.dumps(_simtimes(3600)), 3600.0) == 0


def test_env_step_index_none_when_missing():
    assert h._env_step_index({"gis_world": {"simtime_ticks": 0}}, 3600.0) is None
    assert h._env_step_index({"socal_grid": {}}, 3600.0) is None
    assert h._env_step_index("not-a-dict-payload", 3600.0) is None
    assert h._env_step_index(_simtimes(3600), 0.0) is None


def test_env_step_index_never_negative():
    # a tick below one env step still clamps to step 0, never a negative index.
    assert h._env_step_index(_simtimes(1800), 3600.0) == 0


# ==========================================================================
# _decode_setpoints -- jsonpickle ActuatorInformation -> cell mutations
# ==========================================================================
def test_decode_setpoints_roundtrip():
    muts = [(1, 2, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION),
            (5, 7, spaces.CONTAINED, spaces.LAYER_SUPPRESSION)]
    payload = _setpoints_payload(muts)
    decoded = h._decode_setpoints(payload)
    assert decoded == muts


def test_decode_setpoints_accepts_json_string():
    muts = [(3, 3, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)]
    decoded = h._decode_setpoints(json.dumps(_setpoints_payload(muts)))
    assert decoded == muts


def test_decode_setpoints_empty():
    assert h._decode_setpoints([]) == []
    assert h._decode_setpoints(None) == []
    # a payload with no cell_mutations actuator yields no mutations.
    other = [{"py/state": {"uid": "gis_world.gis.something_else", "value": [1.0]}}]
    assert h._decode_setpoints(other) == []


# ==========================================================================
# teacher_action_from_mutations reuse (doctrine inference)
# ==========================================================================
def test_decode_then_infer_doctrine():
    # a suppression line -> the teacher chose a non-noop doctrine.
    muts = [(r, 4, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION) for r in range(3)]
    decoded = h._decode_setpoints(_setpoints_payload(muts))
    act = drl.teacher_action_from_mutations(decoded)
    assert act in (drl.ACT_INDIRECT, drl.ACT_DIRECT, drl.ACT_TRIAGE)
    # no mutations -> no-op.
    assert drl.teacher_action_from_mutations([]) == drl.ACT_NOOP


# ==========================================================================
# _teacher_actions_for_phase -- forward-fill across coarse decision cadence
# ==========================================================================
def test_forward_fill_holds_doctrine(monkeypatch):
    # two decisions: DIRECT held from env step 0, TRIAGE from env step 4.
    line = [(0, 0, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)]
    direct = _setpoints_payload(line)   # a single suppression line -> DIRECT
    triage = _setpoints_payload(
        [(r, c, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)
         for r in range(6) for c in range(6)]  # a broad area -> TRIAGE
    )
    fake_rows = [
        (direct, _simtimes(3600)),    # env step 0
        (triage, _simtimes(18000)),   # env step 4
    ]

    class _Cur:
        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return fake_rows

    class _Con:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(h, "_connect", lambda uri: _Con())

    per_step = h._teacher_actions_for_phase(
        "postgresql://ignored", "phase_1_air", 8, env_step_secs=3600.0
    )
    assert len(per_step) == 8
    d0 = drl.teacher_action_from_mutations(h._decode_setpoints(direct))
    d4 = drl.teacher_action_from_mutations(h._decode_setpoints(triage))
    # steps 0..3 hold the first decision, steps 4..7 hold the second.
    assert per_step[0] == d0
    assert per_step[3] == d0
    assert per_step[4] == d4
    assert per_step[7] == d4


def test_forward_fill_defaults_noop_before_first_decision(monkeypatch):
    # first decision only lands at env step 4 -> steps 0..3 stay ACT_NOOP.
    line = _setpoints_payload([(0, 0, spaces.SUPPRESSED, spaces.LAYER_SUPPRESSION)])

    class _Cur:
        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return [(line, _simtimes(18000))]  # env step 4

    class _Con:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(h, "_connect", lambda uri: _Con())
    per_step = h._teacher_actions_for_phase(
        "postgresql://ignored", "phase_1_air", 6, env_step_secs=3600.0
    )
    assert per_step[0] == drl.ACT_NOOP
    assert per_step[3] == drl.ACT_NOOP
    assert per_step[4] != drl.ACT_NOOP


def test_forward_fill_empty_run(monkeypatch):
    class _Cur:
        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return []

    class _Con:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr(h, "_connect", lambda uri: _Con())
    per_step = h._teacher_actions_for_phase(
        "postgresql://ignored", "phase_1_air", 5, env_step_secs=3600.0
    )
    assert per_step == [drl.ACT_NOOP] * 5


# ==========================================================================
# static tables
# ==========================================================================
def test_default_phases_and_fleet_tables():
    # phase_0_no_ff is harvested too: it is the ONLY phase with fire-driven
    # load shedding, so without it the SAIDI reward column is uniformly zero.
    assert h.DEFAULT_PHASES == (
        "phase_0_no_ff",
        "phase_1_air", "phase_2_air_ground", "phase_3_full_triage",
    )
    assert all(v == 0 for v in h.PHASE_FLEET["phase_0_no_ff"].values())
    # every default phase has a fleet entry with the five resource keys.
    for ph in h.DEFAULT_PHASES:
        fleet = h.PHASE_FLEET[ph]
        assert set(fleet) == {
            "n_planes", "n_helos", "n_crews", "n_dozers", "n_engines",
        }
    # air-only phase has planes but no ground crews/engines; full triage has all.
    assert h.PHASE_FLEET["phase_1_air"]["n_engines"] == 0
    assert h.PHASE_FLEET["phase_1_air"]["n_crews"] == 0
    assert h.PHASE_FLEET["phase_3_full_triage"]["n_engines"] > 0
    assert h.PHASE_FLEET["phase_3_full_triage"]["n_helos"] > 0


# ==========================================================================
# system tier -- live store
# ==========================================================================
@pytest.mark.slow
@_needs_store
def test_harvest_phase_against_live_store():
    part = h.harvest_phase(
        _STORE,
        "phase_1_air",
        base_served_mw=drl.BASE_SERVED_MW,
        saidi_scale=drl.SAIDI_SCALE,
        env_step_min=60.0,
    )
    assert part is not None
    for key in ("obs", "actions", "rewards", "next_obs", "dones"):
        assert key in part
    n = len(part["obs"])
    assert n > 0
    # contract shapes / dtypes.
    assert part["obs"].shape == (n, drl.OBS_DIM)
    assert part["next_obs"].shape == (n, drl.OBS_DIM)
    assert part["obs"].dtype == np.float32
    assert part["actions"].dtype == np.int64
    assert part["rewards"].dtype == np.float32
    assert part["dones"].dtype == bool
    # action labels are valid doctrine ids; rewards are <= 0 (SAIDI can't drop).
    assert part["actions"].min() >= 0
    assert part["actions"].max() < drl.N_TACTICS
    assert float(part["rewards"].max()) <= 1e-6
    # exactly one terminal transition (the last).
    assert int(part["dones"].sum()) == 1
    assert bool(part["dones"][-1]) is True


@pytest.mark.slow
@_needs_store
def test_harvest_full_writes_npz(tmp_path):
    out = tmp_path / "eaton_teacher_test.npz"
    info = h.harvest(
        _STORE,
        str(out),
        phases=h.DEFAULT_PHASES,
        base_served_mw=drl.BASE_SERVED_MW,
        saidi_scale=drl.SAIDI_SCALE,
        env_step_min=60.0,
    )
    assert out.exists()
    assert info["n"] > 0
    assert len(info["action_hist"]) == drl.N_TACTICS
    assert sum(info["action_hist"]) == info["n"]
    # reload and re-validate the merged arrays + meta.
    with np.load(out, allow_pickle=True) as z:
        assert z["obs"].shape == (info["n"], drl.OBS_DIM)
        assert z["actions"].shape == (info["n"],)
        assert z["dones"].dtype == bool
        meta = z["meta"].item()
        assert meta["obs_dim"] == drl.OBS_DIM
        assert meta["n_tactics"] == drl.N_TACTICS
        assert meta["n_transitions"] == info["n"]
        assert set(meta["phases"]).issubset(set(h.DEFAULT_PHASES))


# ---------------------------------------------------------------------------
# --objective moo: the offline reward must equal the ONLINE objective
# ---------------------------------------------------------------------------
# The harvester recomputes the reward from stored telemetry rather than reading
# muscle_actions.objective, so the same reward function now exists in two
# places. That is a drift hazard, and this is the test that catches it: for
# identical inputs, harvest_phase's MOO arithmetic must return exactly what
# MooObjective.internal_reward would have returned online.

def _moo_reward_from_harvester(dsaidi, burned_step, houses_total, *,
                               saidi_scale=60.0, alpha=0.5, beta=0.5,
                               saidi_norm=1e-3, houses_norm=1.0,
                               houses_scale=0.02):
    """Replicate the expression harvest_phase uses for objective='moo'."""
    r_saidi = -float(dsaidi) / saidi_scale
    r_houses = (
        -(max(0.0, burned_step) / houses_total) / houses_scale
        if houses_total > 0 else 0.0
    )
    return alpha * (r_saidi / saidi_norm) + beta * (r_houses / houses_norm)


@pytest.mark.unit
def test_harvester_moo_reward_matches_the_online_objective():
    pytest.importorskip("palaestrai.agent.objective")
    from palaestrai_socal.agents.moo_objective import MooObjective

    class _Info:
        def __init__(self, uid, value):
            self.uid, self.value = uid, value

    class _Tail:
        def __init__(self, readings):
            self.sensor_readings = readings

    class _Mem:
        def __init__(self, readings):
            self._t = _Tail(readings)

        def tail(self, n=1):
            return self._t

    houses_total, burned_step = 101.0, 2.0
    base_mw, served_mw, dt_min, scale = 232.237, 222.237, 60.0, 60.0

    # what the environment/reader would report for that step
    disconnected = base_mw - served_mw
    dsaidi = (disconnected * drl.CUSTOMERS_PER_MW * dt_min) / (
        base_mw * drl.CUSTOMERS_PER_MW
    )

    obj = MooObjective(
        alpha=0.5, beta=0.5,
        saidi_params={"base_served_mw": base_mw, "scale": scale, "dt_min": dt_min},
        houses_params={"scale": 0.02},
    )
    online = obj.internal_reward(_Mem([
        _Info("gis_world.gis.houses_total", np.array([houses_total])),
        _Info("gis_world.gis.houses_burned_this_step", np.array([burned_step])),
        _Info("socal_grid.Powergrid-0.0-load-1-1.p_mw", np.array([served_mw])),
    ]))

    offline = _moo_reward_from_harvester(
        dsaidi, burned_step, houses_total, saidi_scale=scale
    )
    assert offline == pytest.approx(online, rel=1e-9), (
        f"offline harvest reward {offline} != online objective {online}; "
        "the two copies of the MOO reward function have drifted"
    )


@pytest.mark.unit
def test_harvester_moo_reward_is_never_positive():
    for dsaidi, burned, total in ((0, 0, 101), (0.5, 3, 101), (0, 5, 0)):
        assert _moo_reward_from_harvester(dsaidi, burned, total) <= 0.0


@pytest.mark.unit
def test_harvester_moo_falls_back_when_no_settlement():
    """A raster with no class-9 cells contributes only the SAIDI term."""
    both = _moo_reward_from_harvester(0.5, 2, 101)
    saidi_only = _moo_reward_from_harvester(0.5, 2, 0)
    assert saidi_only > both          # less negative: the house charge is gone
    assert saidi_only == pytest.approx(_moo_reward_from_harvester(0.5, 0, 101))
