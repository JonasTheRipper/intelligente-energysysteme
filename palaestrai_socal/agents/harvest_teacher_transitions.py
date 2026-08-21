"""Harvest offline (s, a, r, s', done) transitions from a scripted-teacher run.

The v0.7 DRL firefighter is CQL-bootstrapped from the *scripted* firefighter's
behaviour. This script reads a completed v0.5 firefighting store (Eaton or
Palisades) back out of PostgreSQL and materialises, for every firefighting
phase, the per-step transitions the teacher generated -- encoded in the DRL
firefighter's exact 17-dim observation / Discrete(4) action / SAIDI-reward
contract (:mod:`palaestrai_socal.agents.firefighter_drl`). The result is a
``.npz`` the :class:`FirefighterSacBrain` loads into its replay buffer before
online training.

Why go through the store (not re-run)?
--------------------------------------
The store already holds every environment frame (``world_states``) and every
firefighter decision (``muscle_actions``). :func:`analysis.store_readers.read_run`
reconstructs the per-step fire grid, served MW, cumulative SAIDI, and wind for a
single phase (identical maths to the environment). We reuse it verbatim so the
harvested observations/rewards match what the live agent will see, then read the
firefighter's ``actuator_setpoints`` (the encoded cell mutations) to label each
step with the doctrine id the teacher effectively chose.

Output ``.npz`` schema
----------------------
``obs`` (N, 17) float32, ``actions`` (N,) int64, ``rewards`` (N,) float32,
``next_obs`` (N, 17) float32, ``dones`` (N,) bool, plus a ``meta`` 0-d object
array with the source store, phases, and contract constants.

CLI
---
    python -m palaestrai_socal.agents.harvest_teacher_transitions \
        --store postgresql://.../palaestrai_eaton_v05 \
        --out data/offline/eaton_teacher_all.npz \
        [--phases phase_1_air,phase_2_air_ground,phase_3_full_triage] \
        [--base-served-mw 1.0] [--saidi-scale 60] [--env-step-min 60]
        [--objective saidi|moo] [--alpha 0.5] [--beta 0.5]
        [--saidi-norm 1e-3] [--houses-norm 1.0] [--houses-scale 0.02]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# allow "python analysis/..." and "-m" both to import analysis + package.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "analysis") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "analysis"))

from palaestrai_socal.agents import firefighter_drl as drl  # noqa: E402
from palaestrai_socal import spaces  # noqa: E402

# Phases harvested by default.
#
# ``phase_0_no_ff`` IS included, despite having no firefighting fleet, because
# it is the only phase that carries grid signal. Measured on the Eaton
# scenario: all three scripted doctrines prevent 100% of fire-driven load
# shedding (served MW min == max == 232.24 across phases 1-3), so their SAIDI
# charge is identically zero. Only the no-response baseline sheds load
# (232.24 -> 208.42 MW, SAIDI 122.4). Harvesting phases 1-3 alone therefore
# yields an all-zero SAIDI buffer that cannot rank anything.
#
# Its transitions are labelled ACT_NOOP (zero fleet -> zero mutations), which
# is exactly the contrast the critic needs: "not acting leads to outage".
#
# Historical note: data/offline/eaton_teacher_all.npz appears to carry SAIDI
# signal (mean -7.8e-5) only because it was harvested before served MW was
# restricted to the agent's own load subscription. Summing every load in the
# grid picks up the grid_probe agent's DummyMuscle, which writes random
# setpoints to Powergrid-0.0-load-0-0.p_mw each step. Reproducing that method
# on a fresh store gives mean -8.1e-5 for the same phases, versus exactly 0
# when restricted -- i.e. that buffer's reward is probe noise, not fire damage.
DEFAULT_PHASES = (
    "phase_0_no_ff",
    "phase_1_air",
    "phase_2_air_ground",
    "phase_3_full_triage",
)

# fleet mix per phase (mirrors the experiment YAML) -> resource-availability
# flags for obs features 13-15 and the wind-grounding gate.
PHASE_FLEET: Dict[str, Dict[str, int]] = {
    # No fleet at all: every resource-availability obs feature reads 0 and the
    # teacher action is necessarily ACT_NOOP. Listed explicitly rather than
    # relying on the all-zeros fallback, so the table covers DEFAULT_PHASES.
    "phase_0_no_ff": dict(
        n_planes=0, n_helos=0, n_crews=0, n_dozers=0, n_engines=0
    ),
    "phase_1_air": dict(n_planes=3, n_helos=0, n_crews=0, n_dozers=0, n_engines=0),
    "phase_2_air_ground": dict(
        n_planes=3, n_helos=0, n_crews=4, n_dozers=2, n_engines=0
    ),
    "phase_3_full_triage": dict(
        n_planes=3, n_helos=2, n_crews=4, n_dozers=2, n_engines=3
    ),
}


def _connect(store_uri: str):
    import psycopg2  # local import: PG store only

    return psycopg2.connect(store_uri)


def _teacher_actions_for_phase(
    store_uri: str,
    phase_uid: str,
    n_env_steps: int,
    *,
    agent_name: str = "firefighter",
    env_step_secs: float = 3600.0,
) -> List[int]:
    """Return the doctrine id (0..3) the teacher held at EACH env step.

    The firefighter agent decides at a coarser cadence than the environment
    steps (in the v0.5 stores it acts every 4th env step: its ``socal_grid``
    simtime advances 14400 s while world_states tick every 3600 s). palaestrAI
    holds a muscle's actuator setpoint between decisions, so a decision taken at
    env step ``k`` stays in force until the next decision. We therefore:

    1. read the firefighter's ``muscle_actions`` for THIS phase only (joined
       through ``agents.experiment_run_phase_id`` -- a real phase filter, not a
       cross join);
    2. decode each row's ``actuator_setpoints`` -> doctrine id via
       :func:`firefighter_drl.teacher_action_from_mutations`;
    3. map each decision to its env-step index from the stored ``socal_grid``
       simtime (``tick / env_step_secs - 1``) and forward-fill across the
       ``n_env_steps`` snaps, so the returned list is per-env-step aligned.

    Steps before the first decision default to ``ACT_NOOP``.
    """
    con = _connect(store_uri)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT ma.actuator_setpoints, ma.simtimes "
            "FROM muscle_actions ma "
            "JOIN agents a ON a.id = ma.agent_id "
            "JOIN experiment_run_phases p ON p.id = a.experiment_run_phase_id "
            "WHERE a.name = %s AND p.uid = %s "
            "ORDER BY ma.id",
            (agent_name, phase_uid),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    # (env_step_index, doctrine_id) for each decision.
    decisions: List[Tuple[int, int]] = []
    for setpoints, simtimes in rows:
        muts = _decode_setpoints(setpoints)
        act = drl.teacher_action_from_mutations(muts)
        idx = _env_step_index(simtimes, env_step_secs)
        decisions.append((idx if idx is not None else len(decisions), act))

    # forward-fill the held doctrine across every env step.
    per_step: List[int] = [drl.ACT_NOOP] * max(0, n_env_steps)
    if not decisions:
        return per_step
    decisions.sort(key=lambda d: d[0])
    di = 0
    current = drl.ACT_NOOP
    for step in range(n_env_steps):
        while di < len(decisions) and decisions[di][0] <= step:
            current = decisions[di][1]
            di += 1
        per_step[step] = current
    return per_step


def _env_step_index(
    simtimes, env_step_secs: float
) -> Optional[int]:
    """Env-step index (0-based) a firefighter decision applies to.

    Reads the ``socal_grid`` simtime tick from the stored ``simtimes`` payload;
    world_states for ``socal_grid`` tick every ``env_step_secs`` starting at
    ``env_step_secs`` (step 1 -> tick 3600). So step index = tick/secs - 1.
    Returns ``None`` when the tick is unavailable (caller falls back to order).
    """
    if isinstance(simtimes, str):
        try:
            payload = json.loads(simtimes)
        except (ValueError, TypeError):
            return None
    else:
        payload = simtimes
    if not isinstance(payload, dict):
        return None
    grid = payload.get("socal_grid") or {}
    tick = grid.get("simtime_ticks")
    if tick is None or not env_step_secs:
        return None
    return max(0, int(round(float(tick) / env_step_secs)) - 1)


def _decode_setpoints(setpoints) -> List[Tuple[int, int, int, int]]:
    """Decode a stored ``actuator_setpoints`` payload into cell mutations.

    palaestrAI stores actuators as jsonpickle ``ActuatorInformation`` dicts; the
    cell_mutations actuator's ``value`` is the fixed-size packed vector produced
    by :func:`spaces.encode_mutations`. We locate it by uid suffix and decode.
    """
    payload = json.loads(setpoints) if isinstance(setpoints, str) else setpoints
    if not payload:
        return []
    for a in payload:
        st = a["py/state"] if isinstance(a, dict) and "py/state" in a else a
        uid = st.get("uid", "") if isinstance(st, dict) else ""
        if uid.endswith("gis.cell_mutations"):
            vec = np.asarray(st.get("value"), dtype=float)
            return list(spaces.decode_mutations(vec))
    return []


def agent_load_uids(
    store_uri: str, phase_uid: str, agent_name: str = "firefighter"
) -> Optional[set]:
    """The ``*-load-*.p_mw`` sensor uids an agent actually subscribed to.

    palaestrAI stores each agent's resolved configuration as JSONB in
    ``agents.configuration``, so the exact subscription is recoverable from the
    store -- no need to re-parse the experiment YAML or hard-code a list that
    would silently drift from it.

    This matters because SaidiObjective derives served MW by summing the
    sensors THAT AGENT sees. The stored world_state carries every load in the
    grid, so an unrestricted sum baselines at ~27,100 MW against the agent's
    ~232 MW -- making offline SAIDI ~117x smaller than the reward the same
    agent computed online. Returns None when the configuration is unavailable,
    in which case the caller falls back to the unrestricted sum.
    """
    con = _connect(store_uri)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT a.configuration FROM agents a "
            "JOIN experiment_run_phases p ON p.id = a.experiment_run_phase_id "
            "WHERE a.name = %s AND p.uid = %s LIMIT 1",
            (agent_name, phase_uid),
        )
        row = cur.fetchone()
    except Exception:
        return None
    finally:
        con.close()
    if not row or not row[0]:
        return None
    cfg = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    sensors = cfg.get("sensors") or []
    uids = {
        str(u) for u in sensors
        if "-load-" in str(u) and str(u).endswith(".p_mw")
    }
    return uids or None


def harvest_phase(
    store_uri: str,
    phase_uid: str,
    *,
    base_served_mw: float,
    saidi_scale: float,
    env_step_min: float,
    agent_name: str = "firefighter",
    objective: str = "saidi",
    alpha: float = 0.5,
    beta: float = 0.5,
    saidi_norm: float = 1.0e-3,
    houses_norm: float = 1.0,
    houses_scale: float = 0.02,
) -> Optional[dict]:
    """Build the transition arrays for a single firefighting phase.

    ``objective`` selects the reward written into the buffer:

    ``"saidi"`` (default)
        ``-delta_saidi / saidi_scale`` -- what SaidiObjective returns online.
    ``"moo"``
        ``alpha * (saidi / saidi_norm) + beta * (houses / houses_norm)``, the
        same arithmetic MooObjective performs, where the house term is
        ``-(houses burned this step / houses_total) / houses_scale``.

    The reward is *recomputed* from the stored per-step telemetry rather than
    read from ``muscle_actions.objective``, deliberately. palaestrAI remembers a
    step as ``(sensors[t-1], setpoints[t-1], rewards[t])`` and computes the
    objective from the *previous* readings, so the stored scalar does not line
    up 1:1 with the ``(s, a, s')`` transitions built here. Recomputing keeps the
    existing, tested alignment (``dsaidi = saidi[i+1] - saidi[i]`` prices
    transition ``i``) and applies the house term with the identical indexing.
    Keep the constants here in sync with the MooObjective params in the
    experiment YAML -- they are the same reward function, written twice.
    """
    from store_readers import read_run  # analysis/ on sys.path

    # Restrict served-MW to the agent's own load subscription so the offline
    # SAIDI is computed on the same basis as the online SaidiObjective.
    load_uids = agent_load_uids(store_uri, phase_uid, agent_name)
    snaps, meta = read_run(
        store_uri,
        env_step_min=env_step_min,
        phase_uid=phase_uid,
        load_uids=load_uids,
    )
    if not snaps:
        return None

    fuel = meta.get("fuel")
    dem = meta.get("dem")
    cell_size_m = meta.get("delta_m")
    base = meta.get("base_served", base_served_mw) or base_served_mw

    fleet = PHASE_FLEET.get(
        phase_uid,
        dict(n_planes=0, n_helos=0, n_crews=0, n_dozers=0, n_engines=0),
    )

    n = len(snaps)
    max_steps = n
    # per-env-step doctrine labels (forward-filled across the teacher's coarser
    # decision cadence), aligned 1:1 with ``snaps``.
    teacher_actions = _teacher_actions_for_phase(
        store_uri,
        phase_uid,
        n,
        agent_name=agent_name,
        env_step_secs=float(env_step_min) * 60.0,
    )

    def _obs_for(i: int) -> np.ndarray:
        snap = snaps[i]
        # reconstruct the cell_state grid from fire_code (BURNING/BURNED_OUT/
        # SUPPRESSED/CONTAINED) so obs fractions match the live agent.
        fc = snap["fire_code"]
        state = np.zeros_like(fc, dtype=np.int8)
        state[fc == 1] = spaces.BURNING
        state[fc == 2] = spaces.BURNED_OUT
        state[fc == 3] = spaces.SUPPRESSED
        state[fc == 5] = spaces.CONTAINED
        prev_saidi = snaps[i - 1]["saidi"] if i > 0 else 0.0
        avail = drl.resource_availability(
            wind_speed=snap["wind_speed"], **fleet
        )
        return drl.extract_obs(
            state=state,
            fuel=fuel,
            dem=dem,
            cell_size_m=cell_size_m,
            wind_speed=snap["wind_speed"],
            wind_dir_deg=meta.get("wind_dir_deg", 45.0),
            served_mw=snap["served_mw"],
            base_served_mw=base,
            saidi=snap["saidi"],
            prev_saidi=prev_saidi,
            step=i + 1,
            max_steps=max_steps,
            saidi_scale=saidi_scale,
            **avail,
        )

    obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
    for i in range(n - 1):
        o = _obs_for(i)
        o2 = _obs_for(i + 1)
        # action label: teacher doctrine for this step (default no-op if the
        # muscle_actions row count differs from snaps).
        a = teacher_actions[i] if i < len(teacher_actions) else drl.ACT_NOOP
        # SAIDI charge for this transition (<= 0), identical to SaidiObjective.
        dsaidi = max(0.0, snaps[i + 1]["saidi"] - snaps[i]["saidi"])
        r_saidi = -float(dsaidi) / (saidi_scale if saidi_scale else 1.0)
        if objective == "moo":
            # Structural charge, identical to BurnedHousesObjective: the houses
            # destroyed during the step that ENDS the transition, as a fraction
            # of the settlement, normalised by houses_scale.
            nxt = snaps[i + 1]
            total_houses = float(nxt.get("houses_total", 0.0) or 0.0)
            burned_step = max(0.0, float(nxt.get("houses_burned_this_step", 0.0) or 0.0))
            r_houses = (
                -(burned_step / total_houses) / (houses_scale or 1.0)
                if total_houses > 0
                else 0.0
            )
            r = (
                alpha * (r_saidi / (saidi_norm or 1.0))
                + beta * (r_houses / (houses_norm or 1.0))
            )
        else:
            r = r_saidi
        obs_list.append(o)
        act_list.append(int(a))
        rew_list.append(float(r))
        next_list.append(o2)
        done_list.append(i + 1 == n - 1)

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.int64),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "next_obs": np.asarray(next_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=bool),
    }


def harvest(
    store_uri: str,
    out_path: str,
    *,
    phases: Tuple[str, ...] = DEFAULT_PHASES,
    base_served_mw: float = drl.BASE_SERVED_MW,
    saidi_scale: float = drl.SAIDI_SCALE,
    env_step_min: float = 60.0,
    agent_name: str = "firefighter",
    objective: str = "saidi",
    alpha: float = 0.5,
    beta: float = 0.5,
    saidi_norm: float = 1.0e-3,
    houses_norm: float = 1.0,
    houses_scale: float = 0.02,
) -> dict:
    """Harvest all firefighting phases in a store into one ``.npz``.

    ``objective="moo"`` writes MooObjective-shaped rewards; see
    :func:`harvest_phase`. A buffer harvested with one reward function must not
    bootstrap a brain trained under another -- the critic would be fitted to a
    mixture of two objectives -- so the choice is recorded in the ``.npz`` meta.
    """
    parts: List[dict] = []
    used_phases: List[str] = []
    for ph in phases:
        try:
            part = harvest_phase(
                store_uri,
                ph,
                base_served_mw=base_served_mw,
                saidi_scale=saidi_scale,
                env_step_min=env_step_min,
                agent_name=agent_name,
                objective=objective,
                alpha=alpha,
                beta=beta,
                saidi_norm=saidi_norm,
                houses_norm=houses_norm,
                houses_scale=houses_scale,
            )
        except ValueError:
            part = None  # phase absent in this store
        if part is not None and len(part["obs"]):
            parts.append(part)
            used_phases.append(ph)

    if not parts:
        raise SystemExit(f"no teacher transitions harvested from {store_uri}")

    merged = {
        k: np.concatenate([p[k] for p in parts], axis=0)
        for k in ("obs", "actions", "rewards", "next_obs", "dones")
    }
    meta = np.array(
        {
            "store_uri": store_uri,
            "phases": used_phases,
            "objective": objective,
            "alpha": alpha,
            "beta": beta,
            "saidi_norm": saidi_norm,
            "houses_norm": houses_norm,
            "houses_scale": houses_scale,
            "obs_dim": drl.OBS_DIM,
            "n_tactics": drl.N_TACTICS,
            "saidi_scale": saidi_scale,
            "base_served_mw": base_served_mw,
            "n_transitions": int(len(merged["obs"])),
        },
        dtype=object,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, meta=meta, **merged)
    return {
        "out": out_path,
        "n": int(len(merged["obs"])),
        "phases": used_phases,
        "action_hist": np.bincount(
            merged["actions"], minlength=drl.N_TACTICS
        ).tolist(),
        "objective": objective,
        "reward_mean": float(merged["rewards"].mean()),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, help="store URI (postgresql://...)")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument(
        "--phases",
        default=",".join(DEFAULT_PHASES),
        help="comma-separated phase uids to harvest",
    )
    p.add_argument("--base-served-mw", type=float, default=drl.BASE_SERVED_MW)
    p.add_argument("--saidi-scale", type=float, default=drl.SAIDI_SCALE)
    p.add_argument("--env-step-min", type=float, default=60.0)
    p.add_argument("--agent-name", default="firefighter")
    p.add_argument(
        "--objective",
        choices=("saidi", "moo"),
        default="saidi",
        help="reward written into the buffer (default: saidi, the v0.7 shape)",
    )
    p.add_argument("--alpha", type=float, default=0.5,
                   help="moo: grid-outage weight")
    p.add_argument("--beta", type=float, default=0.5,
                   help="moo: burned-structures weight")
    p.add_argument("--saidi-norm", type=float, default=1.0e-3,
                   help="moo: per-step SAIDI charge normalising to -1")
    p.add_argument("--houses-norm", type=float, default=1.0,
                   help="moo: per-step house charge normalising to -1")
    p.add_argument("--houses-scale", type=float, default=0.02,
                   help="moo: settlement fraction lost in one step scoring -1")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    ns = _parse_args(argv)
    phases = tuple(s.strip() for s in ns.phases.split(",") if s.strip())
    info = harvest(
        ns.store,
        ns.out,
        phases=phases,
        base_served_mw=ns.base_served_mw,
        saidi_scale=ns.saidi_scale,
        env_step_min=ns.env_step_min,
        agent_name=ns.agent_name,
        objective=ns.objective,
        alpha=ns.alpha,
        beta=ns.beta,
        saidi_norm=ns.saidi_norm,
        houses_norm=ns.houses_norm,
        houses_scale=ns.houses_scale,
    )
    print(
        f"harvested {info['n']} transitions -> {info['out']}\n"
        f"  phases: {', '.join(info['phases'])}\n"
        f"  action histogram (noop/indirect/direct/triage): "
        f"{info['action_hist']}\n"
        f"  objective: {info.get('objective', 'saidi')}\n"
        f"  reward mean: {info['reward_mean']:.3e}"
    )


if __name__ == "__main__":
    main()
