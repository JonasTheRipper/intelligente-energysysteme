"""A/B report: outcomes of training with MooObjective vs SaidiObjective.

The two arms optimise different reward functions, so their objective values are
NOT comparable to each other -- one is a normalised two-axis sum, the other a
raw SAIDI charge three orders of magnitude smaller. Comparing them directly
would be meaningless. This report therefore scores both arms on **shared ground
truth**: what actually happened in the world.

Metrics (all TERMINAL, i.e. end-of-episode values)
--------------------------------------------------
``houses_lost``   class-9 cells destroyed, and as a share of the settlement
``burned_cells``  final fire footprint
``served_mw``     sum of the agent's 14 grid-load sensors at the last decision

Terminal values are used deliberately. With a TakingTurnsSimulationController
and four agents, each agent acts on every 4th environment step (15 decisions
per 60-step episode), so any CUMULATIVE quantity read from an agent's rows is a
1-in-4 subsample and would be biased by the cadence. Terminal state is not.

Episodes are read from ``muscle_actions``, which carries ``episode`` AND
``mode`` -- necessary because ``world_states.episode`` restarts at 0 for the
evaluation episodes and therefore collides with the training ones.

Train vs evaluate
-----------------
Both are reported, but the **evaluate** episodes are the comparison that counts:
there the policy acts deterministically with no exploration noise, so the
numbers reflect what was learned rather than what was sampled.

Usage
-----
    python analysis/ab_report.py --store <uri> [--label run1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MOO_PHASE = "phase_4_drl_train"
SAIDI_PHASE = "phase_5_drl_train_saidi_control"
BURNING, BURNED_OUT = 1, 2


def _connect(uri: str):
    import psycopg2

    return psycopg2.connect(uri)


def _readings(payload) -> Dict[str, np.ndarray]:
    """Decode a stored jsonpickle sensor_readings payload to {uid: array}."""
    d = json.loads(payload) if isinstance(payload, str) else payload
    out: Dict[str, np.ndarray] = {}
    for entry in d if isinstance(d, list) else []:
        st = entry.get("py/state", entry) if isinstance(entry, dict) else {}
        uid = st.get("uid")
        if uid is None:
            continue
        val = st.get("value")
        if isinstance(val, dict) and "values" in val:
            val = val["values"]
        try:
            out[uid] = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
    return out


def _episode_rows(store: str, phase: str, agent: str = "firefighter"):
    """Last row per (mode, episode), plus that episode's mean objective."""
    con = _connect(store)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT ma.mode, ma.episode, ma.id, ma.objective, ma.sensor_readings "
            "FROM muscle_actions ma JOIN agents a ON a.id = ma.agent_id "
            "JOIN experiment_run_phases p ON p.id = a.experiment_run_phase_id "
            "WHERE a.name = %s AND p.uid = %s ORDER BY ma.id",
            (agent, phase),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    groups: Dict[Tuple[str, int], List] = {}
    for mode, ep, _id, obj, payload in rows:
        groups.setdefault((str(mode), int(ep)), []).append((obj, payload))

    out = []
    for (mode, ep), items in sorted(groups.items()):
        objs = [o for o, _ in items if o is not None]
        sm = _readings(items[-1][1])

        def _s(uid_suffix: str) -> Optional[float]:
            for u, v in sm.items():
                if u.endswith(uid_suffix) and v.size:
                    return float(v[0])
            return None

        state = None
        for u, v in sm.items():
            if u.endswith("gis.cell_state"):
                state = v
                break
        burned = (
            int(((state == BURNING) | (state == BURNED_OUT)).sum())
            if state is not None
            else None
        )
        served = sum(
            float(v[0]) for u, v in sm.items()
            if u.endswith(".p_mw") and "-load-" in u and v.size
        )
        out.append(
            {
                "mode": mode,
                "episode": ep,
                "objective_mean": float(np.mean(objs)) if objs else float("nan"),
                "houses_total": _s("gis.houses_total"),
                "houses_lost": _s("gis.houses_burned_total"),
                "burned_cells": burned,
                "served_mw": served,
            }
        )
    return out


def _summary(rows: List[dict], mode: str) -> Dict[str, float]:
    sel = [r for r in rows if r["mode"] == mode]
    if not sel:
        return {}
    def col(k):
        return np.array([r[k] for r in sel if r[k] is not None], dtype=float)
    return {
        "n": len(sel),
        "houses_lost_mean": float(col("houses_lost").mean()),
        "houses_lost_std": float(col("houses_lost").std()),
        "burned_cells_mean": float(col("burned_cells").mean()),
        "served_mw_mean": float(col("served_mw").mean()),
        "objective_mean": float(col("objective_mean").mean()),
    }


def _warn_on_empty_episodes(data: Dict[str, List[dict]]) -> int:
    """Refuse to present a run in which episodes contained no fire.

    An episode that never ignited is indistinguishable, in every metric this
    report prints, from an episode the firefighter suppressed perfectly: zero
    houses lost, zero burned cells, full served MW. That is how a muscle-reset
    leak (see ``tests/test_episode_reset.py``) produced a clean-looking A/B
    table in which 19 of 20 training episodes were empty. Non-zero exit so a
    scripted pipeline cannot quietly build on an invalid run.
    """
    bad = {
        name: [r for r in rows if not r["burned_cells"]]
        for name, rows in data.items()
    }
    if not any(bad.values()):
        return 0
    print("\n!! SUSPECT RUN: episodes with NO fire at all")
    for name, rows in bad.items():
        total = len(data[name])
        if not rows:
            continue
        eps = ", ".join(f"{r['mode'][:2]}{r['episode']}" for r in rows[:12])
        more = "" if len(rows) <= 12 else f", +{len(rows) - 12} more"
        print(f"   {name:6s} {len(rows):3d}/{total:<3d} empty: {eps}{more}")
    print("   An empty episode scores identically to a perfectly-defended one, "
          "so these\n   inflate both arms. Check that every muscle implements "
          "reset() before using\n   these numbers.")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--per-episode", action="store_true",
                   help="also print every episode, not just the summary")
    ns = p.parse_args(argv)

    arms = [("MOO", MOO_PHASE), ("SAIDI", SAIDI_PHASE)]
    data = {name: _episode_rows(ns.store, phase) for name, phase in arms}

    print(f"=== {ns.label} ===")
    for mode in ("train", "evaluate"):
        print(f"\n-- {mode} episodes --")
        print(f"{'arm':6s} {'n':>3s} {'houses lost':>18s} {'burned cells':>13s} "
              f"{'served MW':>10s} {'objective':>12s}")
        for name, _ in arms:
            s = _summary(data[name], mode)
            if not s:
                print(f"{name:6s}  (none)")
                continue
            tot = next((r["houses_total"] for r in data[name] if r["houses_total"]), 0) or 0
            pct = 100.0 * s["houses_lost_mean"] / tot if tot else float("nan")
            print(f"{name:6s} {s['n']:3d} {s['houses_lost_mean']:8.2f} +-{s['houses_lost_std']:5.2f}"
                  f" ({pct:4.1f}%) {s['burned_cells_mean']:13.1f} "
                  f"{s['served_mw_mean']:10.2f} {s['objective_mean']:12.4g}")

    if ns.per_episode:
        for name, _ in arms:
            print(f"\n-- {name} per episode --")
            for r in data[name]:
                print(f"  {r['mode']:9s} ep {r['episode']:2d}  houses_lost="
                      f"{r['houses_lost']}  burned={r['burned_cells']}  "
                      f"served={r['served_mw']:.2f}  obj={r['objective_mean']:.4g}")

    rc = _warn_on_empty_episodes(data)

    print("\nNOTE: 'objective' is NOT comparable across arms (different reward "
          "functions). Compare houses lost / burned cells / served MW.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
