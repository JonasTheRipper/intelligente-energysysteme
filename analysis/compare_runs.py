"""Compare two palaestrAI stores step by step -- the whitelist equivalence check.

Motivation
----------
The ARL sensor whitelist (``keep_sensors`` on ``SocalMidasGridEnvironment``)
removes sensors before mosaik wires them, so they are never created, shipped or
stored. Sensors are read-only reports, so *by construction* this cannot change
the simulation -- but "by construction" is an argument, not a measurement. This
script turns it into one.

What is compared
----------------
Only quantities BOTH stores contain, since the filtered run deliberately holds
fewer sensors:

* ``gis.cell_state``  -- SHA-256 of the raster each step. This is the fire
  itself, and it is what the whole scenario turns on.
* ``gis.houses_*``    -- the structural telemetry.
* the ``*-load-*.p_mw`` sensors, restricted to a common uid set.

What is deliberately NOT compared
---------------------------------
Nothing here depends on the power flow's random perturbation, because the
verification experiments drop the ``grid_probe`` agent. Its ``DummyMuscle``
calls ``actuator.space.sample()`` on a gymnasium ``Box`` that is seeded from OS
entropy, so it writes a different random setpoint into ``load-0-0.p_mw`` on
every run -- making even two IDENTICAL configurations non-comparable. Verified:
two freshly constructed Boxes with the same parameters return different first
samples.

Usage
-----
    python analysis/compare_runs.py --a <store-uri> --b <store-uri> \
        [--phase phase_0_no_ff] [--label-a OFF --label-b ON]

Exit code 0 when every compared quantity matches, 1 on the first divergence.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store_readers as sr  # noqa: E402


def _grid_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.int16).tobytes()
    ).hexdigest()[:16]


def _phase_frames(store: str, phase: str) -> List[Dict[str, object]]:
    """Per-step comparable quantities for one phase of one store."""
    con, ph = sr._connect(store)
    try:
        gis = sr._fetch_env_rows(con, "gis_world", ph, phase, None, None)
        grid = sr._fetch_env_rows(con, "socal_grid", ph, phase, None, None)
    finally:
        con.close()

    frames = []
    for i, (_id, dump) in enumerate(gis):
        sm = sr._sensors_by_suffix(dump)
        state = sm.get("gis.cell_state")
        if state is None:
            continue

        def _scalar(key: str) -> Optional[float]:
            v = sm.get(key)
            return (
                float(np.asarray(v).ravel()[0])
                if v is not None and np.asarray(v).size
                else None
            )

        loads: Dict[str, float] = {}
        if i < len(grid):
            for uid, v in sr._sensors_by_suffix(grid[i][1]).items():
                if uid.endswith(".p_mw") and "-load-" in uid:
                    loads[uid] = float(np.asarray(v).ravel()[0])

        frames.append(
            {
                "step": i,
                "cell_state_sha": _grid_hash(state),
                "houses_total": _scalar("gis.houses_total"),
                "houses_burned_total": _scalar("gis.houses_burned_total"),
                "loads": loads,
            }
        )
    return frames


def compare_phase(
    store_a: str, store_b: str, phase: str, la: str, lb: str
) -> Tuple[bool, List[str]]:
    fa, fb = _phase_frames(store_a, phase), _phase_frames(store_b, phase)
    msgs: List[str] = []
    if len(fa) != len(fb):
        return False, [f"step count differs: {la}={len(fa)} {lb}={len(fb)}"]
    if not fa:
        return False, [f"no frames found for phase {phase}"]

    common = set(fa[0]["loads"]) & set(fb[0]["loads"])
    msgs.append(
        f"  {len(fa)} steps | {len(common)} load sensors in common "
        f"({len(fa[0]['loads'])} in {la}, {len(fb[0]['loads'])} in {lb})"
    )

    ok = True
    for a, b in zip(fa, fb):
        if a["cell_state_sha"] != b["cell_state_sha"]:
            msgs.append(
                f"  step {a['step']}: FIRE STATE DIFFERS "
                f"{a['cell_state_sha']} != {b['cell_state_sha']}"
            )
            ok = False
            break
        for key in ("houses_total", "houses_burned_total"):
            if a[key] != b[key]:
                msgs.append(
                    f"  step {a['step']}: {key} differs {a[key]} != {b[key]}"
                )
                ok = False
        worst = max(
            (abs(a["loads"][u] - b["loads"][u]) for u in common), default=0.0
        )
        if worst > 0.0:
            msgs.append(
                f"  step {a['step']}: load p_mw differs, max |delta| = {worst:.6g} MW"
            )
            ok = False
        if not ok:
            break
    if ok:
        msgs.append(
            f"  IDENTICAL: {len(fa)} steps, fire-state hashes, house telemetry "
            f"and {len(common)} load sensors all match exactly"
        )
    return ok, msgs


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="store URI of run A")
    p.add_argument("--b", required=True, help="store URI of run B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--phase", action="append", help="phase uid (repeatable)")
    ns = p.parse_args(argv)

    phases = ns.phase or [d["uid"] for d in sr.list_phases(ns.a)]
    all_ok = True
    print(f"comparing {ns.label_a} vs {ns.label_b}")
    for phase in phases:
        print(f"\n[{phase}]")
        ok, msgs = compare_phase(ns.a, ns.b, phase, ns.label_a, ns.label_b)
        for m in msgs:
            print(m)
        all_ok &= ok
    print("\n" + ("ALL PHASES IDENTICAL" if all_ok else "DIVERGENCE FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
