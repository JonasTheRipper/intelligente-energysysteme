"""SAIDI-reward + acres-saved report for the v0.7 DRL FirefighterAgent.

Store-only analysis of a completed v0.7 DRL firefighting run. Unlike the v0.3
:mod:`analysis.firefighter_report` (which sweeps ``n_planes`` across separate
stores), this report reads a *single* multi-phase DRL store and quantifies what
the *learned* firefighter (the ``phase_5_drl_test`` evaluation phase) achieves
against the scripted baseline phase, plus the SAC/CQL SAIDI-reward learning
curve over the ``phase_4_drl_train`` training episodes.

It reuses :func:`analysis.store_readers.read_run` (per-step fire grid + SAIDI)
and :func:`analysis.store_readers.read_agent_objectives` (the firefighter's
per-decision objective trace) verbatim, so the numbers match the environment's
own maths and never require re-running a simulation.

Outputs
-------
* ``drl_firefighter_report.txt`` -- the headline table (burned acres, final
  SAIDI, acres/SAIDI saved of the DRL test phase vs the baseline phase).
* ``drl_learning_curve.png`` -- per-episode mean SAIDI-reward over training.
* ``drl_saidi_compare.png`` -- cumulative SAIDI trajectory, baseline vs DRL.
* (``--paper``) ``drl_firefighter_report.tex`` -- a LaTeX ``tabular`` of the
  same headline table for the write-up.

CLI
---
    python analysis/drl_firefighter_report.py \
        --store postgresql://.../palaestrai_eaton_v05 \
        --train-phase phase_4_drl_train \
        --test-phase  phase_5_drl_test \
        --run baseline=phase_1_air \
        --out-dir _outputs/drl_report_eaton [--paper] [--experiment eaton]

``--run NAME=PHASE`` names a comparison/baseline phase in THIS store (repeatable;
the first is treated as the scripted baseline). ``--conf`` is accepted for
symmetry with the other analysis entry points but is unused here (store-only).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never a display.
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import (  # noqa: E402
    read_agent_objectives,
    read_run,
)

SQM_PER_ACRE = 4046.8564224


# ---- per-phase KPIs -----------------------------------------------------
def phase_kpis(
    store_uri: str,
    phase_uid: str,
    *,
    gis_uid: str = "gis_world",
    grid_uid: str = "socal_grid",
    env_step_min: float = 60.0,
) -> dict:
    """Final burned acres + final cumulative SAIDI for one phase."""
    snaps, meta = read_run(
        store_uri,
        gis_uid=gis_uid,
        grid_uid=grid_uid,
        env_step_min=env_step_min,
        phase_uid=phase_uid,
    )
    delta_m = float(meta["delta_m"])
    last = snaps[-1]["fire_code"]
    burned_cells = int(np.count_nonzero((last == 1) | (last == 2)))
    acres = burned_cells * (delta_m * delta_m) / SQM_PER_ACRE
    return {
        "phase": phase_uid,
        "burned_acres": acres,
        "burned_cells": burned_cells,
        "cell_size_m": delta_m,
        "final_saidi": float(snaps[-1]["saidi"]),
        "saidi_trace": [float(s["saidi"]) for s in snaps],
    }


def learning_curve(
    store_uri: str,
    train_phase: str,
    agent_name: str = "firefighter",
) -> Tuple[List[int], List[float]]:
    """Per-episode mean SAIDI-reward over the training phase.

    Groups the firefighter's per-decision objectives by episode and returns
    ``(episodes, mean_reward_per_episode)`` sorted by episode. Empty lists when
    the phase stored no decisions (e.g. a store without a DRL train phase).
    """
    rows = read_agent_objectives(store_uri, agent_name, phase_uid=train_phase)
    by_ep: Dict[int, List[float]] = {}
    for r in rows:
        val = r["objective"]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        by_ep.setdefault(int(r["episode"]), []).append(float(val))
    episodes = sorted(by_ep)
    means = [float(np.mean(by_ep[e])) for e in episodes]
    return episodes, means


# ---- report assembly ----------------------------------------------------
def build_report(
    store_uri: str,
    test_phase: str,
    baseline_runs: Dict[str, str],
    *,
    env_step_min: float = 60.0,
) -> Tuple[List[dict], dict]:
    """Headline rows (baseline phases + DRL test) and the DRL test KPIs.

    The first entry of ``baseline_runs`` (insertion order) is the scripted
    baseline against which acres/SAIDI saved are computed.
    """
    rows: List[dict] = []
    for name, phase in baseline_runs.items():
        k = phase_kpis(store_uri, phase, env_step_min=env_step_min)
        k["name"] = name
        rows.append(k)
    drl = phase_kpis(store_uri, test_phase, env_step_min=env_step_min)
    drl["name"] = "drl_test"
    rows.append(drl)

    base = rows[0]
    for r in rows:
        r["acres_saved"] = base["burned_acres"] - r["burned_acres"]
        r["pct_acres_saved"] = (
            100.0 * r["acres_saved"] / base["burned_acres"]
            if base["burned_acres"] > 0
            else 0.0
        )
        r["saidi_saved"] = base["final_saidi"] - r["final_saidi"]
        r["pct_saidi_saved"] = (
            100.0 * r["saidi_saved"] / base["final_saidi"]
            if base["final_saidi"] > 0
            else 0.0
        )
    return rows, drl


def format_report(rows: List[dict]) -> str:
    base = rows[0]
    out = [
        f"DRL firefighter report (baseline = {base['name']}/{base['phase']})",
        f"cell size ~ {base['cell_size_m']:.0f} m",
        "",
        f"{'name':>12} {'phase':>20} {'burned_acres':>13} "
        f"{'final_saidi':>12} {'acres_saved':>12} {'saidi_saved':>12}",
    ]
    for r in rows:
        out.append(
            f"{r['name']:>12} {r['phase']:>20} {r['burned_acres']:>13,.1f} "
            f"{r['final_saidi']:>12.4f} {r['acres_saved']:>12,.1f} "
            f"{r['saidi_saved']:>12.4f}"
        )
    return "\n".join(out)


def latex_table(rows: List[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Run & Phase & Burned (ac) & Final SAIDI & "
        r"Acres saved & SAIDI saved \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['name']} & {r['phase'].replace('_', chr(92) + '_')} & "
            f"{r['burned_acres']:,.1f} & {r['final_saidi']:.4f} & "
            f"{r['acres_saved']:,.1f} & {r['saidi_saved']:.4f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


# ---- plots --------------------------------------------------------------
def plot_learning_curve(
    episodes: List[int], means: List[float], out_path: str
) -> Optional[str]:
    if not episodes:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    ax.plot(episodes, means, "-o", ms=3, lw=1.4, color="#c1440e")
    ax.set_xlabel("training episode")
    ax.set_ylabel("mean SAIDI-reward  ($-\\Delta$SAIDI / scale)")
    ax.set_title("DRL firefighter learning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_saidi_compare(rows: List[dict], out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    for r in rows:
        trace = r.get("saidi_trace") or []
        style = "-" if r["name"] == "drl_test" else "--"
        lw = 2.0 if r["name"] == "drl_test" else 1.2
        ax.plot(
            range(1, len(trace) + 1),
            trace,
            style,
            lw=lw,
            label=f"{r['name']} ({r['phase']})",
        )
    ax.set_xlabel("env step")
    ax.set_ylabel("cumulative SAIDI")
    ax.set_title("Cumulative SAIDI: scripted baseline vs DRL firefighter")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _parse_run(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        raise ValueError(f"--run expects NAME=PHASE, got {arg!r}")
    name, phase = arg.split("=", 1)
    return name, phase


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True, help="store URI (postgresql://)")
    ap.add_argument("--out-dir", default="_outputs/drl_report")
    ap.add_argument("--train-phase", default="phase_4_drl_train")
    ap.add_argument("--test-phase", default="phase_5_drl_test")
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PHASE",
        help="baseline/comparison NAME=PHASE in this store (repeatable; "
        "first is the scripted baseline)",
    )
    ap.add_argument("--experiment", default="eaton", help="label for outputs")
    ap.add_argument("--env-step-min", type=float, default=60.0)
    ap.add_argument("--agent-name", default="firefighter")
    ap.add_argument("--paper", action="store_true", help="also emit LaTeX")
    ap.add_argument("--conf", default=None, help="unused (store-only); accepted")
    args = ap.parse_args(argv)

    if not args.run:
        # default baseline: the first scripted firefighting phase.
        args.run = ["baseline=phase_1_air"]
    baseline_runs = dict(_parse_run(a) for a in args.run)

    os.makedirs(args.out_dir, exist_ok=True)
    rows, _drl = build_report(
        args.store,
        args.test_phase,
        baseline_runs,
        env_step_min=args.env_step_min,
    )

    txt = format_report(rows)
    print(txt)
    txt_path = os.path.join(args.out_dir, "drl_firefighter_report.txt")
    with open(txt_path, "w") as fh:
        fh.write(txt + "\n")

    episodes, means = learning_curve(
        args.store, args.train_phase, agent_name=args.agent_name
    )
    lc = plot_learning_curve(
        episodes, means, os.path.join(args.out_dir, "drl_learning_curve.png")
    )
    sc = plot_saidi_compare(
        rows, os.path.join(args.out_dir, "drl_saidi_compare.png")
    )

    written = [txt_path, sc] + ([lc] if lc else [])
    if args.paper:
        tex_path = os.path.join(args.out_dir, "drl_firefighter_report.tex")
        with open(tex_path, "w") as fh:
            fh.write(latex_table(rows) + "\n")
        written.append(tex_path)

    print("\nwrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
