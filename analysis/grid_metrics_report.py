"""Static A/B grid-metrics comparison PNG from one palaestrAI store.

The still-image companion to :mod:`analysis.make_comparison_timelapse`. It reads
BOTH phases of the v0.3 A/B run (``phase_0_no_ff`` vs ``phase_1_with_ff``) via
the phase-aware :func:`analysis.store_readers.read_run` and emits one
multi-panel PNG comparing the baseline (A) against the with-firefighter run (B):

  * cumulative SAIDI (customer-minutes / customer)
  * min & mean bus voltage p.u.
  * total served / consumed power (MW)
  * intertie / tie-line power flow (MW; labelled a proxy when the store held no
    line-flow sensors)

Final-state deltas (acres saved, SAIDI reduction, MW preserved) are annotated.

Run:
  python analysis/grid_metrics_report.py \
    --store sqlite:///_outputs/palaestrai_v03_ab.db \
    --phase-a phase_0_no_ff --phase-b phase_1_with_ff \
    --out analysis/grid_metrics_ab.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run, list_phases  # noqa: E402

SQM_PER_ACRE = 4046.8564224


def _arr(snaps, key):
    return np.array([float(s.get(key, np.nan)) for s in snaps])


def _burned_acres(snaps, meta) -> float:
    delta_m = float(meta["delta_m"])
    last = snaps[-1]["fire_code"]
    cells = int(np.count_nonzero((last == 1) | (last == 2)))
    return cells * (delta_m * delta_m) / SQM_PER_ACRE


# distinct, color-blind-friendly-ish palette; index 0 = baseline
_PHASE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#17becf", "#8c564b",
]


def _phase_short(n_planes: int) -> str:
    """Short legend label for a phase given its plane count."""
    return "no FF" if (n_planes is None or n_planes <= 0) else f"{n_planes} planes"


def _infer_planes(uid: str) -> int:
    """Best-effort plane count from a phase uid.

    ``*_no_ff`` -> 0; a trailing ``ff<N>`` -> N; otherwise default 3.
    """
    import re
    if not uid:
        return 3
    low = uid.lower()
    if "no_ff" in low or low.endswith("_no_ff"):
        return 0
    m = re.search(r"ff(\d+)", low)
    if m:
        return int(m.group(1))
    return 3


def build_figure_n(phases, title=None):
    """Build a 4-panel comparison figure for *N* phases.

    ``phases`` is a list of dicts, each with keys ``snaps``, ``meta`` and
    (optionally) ``n_planes`` / ``uid``. Phase 0 is treated as the baseline
    for acres-saved deltas. Returns ``(fig, deltas)`` where ``deltas`` maps
    each phase label to its final-state metrics.
    """
    if not phases:
        raise ValueError("build_figure_n needs at least one phase")

    # align all phases to the shortest series
    n = min(len(p["snaps"]) for p in phases)
    for p in phases:
        p["snaps"] = p["snaps"][:n]
        if "n_planes" not in p or p["n_planes"] is None:
            p["n_planes"] = _infer_planes(p.get("uid", ""))
        p["label"] = _phase_short(p["n_planes"])
        p["acres"] = _burned_acres(p["snaps"], p["meta"])

    days = _arr(phases[0]["snaps"], "day")
    proxy = any(bool(p["meta"].get("intertie_is_proxy", True)) for p in phases)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle(title or "SoCal Eaton Fire — grid impact across firefighter "
                 "fleets", fontsize=15, fontweight="bold")
    (ax_s, ax_v), (ax_m, ax_t) = axes

    for i, p in enumerate(phases):
        c = _PHASE_COLORS[i % len(_PHASE_COLORS)]
        snaps, lab = p["snaps"], p["label"]
        ax_s.plot(days, _arr(snaps, "saidi"), color=c, lw=2.2, label=lab)
        ax_v.plot(days, _arr(snaps, "vmin_pu"), color=c, lw=1.4, ls="--",
                  label=f"{lab} min")
        ax_v.plot(days, _arr(snaps, "vmean_pu"), color=c, lw=2.2,
                  label=f"{lab} mean")
        ax_m.plot(days, _arr(snaps, "served_mw"), color=c, lw=2.2, label=lab)
        ax_t.plot(days, _arr(snaps, "intertie_mw"), color=c, lw=2.2, label=lab)

    ax_s.set_title("Cumulative SAIDI", fontsize=11)
    ax_s.set_ylabel("SAIDI (customer-min / customer)")
    ax_v.set_title("Bus voltage (min dashed / mean solid)", fontsize=11)
    ax_v.set_ylabel("Voltage (p.u.)")
    ax_m.set_title("Total served / consumed power", fontsize=11)
    ax_m.set_ylabel("Served (MW)")
    tie_ttl = "Intertie / tie-line power flow" + ("  (proxy)" if proxy else "")
    ax_t.set_title(tie_ttl, fontsize=11)
    ax_t.set_ylabel("MW (proxy = served import)" if proxy else "MW")
    if proxy:
        ax_t.annotate("proxy: no tie-line sensor stored; "
                      "intertie \u2248 served load import",
                      xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
                      va="top", fontsize=7.5, color="#555")

    for ax in (ax_s, ax_v, ax_m, ax_t):
        ax.set_xlabel("Simulation time (days)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9, ncol=2)

    # banner: per-phase final burned acres + acres saved vs baseline
    base_acres = phases[0]["acres"]
    deltas = {}
    parts = []
    for i, p in enumerate(phases):
        saved = base_acres - p["acres"]
        deltas[p["label"]] = {
            "acres": p["acres"],
            "acres_saved_vs_baseline": saved,
            "saidi_final": float(_arr(p["snaps"], "saidi")[-1]),
            "n_planes": p["n_planes"],
        }
        if i == 0:
            parts.append(f"{p['label']}: {p['acres']:,.0f} ac (baseline)")
        else:
            parts.append(f"{p['label']}: {p['acres']:,.0f} ac "
                         f"(saved {saved:,.0f})")
    fig.text(0.5, 0.005, "Final burned area  —  " + "   |   ".join(parts),
             ha="center", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, deltas


def build_figure(snaps_a, meta_a, snaps_b, meta_b, title=None):
    """Build the 4-panel A/B comparison figure; return the matplotlib Figure."""
    n = min(len(snaps_a), len(snaps_b))
    snaps_a, snaps_b = snaps_a[:n], snaps_b[:n]
    days = _arr(snaps_b, "day")

    saidi_a, saidi_b = _arr(snaps_a, "saidi"), _arr(snaps_b, "saidi")
    vmin_a, vmin_b = _arr(snaps_a, "vmin_pu"), _arr(snaps_b, "vmin_pu")
    vmean_a, vmean_b = _arr(snaps_a, "vmean_pu"), _arr(snaps_b, "vmean_pu")
    mw_a, mw_b = _arr(snaps_a, "served_mw"), _arr(snaps_b, "served_mw")
    tie_a, tie_b = _arr(snaps_a, "intertie_mw"), _arr(snaps_b, "intertie_mw")
    proxy = bool(meta_a.get("intertie_is_proxy", True)
                 or meta_b.get("intertie_is_proxy", True))

    acres_a = _burned_acres(snaps_a, meta_a)
    acres_b = _burned_acres(snaps_b, meta_b)
    acres_saved = acres_a - acres_b
    saidi_red = float(saidi_a[-1] - saidi_b[-1])
    mw_pres = float(mw_b[-1] - mw_a[-1])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle(title or "SoCal Eaton Fire — grid impact: baseline vs "
                 "firefighters", fontsize=15, fontweight="bold")
    (ax_s, ax_v), (ax_m, ax_t) = axes

    ca, cb = "#1f77b4", "#d62728"

    ax_s.plot(days, saidi_a, color=ca, lw=2.2, label="A: no firefighters")
    ax_s.plot(days, saidi_b, color=cb, lw=2.2, label="B: with firefighters")
    ax_s.set_title("Cumulative SAIDI", fontsize=11)
    ax_s.set_ylabel("SAIDI (customer-min / customer)")
    ax_s.annotate(f"SAIDI reduction: {saidi_red:,.0f}",
                  xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                        fc="#eef6ff", ec=cb))

    ax_v.plot(days, vmin_a, color=ca, lw=1.6, ls="--", label="A min")
    ax_v.plot(days, vmean_a, color=ca, lw=2.2, label="A mean")
    ax_v.plot(days, vmin_b, color=cb, lw=1.6, ls="--", label="B min")
    ax_v.plot(days, vmean_b, color=cb, lw=2.2, label="B mean")
    ax_v.set_title("Bus voltage (min / mean)", fontsize=11)
    ax_v.set_ylabel("Voltage (p.u.)")

    ax_m.plot(days, mw_a, color=ca, lw=2.2, label="A: no firefighters")
    ax_m.plot(days, mw_b, color=cb, lw=2.2, label="B: with firefighters")
    ax_m.set_title("Total served / consumed power", fontsize=11)
    ax_m.set_ylabel("Served (MW)")
    ax_m.annotate(f"MW preserved (B−A, final): {mw_pres:,.1f}",
                  xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                  fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                        fc="#eef6ff", ec=cb))

    tie_ttl = "Intertie / tie-line power flow" + ("  (proxy)" if proxy else "")
    ax_t.plot(days, tie_a, color=ca, lw=2.2, label="A: no firefighters")
    ax_t.plot(days, tie_b, color=cb, lw=2.2, label="B: with firefighters")
    ax_t.set_title(tie_ttl, fontsize=11)
    ax_t.set_ylabel("MW (proxy = served import)" if proxy else "MW")
    if proxy:
        ax_t.annotate("proxy: no tie-line sensor stored; "
                      "intertie ≈ served load import",
                      xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
                      va="top", fontsize=7.5, color="#555")

    for ax in (ax_s, ax_v, ax_m, ax_t):
        ax.set_xlabel("Simulation time (days)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # overall acres-saved banner
    fig.text(0.5, 0.005,
             f"Final burned area  A: {acres_a:,.1f} ac   B: {acres_b:,.1f} ac   "
             f"->  acres saved by firefighters: {acres_saved:,.1f}",
             ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, {
        "acres_a": acres_a, "acres_b": acres_b, "acres_saved": acres_saved,
        "saidi_reduction": saidi_red, "mw_preserved": mw_pres,
        "intertie_is_proxy": proxy,
    }


def _resolve_phases(store, phase_a, phase_b):
    if phase_a and phase_b:
        return phase_a, phase_b
    phases = list_phases(store)
    if len(phases) < 2:
        raise ValueError(
            f"store has {len(phases)} phase(s); A/B needs 2. Found: {phases}")
    return (phase_a or phases[0]["uid"], phase_b or phases[1]["uid"])


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--phases", default=None,
                    help="comma list of phase uids (or uid:n_planes) for an "
                         "N-way comparison, e.g. "
                         "phase_0_no_ff:0,phase_1_with_ff:3,phase_2_with_ff7:7")
    ap.add_argument("--phase-a", default=None,
                    help="phase uid for the baseline (default: 1st phase)")
    ap.add_argument("--phase-b", default=None,
                    help="phase uid for the with-FF run (default: 2nd phase)")
    ap.add_argument("--gis-uid", default="gis_world")
    ap.add_argument("--grid-uid", default="socal_grid")
    ap.add_argument("--out", default=os.path.join(_HERE, "grid_metrics_ab.png"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.phases:
        phases = []
        for tok in args.phases.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" in tok:
                uid, n_str = tok.rsplit(":", 1)
                n_planes = int(n_str)
            else:
                uid, n_planes = tok, None
            snaps, meta = read_run(args.store, gis_uid=args.gis_uid,
                                   grid_uid=args.grid_uid, phase_uid=uid)
            phases.append({"snaps": snaps, "meta": meta,
                           "uid": uid, "n_planes": n_planes})
        labels = [p.get("uid") for p in phases]
        print(f"[grid-metrics] {len(phases)} phases: {labels}")
        fig, deltas = build_figure_n(phases, title=args.title)
    else:
        phase_a, phase_b = _resolve_phases(args.store, args.phase_a,
                                           args.phase_b)
        print(f"[grid-metrics] phase A = {phase_a!r}  phase B = {phase_b!r}")
        snaps_a, meta_a = read_run(args.store, gis_uid=args.gis_uid,
                                   grid_uid=args.grid_uid, phase_uid=phase_a)
        snaps_b, meta_b = read_run(args.store, gis_uid=args.gis_uid,
                                   grid_uid=args.grid_uid, phase_uid=phase_b)
        fig, deltas = build_figure(snaps_a, meta_a, snaps_b, meta_b,
                                   title=args.title)
    fig.savefig(args.out, dpi=130)
    print(f"[grid-metrics] wrote {args.out}")
    print(f"[grid-metrics] deltas: {deltas}")


if __name__ == "__main__":
    main()
