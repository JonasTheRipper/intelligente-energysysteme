"""The whitelist validation figure.

Three panels, each answering one question a reviewer will ask.

(a) "Does the scenario actually do anything?"
    Burned area over time for all four scripted phases, OFF drawn solid and ON
    over it dashed. Without this the reader has no reason to believe the
    equivalence panel is measuring a live simulation rather than two empty runs
    -- and an empty run is exactly how the episode-reset bug presented.

(b) "Is the equivalence real, or is the comparison blind?"
    Per-step count of cells whose state differs -- the quantity the SHA-256
    check actually tests -- for the three comparisons. A flat line at zero IS the
    finding, which is precisely why it needs the sensitivity control on the same
    axes: NEG perturbs the fire driver's base_speed by 3.6%% with terrain and
    house count held identical, so its curve leaving zero proves the measurement
    can see a change in the trajectory itself.

    An overlay of the OFF and ON curves would NOT do this job: two identical
    lines drawn on top of each other are visually indistinguishable from having
    plotted one series twice, so the reader cannot tell a result from a bug.

(c) "Where does the speedup come from, and is it just startup?"
    Wall clock decomposed into per-phase startup and per-step cost. With four
    arms at two lengths the model ``total = 4 * (startup + steps * per_step)``
    is exactly determined for each of OFF and ON, so both terms are measured
    rather than asserted.

Usage
-----
    python analysis/whitelist_figure.py [--out FIG.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store_readers as sr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "_outputs", "whitelist_validation")
PGURI = os.environ.get(
    "PGURI", "postgresql://palaestrai:socal_local@127.0.0.1:5433"
)
BURNED = (1, 2)  # BURNING, BURNED_OUT
PHASES = ["phase_0_no_ff", "phase_1_air", "phase_2_air_ground",
          "phase_3_full_triage"]
NICE = {"phase_0_no_ff": "no response", "phase_1_air": "air",
        "phase_2_air_ground": "air+ground", "phase_3_full_triage": "full triage"}


def rasters(arm: str, phase: str) -> List[np.ndarray]:
    """The per-step cell_state raster for one phase of one arm."""
    con, ph = sr._connect(f"{PGURI}/wlval_{arm}")
    try:
        rows = sr._fetch_env_rows(con, "gis_world", ph, phase, None, None)
    finally:
        con.close()
    out = []
    for _id, dump in rows:
        st = sr._sensors_by_suffix(dump).get("gis.cell_state")
        if st is not None:
            out.append(np.asarray(st).ravel())
    return out


def burned_series(arm: str, phase: str) -> List[int]:
    return [int(np.count_nonzero(np.isin(r, BURNED))) for r in rasters(arm, phase)]


def cells_differing(a: List[np.ndarray], b: List[np.ndarray]) -> np.ndarray:
    """Per-step count of cells whose state differs between two runs.

    This is the quantity the SHA-256 equivalence check actually tests, so it is
    the honest thing to plot. The obvious alternative -- the difference in
    TOTAL burned cells -- is a coarse proxy that can read zero while the two
    rasters disagree everywhere, since a fire of the same size in a different
    place scores identically.
    """
    n = min(len(a), len(b))
    return np.array([int(np.count_nonzero(a[i] != b[i])) for i in range(n)])


def _timings() -> Dict[str, float]:
    t: Dict[str, float] = {}
    path = os.path.join(OUT_DIR, "timings.json")
    if os.path.exists(path):
        t.update({k: float(v) for k, v in json.load(open(path)).items()})
    extra = os.path.join(OUT_DIR, "timings_extra.txt")
    if os.path.exists(extra):
        for line in open(extra):
            parts = line.split()
            if len(parts) == 2:
                t[parts[0]] = float(parts[1])
    return t


def _decompose(long_s: float, short_s: float, n_phase: int = 4,
               long_steps: int = 60, short_steps: int = 8
               ) -> Optional[Tuple[float, float]]:
    """Solve total = n_phase * (startup + steps * per_step) for both terms."""
    if not long_s or not short_s:
        return None
    per_step = ((long_s - short_s) / n_phase) / (long_steps - short_steps)
    startup = long_s / n_phase - long_steps * per_step
    return startup, per_step


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="phase_1_air",
                   help="phase shown in the equivalence figure")
    p.add_argument("--outdir", default=OUT_DIR)
    ns = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 200,
    })
    os.makedirs(ns.outdir, exist_ok=True)

    # =====================================================================
    # Figure 1 -- equivalence
    # =====================================================================
    # Two panels rather than one: the left shows the scenario is a live,
    # non-trivial simulation, the right shows the runs agree cell-for-cell.
    # Either alone is unconvincing -- a flat zero line is also what two empty
    # runs produce, and a burned-area curve alone says nothing about agreement.
    fig1, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    seen = set()
    colours = {"no response": "#762a83", "with response": "#1b7837"}
    for phase in PHASES:
        try:
            off, on = burned_series("off", phase), burned_series("on", phase)
        except Exception:  # noqa: BLE001
            continue
        group = "no response" if phase == "phase_0_no_ff" else "with response"
        lab_off = lab_on = None
        if group not in seen:
            seen.add(group)
            lab_off = f"{group}, whitelist OFF"
            lab_on = f"{group}, whitelist ON"
        axa.plot(off, "-", color=colours[group], lw=2.0, label=lab_off)
        axa.plot(on, ":", color="black", lw=1.0, label=lab_on)
    axa.set_xlabel("environment step")
    axa.set_ylabel("burned cells")
    axa.set_title("(a) Fire growth, 4 scripted phases")
    axa.legend(loc="upper left", frameon=False)
    axa.grid(alpha=0.25)

    series = {}
    for arm in ("off", "off2", "on", "negb", "neg"):
        try:
            series[arm] = rasters(arm, ns.phase)
        except Exception as exc:  # noqa: BLE001
            print(f"  {arm}: unavailable ({exc})")

    for a, b, label, colour, style, lw in [
        ("negb", "neg", "control: +3.6% wind speed", "#d6604d", "--", 1.8),
        ("off", "off2", "same config, twice", "#2166ac", "-", 2.4),
        ("off", "on", "whitelist OFF vs ON", "#1b7837", "-", 2.4),
    ]:
        if a in series and b in series:
            d = cells_differing(series[a], series[b])
            axb.plot(np.arange(len(d)), d, style, color=colour, label=label, lw=lw)
            print(f"  {label:28s} max cells differing = {d.max() if len(d) else 0}")
    axb.set_xlabel("environment step")
    axb.set_ylabel("cells differing")
    axb.set_title("(b) Disagreement between runs")
    axb.legend(loc="upper left", frameon=False)
    axb.grid(alpha=0.25)

    fig1.tight_layout()
    f1 = os.path.join(ns.outdir, "fig1_equivalence.png")
    fig1.savefig(f1, bbox_inches="tight")
    print("wrote", f1)

    # =====================================================================
    # Figure 2 -- per-step cost
    # =====================================================================
    t = _timings()
    off_d = _decompose(t.get("off", 0.0), t.get("negb", 0.0))
    on_d = _decompose(t.get("on", 0.0), t.get("onb", 0.0))
    if not (off_d and on_d):
        print("  (figure 2 needs off/negb/on/onb timings)")
        return 0

    fig2, (axc, axd) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                    gridspec_kw={"width_ratios": [1, 1.25]})

    x = np.arange(2)
    w = 0.36
    axc.bar(x - w / 2, [off_d[0], off_d[1]], w, label="whitelist OFF",
            color="#2166ac")
    axc.bar(x + w / 2, [on_d[0], on_d[1]], w, label="whitelist ON",
            color="#1b7837")
    for xi, (a, b) in zip(x, [(off_d[0], on_d[0]), (off_d[1], on_d[1])]):
        axc.text(xi, max(a, b) * 1.25, f"{a / b:.2f}x", ha="center",
                 fontsize=9, fontweight="bold")
    axc.set_xticks(x)
    axc.set_xticklabels(["startup\n(per phase)", "simulation\n(per step)"])
    axc.set_ylabel("seconds")
    axc.set_yscale("log")
    axc.set_ylim(0.8, 200)
    axc.set_title("(a) Where the cost goes")
    axc.legend(frameon=False, loc="upper right")
    axc.grid(alpha=0.25, axis="y")

    # The blended speedup is not a constant of the system: it depends on how
    # many steps amortise the fixed startup. Showing that curve stops the
    # single headline ratio from being read as scenario-independent.
    steps = np.arange(1, 601)
    ratio = ((off_d[0] + steps * off_d[1]) / (on_d[0] + steps * on_d[1]))
    axd.plot(steps, ratio, color="#1b7837", lw=2.2)
    axd.axhline(off_d[1] / on_d[1], color="#d6604d", ls="--", lw=1.4,
                label=f"asymptote {off_d[1] / on_d[1]:.2f}x (per-step)")
    for n, lbl in ((60, "this study\n(60 steps)"),):
        r = (off_d[0] + n * off_d[1]) / (on_d[0] + n * on_d[1])
        axd.plot([n], [r], "o", color="black", ms=5)
        axd.annotate(f"{lbl}\n{r:.1f}x", (n, r), textcoords="offset points",
                     xytext=(12, -20), fontsize=8)
    axd.set_xscale("log")
    axd.set_xlabel("steps per episode")
    axd.set_ylabel("end-to-end speedup")
    axd.set_title("(b) Speedup vs episode length")
    axd.legend(frameon=False, loc="lower right")
    axd.grid(alpha=0.25)

    fig2.tight_layout()
    f2 = os.path.join(ns.outdir, "fig2_speedup.png")
    fig2.savefig(f2, bbox_inches="tight")
    print("wrote", f2)
    print(f"  OFF  startup={off_d[0]:6.1f}s/phase  per_step={off_d[1]:5.2f}s")
    print(f"  ON   startup={on_d[0]:6.1f}s/phase  per_step={on_d[1]:5.2f}s")
    print(f"  speedup: startup {off_d[0] / on_d[0]:.2f}x  "
          f"per-step {off_d[1] / on_d[1]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
