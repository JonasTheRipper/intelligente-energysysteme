"""Standalone cost-decomposition figure for the whitelist half-pager.

One panel: per-phase startup and per-step simulation cost, whitelist off vs on.
Vector PDF for LaTeX, plus a PNG for quick viewing.

Why this panel alone carries the result
---------------------------------------
The equivalence claim is better made as a table than a plot -- two identical
trajectories drawn on the same axes are one visible line, which reads as a
plotting error rather than a finding. What a plot *can* show that a number
cannot is that the speedup is confined to one of the two cost terms: startup is
untouched, and the whole gain sits in the per-step term. That is the reason the
end-to-end ratio depends on episode length, and it is the thing a reader would
otherwise have to take on trust from a single headline number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "_outputs", "whitelist_validation")


def _timings():
    t = {}
    p = os.path.join(OUT_DIR, "timings.json")
    if os.path.exists(p):
        t.update({k: float(v) for k, v in json.load(open(p)).items()})
    e = os.path.join(OUT_DIR, "timings_extra.txt")
    if os.path.exists(e):
        for line in open(e):
            f = line.split()
            if len(f) == 2:
                t[f[0]] = float(f[1])
    return t


def _decompose(long_s, short_s, n_phase=4, long_steps=60, short_steps=8):
    per_step = ((long_s - short_s) / n_phase) / (long_steps - short_steps)
    return long_s / n_phase - long_steps * per_step, per_step


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(OUT_DIR, "fig_cost"))
    ns = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    t = _timings()
    for need in ("off", "negb", "on", "onb"):
        if need not in t:
            sys.exit(f"missing timing for arm {need!r}; run the validation first")
    off = _decompose(t["off"], t["negb"])
    on = _decompose(t["on"], t["onb"])

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, off, w, label="whitelist off", color="#4a6b8a",
           edgecolor="none")
    ax.bar(x + w / 2, on, w, label="whitelist on", color="#1b7837",
           edgecolor="none")
    for xi, (a, b) in zip(x, zip(off, on)):
        ax.text(xi, max(a, b) * 1.45, f"{a / b:.2f}$\\times$", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["startup\n(per phase)", "simulation\n(per step)"])
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 260)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, axis="y", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ns.out}.{ext}", bbox_inches="tight", dpi=300)
        print("wrote", f"{ns.out}.{ext}")
    print(f"  off startup={off[0]:.1f}s per_step={off[1]:.2f}s")
    print(f"  on  startup={on[0]:.1f}s per_step={on[1]:.2f}s")
    print(f"  ratios: startup {off[0]/on[0]:.2f}x  per-step {off[1]/on[1]:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
