"""Render a sim-vs-real perimeter validation figure for a calibrated fire.

Rebuilds the *no-firefighting* baseline via the production WildfireDriver path
(identical kwargs to verify_calibration.py, including containment_margin=2),
advances it to the experiment's final env step, and draws a single map showing:

  * hillshaded SoCal terrain (the calibration DEM crop)
  * the SIMULATED final burn scar (translucent fill)
  * the OFFICIAL CAL FIRE perimeter (bright dashed outline)
  * a metrics box (final-step Dice, Jaccard, simulated vs real acres, area %%)

This is the static counterpart to the animated comparison timelapse: it is the
one-glance "did we match the real fire?" figure.

Run:
    cd /home/user/workspace/socal-wildfires
    source .venv/bin/activate && export PYTHONPATH=$PWD
    python analysis/make_sim_vs_real.py --fire eaton     --out analysis/_v05_eaton/sim_vs_real_eaton.png
    python analysis/make_sim_vs_real.py --fire palisades --out analysis/_v05_palisades/sim_vs_real_palisades.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_cma import gis  # noqa: E402
from wildfire_cma.cma import UNBURNED  # noqa: E402
from palaestrai_socal.agents.wildfire_core import WildfireDriver  # noqa: E402
from analysis.perimeter_validation import (  # noqa: E402
    load_perimeter_polygons,
    rasterize_perimeter,
    score,
)
from analysis.make_timelapse import _hillshade  # noqa: E402

# Validated calibration parameters (mirror analysis/verify_calibration.py).
EATON_BB = (-118.1621, 34.1619, -118.0131, 34.2378)
_PAD = 0.015
_EATON_BOUNDS = (EATON_BB[0] - _PAD, EATON_BB[1] - _PAD,
                 EATON_BB[2] + _PAD, EATON_BB[3] + _PAD)
_dlat = _EATON_BOUNDS[3] - _EATON_BOUNDS[1]
_dlon = _EATON_BOUNDS[2] - _EATON_BOUNDS[0]
_midlat = (_EATON_BOUNDS[1] + _EATON_BOUNDS[3]) / 2

FIRES = {
    "eaton": dict(
        perimeter_path="data/perimeters/eaton_perimeter.geojson",
        ignition=(-118.0935761, 34.1860422),
        bounds=_EATON_BOUNDS,
        nrows=int(_dlat * 111000 / 90),
        ncols=int(_dlon * 111000 * math.cos(math.radians(_midlat)) / 90),
        base_speed=14.0, boundary_gain=0.3, moisture=0.13, kappa=1.5,
        fuel_reclass=False, containment_margin=2,
        title="SoCal Eaton Fire (Jan 2025) — simulated vs. official perimeter",
    ),
    "palisades": dict(
        perimeter_path="data/perimeters/palisades_perimeter.geojson",
        ignition=(-118.5426, 34.0781),
        bounds=(-118.7009, 34.0148, -118.4856, 34.1444),
        nrows=159, ncols=219,
        base_speed=16.0, boundary_gain=0.6, moisture=0.08, kappa=1.5,
        fuel_reclass=True, containment_margin=2,
        title="SoCal Palisades Fire (Jan 2025) — simulated vs. official perimeter",
    ),
}


def build_final_state(cfg, max_steps=60, seed=47):
    raster = gis.socal_from_srtm(
        nrows=cfg["nrows"], ncols=cfg["ncols"], bounds=cfg["bounds"], seed=seed)
    driver = WildfireDriver(
        fuel=raster.fuel, dem=raster.dem, delta_m=raster.delta_m,
        bounds=cfg["bounds"], ignition_points=[cfg["ignition"]],
        ignition_step=1, env_step_min=60.0, dt_cma_min=5.0, t_burn_steps=6,
        kappa=cfg["kappa"], dead_fuel_moisture=cfg["moisture"],
        wind_speed=cfg["base_speed"], wind_dir_deg=45.0, seed=seed,
        perimeter_path=cfg["perimeter_path"], base_speed=cfg["base_speed"],
        boundary_gain=cfg["boundary_gain"], fuel_reclass=cfg["fuel_reclass"],
        containment_margin=cfg["containment_margin"],
    )
    state = np.full((cfg["nrows"], cfg["ncols"]), UNBURNED, dtype=np.int8)
    for _ in range(max_steps):
        for (r, c, s, _layer) in driver.step(state):
            state[r, c] = s
    return raster, state


def render(fire: str, out: str):
    cfg = FIRES[fire]
    raster, state = build_final_state(cfg)
    bounds = cfg["bounds"]
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])  # lon0,lon1,lat0,lat1

    polys = load_perimeter_polygons(cfg["perimeter_path"])
    real_mask = rasterize_perimeter(polys, bounds, cfg["nrows"], cfg["ncols"])
    sim_mask = state != UNBURNED
    m = score(sim_mask, real_mask, raster.delta_m)

    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    # terrain hillshade backdrop
    hs = _hillshade(raster.dem, raster.delta_m, z_factor=2.2)
    ax.imshow(hs, cmap="Greys_r", origin="upper", extent=extent,
              alpha=0.85, zorder=0, interpolation="bilinear")
    ocean = raster.dem <= 0.0
    if ocean.any():
        ocean_rgba = np.zeros((*ocean.shape, 4))
        ocean_rgba[ocean] = (0.20, 0.34, 0.52, 0.85)
        ax.imshow(ocean_rgba, origin="upper", extent=extent, zorder=0.5,
                  interpolation="nearest")

    # simulated burn scar (translucent orange fill)
    sim_rgba = np.zeros((*sim_mask.shape, 4))
    sim_rgba[sim_mask] = (0.90, 0.30, 0.05, 0.50)
    ax.imshow(sim_rgba, origin="upper", extent=extent, zorder=2,
              interpolation="nearest")

    # official perimeter outline
    for poly in polys:
        if not poly:
            continue
        ring = poly[0]
        ax.plot(ring[:, 0], ring[:, 1], color="#101010", lw=3.4, alpha=0.55,
                solid_capstyle="round", zorder=3.8)
        ax.plot(ring[:, 0], ring[:, 1], color="#00e5ff", lw=1.9,
                ls=(0, (6, 3)), alpha=0.97, zorder=4.0)

    # ignition marker
    ax.plot(cfg["ignition"][0], cfg["ignition"][1], marker="*", ms=15,
            mfc="#ffd400", mec="#111111", mew=0.9, zorder=5,
            label="Ignition")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(cfg["title"], fontsize=12.5, fontweight="bold", pad=8)

    box = (
        f"FINAL step (60 env steps, containment_margin=2)\n"
        f"Dice        = {m['dice']:.3f}\n"
        f"Jaccard     = {m['jaccard']:.3f}\n"
        f"Sim area    = {m['sim_acres']:,.0f} ac\n"
        f"Real area   = {m['real_acres']:,.0f} ac\n"
        f"Area error  = {m['area_pct_err']:+.1f}%"
    )
    t = ax.text(0.015, 0.985, box, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, family="monospace", fontweight="bold",
                color="#111111",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#333",
                          alpha=0.92))
    t.set_zorder(6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#00e5ff", lw=1.9, ls=(0, (6, 3)),
               label="Official CAL FIRE perimeter"),
        Line2D([0], [0], marker="s", color="none", mfc=(0.90, 0.30, 0.05, 0.6),
               mec="none", ms=10, label="Simulated final burn"),
        Line2D([0], [0], marker="*", color="none", mfc="#ffd400",
               mec="#111111", ms=12, label="Ignition point"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[sim-vs-real] {fire}: Dice={m['dice']:.3f} "
          f"area%={m['area_pct_err']:+.1f}% -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", required=True, choices=sorted(FIRES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    render(args.fire, args.out)


if __name__ == "__main__":
    main()
