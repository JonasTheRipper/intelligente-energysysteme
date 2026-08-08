"""Build a wildfire/grid timelapse animation from the v0.2 palaestrAI store.

v0.2 is **store-only**: this no longer re-runs the environment. It reconstructs
every frame from the ``world_states`` rows the two-environment run wrote to the
palaestrAI store (see :mod:`analysis.store_readers`), so the animation
reflects exactly what the agent-driven co-simulation produced:

* the per-step hazard grid ``S`` (UNBURNED / BURNING / BURNED_OUT) from the
  ``gis_world`` env's ``gis.cell_state`` sensor,
* the served-MW shortfall (-> customers out -> cumulative SAIDI) derived from
  the ``socal_grid`` env's load ``p_mw`` sensors as the DamageMapperAgent sheds
  fire-affected load.

It then renders a two-panel animated GIF (and an MP4 if ffmpeg is available)
using the **unchanged v0.1 ``render()``**:

  LEFT  — GIS map: hillshaded terrain, live fire perimeter (burning + burned),
          the ignition star, and a moving clock/wind annotation.
  RIGHT — the cumulative SAIDI curve drawn progressively, with a marker riding
          the curve at the current step and a small KPI readout.

Run:
  python analysis/make_timelapse.py \
    --store postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai --stride 2
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import matplotlib.animation as animation  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run  # noqa: E402

# ignition marker for the render() star. Matches the WildfireCmaAgent default
# ``ignition_points`` in experiment_multihazard.yml (the Eaton-like LA origin).
IGNITION_LON, IGNITION_LAT = -118.13, 34.19


def capture_from_store(
    store_uri: str = "postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai",
    gis_uid: str = "gis_world",
    grid_uid: str = "socal_grid",
    env_step_min: float = 60.0,
):
    """Reconstruct ``(snaps, meta)`` for ``render()`` entirely from the store."""
    print(f"[timelapse] reading store {store_uri}")
    snaps, meta = read_run(
        store_uri, gis_uid=gis_uid, grid_uid=grid_uid, env_step_min=env_step_min
    )
    print(f"[timelapse] reconstructed {len(snaps)} frames; "
          f"baseline served = {meta['base_served']:,.0f} MW")
    return snaps, meta


def _hillshade(dem, delta_m, azdeg=315.0, altdeg=45.0, z_factor=2.0):
    """Standard Lambertian hillshade [0,1] from a DEM grid.

    azdeg: light azimuth (deg clockwise from north). altdeg: sun altitude.
    z_factor exaggerates relief so the synthetic SoCal terrain reads clearly.
    """
    gy, gx = np.gradient(dem * z_factor, delta_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(360.0 - azdeg + 90.0)
    alt = np.deg2rad(altdeg)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0)


def _topo_basemap(dem, delta_m, ocean_mask=None):
    """Blend an elevation colormap with a hillshade -> RGB topographic image."""
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    # terrain-style hypsometric tint
    lo, hi = np.percentile(dem, 2), np.percentile(dem, 98)
    norm = Normalize(vmin=lo, vmax=max(hi, lo + 1.0))
    tint = mpl.colormaps["terrain"](norm(dem))[:, :, :3]
    hs = _hillshade(dem, delta_m)[:, :, None]
    # multiplicative shading, lightened so colours stay legible
    rgb = tint * (0.45 + 0.55 * hs)
    if ocean_mask is not None:
        rgb[ocean_mask] = np.array([0.78, 0.85, 0.92])  # pale blue ocean/non-fuel
    return np.clip(rgb, 0, 1)


def render(snaps, meta, outdir=None, stride=1, fps=12, title=None):
    outdir = outdir or _HERE
    os.makedirs(outdir, exist_ok=True)
    extent = meta["extent"]
    all_lines = meta["all_lines"]
    fuel = meta["fuel"]
    dem = meta["dem"]
    delta_m = meta["delta_m"]

    # frame subset (stride) to keep the GIF light
    frames = list(range(0, len(snaps), stride))
    if frames[-1] != len(snaps) - 1:
        frames.append(len(snaps) - 1)
    print(f"[timelapse] rendering {len(frames)} frames (stride={stride})")

    days_all = np.array([s["day"] for s in snaps])
    saidi_all = np.array([s["saidi"] for s in snaps])
    saidi_max = float(saidi_all.max()) * 1.05 + 1.0

    # static line segments as a LineCollection (lon/lat) for fast redraw
    seg_list = [np.asarray(v) for v in all_lines.values()]
    line_index = list(all_lines.keys())  # parallel to seg_list

    # --- figure scaffold ---------------------------------------------------
    fig, (axm, axs) = plt.subplots(
        1, 2, figsize=(15.5, 7.2),
        gridspec_kw={"width_ratios": [1.32, 1.0]},
    )
    fig.suptitle(
        title or "SoCal Santa-Ana Wildfire — GIS spread, grid line outages & SAIDI accrual",
        fontsize=15, fontweight="bold", y=0.975)

    # ---- LEFT: GIS map static layers -------------------------------------
    # topographic basemap: hillshaded elevation (terrain hypsometric tint).
    # Non-burnable cells (fuel==0 -> ocean/urban/fuel-breaks) are tinted pale
    # blue so the coastline and water read like a real topo map.
    # Only tint genuine water (sea level / Salton Trough) as ocean -- the
    # scattered non-burnable urban/alpine cells stay on the terrain tint so
    # the basemap reads as a clean topographic map.
    ocean_mask = (dem <= 0.0)
    topo_rgb = _topo_basemap(dem, delta_m, ocean_mask=ocean_mask)
    axm.imshow(topo_rgb, origin="upper", extent=extent, zorder=0,
               interpolation="bilinear")
    # elevation contour lines for terrain readability
    nr, nc = dem.shape
    lons = np.linspace(extent[0], extent[1], nc)
    lats = np.linspace(extent[3], extent[2], nr)  # origin='upper' -> top=maxlat
    LON, LAT = np.meshgrid(lons, lats)
    try:
        cs = axm.contour(LON, LAT, dem, levels=8, colors="#5a4a36",
                         linewidths=0.35, alpha=0.45, zorder=1)
    except Exception as e:
        print("[timelapse] contour skipped:", e)
    # all transmission lines (dark, slightly bolder for topo contrast)
    base_lc = LineCollection(seg_list, colors="#33373b", linewidths=0.4,
                             alpha=0.6, zorder=2)
    axm.add_collection(base_lc)
    # failed-line overlay (updated each frame)
    fail_lc = LineCollection([], colors="#d10000", linewidths=1.3,
                             alpha=0.95, zorder=4)
    axm.add_collection(fail_lc)
    # fire raster (updated each frame): 0 transparent, 1 burning, 2 burned-out,
    # 3 SUPPRESSED (v0.3 retardant line, tinted retardant-pink). Code 3 never
    # occurs in a v0.2 run, so the burn-scar render is byte-identical there.
    # Semi-transparent burn scar so the underlying terrain stays visible and
    # you can judge whether the spread follows ridges/valleys.
    fire_cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.33, 0.0, 0.92),
                                (0.18, 0.18, 0.18, 0.62),
                                (0.96, 0.36, 0.55, 0.90)])
    fire_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], fire_cmap.N)
    fire_im = axm.imshow(np.zeros((2, 2), dtype=np.int8), cmap=fire_cmap,
                         norm=fire_norm, origin="upper", extent=extent,
                         zorder=3, interpolation="nearest")
    axm.plot([IGNITION_LON], [IGNITION_LAT], marker="*", ms=17,
             color="#1565c0", mec="white", mew=0.8, zorder=6, label="Ignition")
    axm.set_xlim(extent[0], extent[1])
    axm.set_ylim(extent[2], extent[3])
    axm.set_xlabel("Longitude")
    axm.set_ylabel("Latitude")
    axm.set_title("Fire perimeter & failed transmission lines (hillshaded terrain)",
                  fontsize=11)

    # elevation colorbar (terrain hypsometric tint reference)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    elo, ehi = float(np.percentile(dem, 2)), float(np.percentile(dem, 98))
    sm = cm.ScalarMappable(norm=Normalize(vmin=elo, vmax=ehi),
                           cmap="terrain")
    cb = fig.colorbar(sm, ax=axm, fraction=0.035, pad=0.02, shrink=0.78)
    cb.set_label("Elevation (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#ff5400", label="Active fire (burning)"),
        Patch(facecolor="#4d4d4d", label="Burned out"),
        Patch(facecolor="#f55c8c", label="Retardant line (suppressed)"),
        Line2D([0], [0], color="#d10000", lw=2.0, label="Failed line"),
        Line2D([0], [0], color="#33373b", lw=1.0, label="Healthy line"),
        Line2D([0], [0], color="#5a4a36", lw=0.8, alpha=0.6,
               label="Elevation contour"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1565c0",
               markersize=12, label="Ignition", lw=0),
    ]
    axm.legend(handles=legend_handles, loc="lower left", fontsize=8,
               framealpha=0.92)

    # clock / wind annotation
    clock_txt = axm.text(0.985, 0.975, "", transform=axm.transAxes, ha="right",
                         va="top", fontsize=10.5, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                   ec="#888", alpha=0.92))

    # ---- RIGHT: SAIDI curve ----------------------------------------------
    axs.set_xlim(0, days_all[-1])
    axs.set_ylim(0, saidi_max)
    axs.set_xlabel("Simulation time (days)")
    axs.set_ylabel("Cumulative SAIDI (customer-minutes / customer)",
                   color="#7a0000")
    axs.tick_params(axis="y", labelcolor="#7a0000")
    axs.set_title("System reliability: cumulative SAIDI", fontsize=11)
    axs.grid(alpha=0.3)
    for d in range(1, int(days_all[-1]) + 1):
        axs.axvline(d, color="gray", lw=0.6, ls=":")
    # day-phase shading (Santa-Ana window days 1-3)
    axs.axvspan(0, 3, color="#ffd9b0", alpha=0.35, zorder=0,
                label="Santa-Ana window (d1–3)")
    (saidi_line,) = axs.plot([], [], color="#7a0000", lw=2.6, zorder=3)
    (saidi_dot,) = axs.plot([], [], "o", color="#7a0000", ms=8,
                            mec="white", mew=1.0, zorder=4)
    kpi_txt = axs.text(0.03, 0.97, "", transform=axs.transAxes, va="top",
                       ha="left", fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.45", fc="#fff7f2",
                                 ec="#d10000", alpha=0.95))
    axs.legend(loc="lower right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def _update(fi):
        s = snaps[fi]
        # fire raster
        fire_im.set_data(s["fire_code"])
        # failed lines
        fl = s["failed_lines"]
        fail_segs = [all_lines[ln] for ln in fl if ln in all_lines]
        fail_segs = [np.asarray(x) for x in fail_segs]
        fail_lc.set_segments(fail_segs)
        # clock (+ a v0.3 retardant HUD line only when a line is active, so a
        # v0.2 run with no firefighter shows the exact same annotation as before)
        clock = (f"Day {s['day']:.2f}  (h{int(s['hour'])})\n"
                 f"Wind {s['wind_speed']:.0f} m/s")
        supp_n = int(s.get("suppressed_n", 0))
        if supp_n > 0:
            grounded = s["wind_speed"] >= 18.0
            clock += (f"\nRetardant: {supp_n:,} cells"
                      f"{'  (FLEET GROUNDED)' if grounded else ''}")
        clock_txt.set_text(clock)
        # SAIDI curve progressive
        upto = fi + 1
        saidi_line.set_data(days_all[:upto], saidi_all[:upto])
        saidi_dot.set_data([s["day"]], [s["saidi"]])
        kpi_txt.set_text(
            f"SAIDI: {s['saidi']:,.0f} min\n"
            f"Served: {s['served_mw']:,.0f} MW\n"
            f"Failed lines: {s['failed_line_n']:,}\n"
            f"Failed buses: {s['failed_bus_n']:,}\n"
            f"Cust. out: {s['cust_disc']:,.0f}"
        )
        return (fire_im, fail_lc, saidi_line, saidi_dot, clock_txt, kpi_txt)

    ani = animation.FuncAnimation(fig, _update, frames=frames,
                                  interval=1000.0 / fps, blit=False)

    gif_path = os.path.join(outdir, "wildfire_timelapse.gif")
    print(f"[timelapse] writing GIF -> {gif_path}")
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=90)
    print("[timelapse] GIF done")

    # try an MP4 too (smaller, smoother) if ffmpeg present
    mp4_path = os.path.join(outdir, "wildfire_timelapse.mp4")
    try:
        ani.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
                 dpi=120)
        print(f"[timelapse] MP4 done -> {mp4_path}")
    except Exception as e:
        print("[timelapse] MP4 skipped:", e)
        mp4_path = None

    plt.close(fig)
    return gif_path, mp4_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai",
                    help="palaestrAI store URI (TimescaleDB or sqlite:// path)")
    ap.add_argument("--gis-uid", default="gis_world")
    ap.add_argument("--grid-uid", default="socal_grid")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth hourly snapshot")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ignition-lon", type=float, default=None,
                    help="override ignition star longitude (scenario-specific)")
    ap.add_argument("--ignition-lat", type=float, default=None,
                    help="override ignition star latitude (scenario-specific)")
    ap.add_argument("--title", default=None,
                    help="override the figure suptitle (scenario-specific)")
    args = ap.parse_args()

    if args.ignition_lon is not None and args.ignition_lat is not None:
        globals()["IGNITION_LON"] = args.ignition_lon
        globals()["IGNITION_LAT"] = args.ignition_lat

    snaps, meta = capture_from_store(
        store_uri=args.store, gis_uid=args.gis_uid, grid_uid=args.grid_uid)
    render(snaps, meta, outdir=args.outdir, stride=args.stride, fps=args.fps,
           title=args.title)
