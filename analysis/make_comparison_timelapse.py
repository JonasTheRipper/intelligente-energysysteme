"""N-phase firefighter comparison timelapse from ONE palaestrAI store.

The v0.3 A/B/C experiment (``palaestrai_socal/experiment_eaton_local_ab.yml``)
runs the same fine-grid Eaton scenario several times into one store, varying only
the firefighter fleet size:

  * ``phase_0_no_ff``    -- firefighter ``n_planes: 0`` (baseline)
  * ``phase_1_with_ff``  -- firefighter ``n_planes: 3``
  * ``phase_2_with_ff7`` -- firefighter ``n_planes: 7``

This renderer reads ANY number of phases (via the phase-aware
:func:`analysis.store_readers.read_run`) and animates them stacked vertically,
time-synced by frame index (every phase shares ``max_steps`` so frame ``i`` is
the same simulated hour in each):

  LEFT  : one fire map PER PHASE, stacked top-to-bottom in the order given
          (e.g. TOP = no firefighters, then 3 tankers, then 7 tankers). Same
          extent/scale, hillshaded terrain reused from
          :mod:`analysis.make_timelapse`. Each firefighter phase tints its
          retardant line retardant-pink and overlays fading aero-tanker plane
          icons at the leading edge of the cells suppressed *this* step (derived
          from the per-step SUPPRESSED diff -- see :mod:`analysis.plane_icons`).
          The baseline (n_planes==0) phase shows no planes / no retardant.
  RIGHT : four stacked metric axes, one line per phase -- cumulative SAIDI,
          min/mean bus voltage p.u., served/consumed MW, and intertie MW
          (labelled a proxy when the store held no line-flow sensors).

It does NOT modify ``make_timelapse.py`` (the v0.2 single-phase renderer stays
byte-identical); it imports its basemap helpers. Backwards compatible: the legacy
``--phase-a/--phase-b`` two-phase CLI still works; the new ``--phases`` accepts a
comma-separated list of any length, each optionally tagged with its plane count
as ``uid:n`` (e.g. ``phase_0_no_ff:0,phase_1_with_ff:3,phase_2_with_ff7:7``).

Run (after the experiment has been run by the parent agent):
  python analysis/make_comparison_timelapse.py \
    --store <store-uri> \
    --phases phase_0_no_ff:0,phase_1_with_ff:3,phase_2_with_ff7:7 --stride 2
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
import matplotlib.animation as animation  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run, list_phases  # noqa: E402
from analysis.make_timelapse import _topo_basemap  # noqa: E402
from analysis.plane_icons import plane_positions  # noqa: E402

# SUPPRESSED retardant code surfaced by store_readers as fire_code==3.
SUPPRESSED_CODE = 3
# CONTAINED ground-line / point-protection code (v0.4), surfaced as fire_code==5.
CONTAINED_CODE = 5
# how many frames a plane icon lingers (fading) after it lays a drop.
PLANE_FADE_FRAMES = 3

# qualitative per-phase colours for the metric curves (baseline first).
_PHASE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                 "#17becf", "#8c564b"]


def _fire_cmap():
    """Per-tactic fire colormap (v0.4).

    Codes: 0 UNBURNED (transparent), 1 BURNING (orange), 2 BURNED_OUT (grey),
    3 SUPPRESSED retardant/wetline (pink), 4 FLOODED (blue, reserved/unused),
    5 CONTAINED ground line / point protection (dozer brown). The pink vs brown
    split colours retardant-air vs ground-containment tactics apart.
    """
    cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.33, 0.0, 0.92),
                           (0.18, 0.18, 0.18, 0.62),
                           (0.96, 0.36, 0.55, 0.90),
                           (0.20, 0.45, 0.85, 0.85),
                           (0.55, 0.36, 0.18, 0.92)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    return cmap, norm


def _suppressed_mask(snap) -> np.ndarray:
    """Boolean SUPPRESSED grid for a snap (fire_code == 3)."""
    return snap["fire_code"] == SUPPRESSED_CODE


def _precompute_plane_events(snaps, n_planes):
    """For each frame, the (row,col) plane positions newly laid that step.

    Returns ``events[i] = [(r,c), ...]`` derived from the SUPPRESSED diff
    between frame ``i-1`` and ``i``. Frame 0 has no predecessor -> []. A phase
    with ``n_planes==0`` (the baseline) yields no events at all.
    """
    events = [[] for _ in snaps]
    if n_planes <= 0:
        return events
    for i in range(1, len(snaps)):
        prev = _suppressed_mask(snaps[i - 1])
        curr = _suppressed_mask(snaps[i])
        events[i] = plane_positions(prev, curr, n_planes)
    return events


def _active_planes(events, fi, extent, nr, nc):
    """Plane icons visible at frame ``fi`` with their fade alpha.

    A drop laid at frame ``j`` stays visible for ``PLANE_FADE_FRAMES`` frames,
    its alpha decaying linearly. Returns ``[(lon, lat, alpha), ...]``.
    """
    minlon, maxlon, minlat, maxlat = extent
    out = []
    for back in range(PLANE_FADE_FRAMES):
        j = fi - back
        if j < 0:
            continue
        alpha = 1.0 - back / float(PLANE_FADE_FRAMES)
        for (r, c) in events[j]:
            lon = minlon + (c + 0.5) / nc * (maxlon - minlon)
            lat = maxlat - (r + 0.5) / nr * (maxlat - minlat)
            out.append((lon, lat, alpha))
    return out


def _draw_map(ax, meta, title):
    """Static basemap (hillshaded terrain) + an empty fire raster; return im."""
    extent = meta["extent"]
    dem = meta["dem"]
    delta_m = meta["delta_m"]
    ocean_mask = (dem <= 0.0)
    topo_rgb = _topo_basemap(dem, delta_m, ocean_mask=ocean_mask)
    ax.imshow(topo_rgb, origin="upper", extent=extent, zorder=0,
              interpolation="bilinear")
    cmap, norm = _fire_cmap()
    fire_im = ax.imshow(np.zeros((2, 2), dtype=np.int8), cmap=cmap, norm=norm,
                        origin="upper", extent=extent, zorder=3,
                        interpolation="nearest")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    return fire_im


def _phase_label(n_planes):
    """Human label for a phase given its plane count."""
    return "no firefighters" if n_planes <= 0 else f"{n_planes} aero tankers"


def _phase_short(n_planes):
    """Short legend tag for a phase."""
    return "no FF" if n_planes <= 0 else f"{n_planes} planes"


def _disp_label(p):
    """Descriptive display label for a phase, honouring a caller-supplied
    ``label`` (e.g. resource mix); falls back to the plane-count label."""
    return p.get("label") or _phase_label(p["n_planes"])


def _disp_short(p):
    """Short legend tag, honouring a caller-supplied ``label``."""
    return p.get("label") or _phase_short(p["n_planes"])


def render_comparison(
    phases, *legacy, n_planes=3, outdir=None, stride=1, fps=12, title=None,
):
    """Render an N-row firefighter timelapse to GIF (+ MP4 if ffmpeg present).

    Preferred call (N phases): ``render_comparison(phases, ...)`` where
    ``phases`` is an ordered list of dicts, each with keys:
        ``snaps``     per-step snap dicts from read_run
        ``meta``      meta dict from read_run
        ``n_planes``  fleet size for this phase (0 = baseline)
        ``uid``       phase uid (provenance only)
    Rendered top-to-bottom in the given order.

    Backwards-compatible 2-phase call (kept for older callers/tests):
    ``render_comparison(snaps_a, meta_a, snaps_b, meta_b, n_planes=3, ...)``
    -- detected when ``phases`` is not a list-of-dicts and three positional
    ``legacy`` args follow it.
    """
    # --- backwards-compat: (snaps_a, meta_a, snaps_b, meta_b, n_planes=...) ---
    if legacy and len(legacy) == 3:
        snaps_a, (meta_a, snaps_b, meta_b) = phases, legacy
        phases = [
            {"snaps": snaps_a, "meta": meta_a, "n_planes": 0,
             "uid": "phase_a"},
            {"snaps": snaps_b, "meta": meta_b, "n_planes": int(n_planes),
             "uid": "phase_b"},
        ]

    outdir = outdir or _HERE
    os.makedirs(outdir, exist_ok=True)
    if not phases:
        raise ValueError("no phases to render")

    nframes = min(len(p["snaps"]) for p in phases)
    if nframes == 0:
        raise ValueError("no frames to render (a phase is empty)")
    for p in phases:
        p["snaps"] = p["snaps"][:nframes]

    nP = len(phases)
    # use the first phase's geometry for plane lon/lat mapping (all identical).
    extent = phases[0]["meta"]["extent"]
    nr, nc = phases[0]["meta"]["dem"].shape

    frames = list(range(0, nframes, stride))
    if frames[-1] != nframes - 1:
        frames.append(nframes - 1)

    days = np.array([s["day"] for s in phases[0]["snaps"]])

    def arr(snaps, key):
        return np.array([float(s.get(key, np.nan)) for s in snaps])

    # per-phase metric arrays
    for p in phases:
        s = p["snaps"]
        p["saidi"] = arr(s, "saidi")
        p["vmin"] = arr(s, "vmin_pu")
        p["vmean"] = arr(s, "vmean_pu")
        p["mw"] = arr(s, "served_mw")
        p["tie"] = arr(s, "intertie_mw")
        p["color"] = _PHASE_COLORS[len(_PHASE_COLORS) - 1] \
            if False else _PHASE_COLORS[phases.index(p) % len(_PHASE_COLORS)]
        p["events"] = _precompute_plane_events(s, p["n_planes"])

    proxy = any(bool(p["meta"].get("intertie_is_proxy", True)) for p in phases)

    # --- figure: N map rows (left) + 4 metric axes (right) ----------------
    # Left column gets nP equal map rows; right column gets 4 equal metric axes.
    # Use a row count that is the LCM-ish max(nP, 4) so both columns tile
    # cleanly: build an (nrows x 2) gridspec where nrows = nP * 4 / gcd, then
    # span. Simpler + robust: nrows = nP * 4, maps span 4 rows each, metrics
    # span nP rows each.
    import math
    g = math.gcd(nP, 4)
    nrows = nP * 4 // g
    map_span = nrows // nP       # rows per map
    met_span = nrows // 4        # rows per metric axis

    fig_h = max(7.5, 3.0 * nP + 1.0)
    fig = plt.figure(figsize=(15.5, fig_h))
    gs = fig.add_gridspec(
        nrows, 2, width_ratios=[1.18, 1.0],
        hspace=1.15, wspace=0.22,
        left=0.055, right=0.985, top=0.92, bottom=0.06,
    )

    map_axes = []
    fire_ims = []
    plane_artist_lists = []
    hud_txts = []
    for i, p in enumerate(phases):
        ax = fig.add_subplot(gs[i * map_span:(i + 1) * map_span, 0])
        lbl = _disp_label(p)
        tag = "Baseline" if p["n_planes"] <= 0 else "Firefighters"
        fim = _draw_map(ax, p["meta"], f"{tag} — {lbl.upper()}")
        ax.set_ylabel("Latitude", fontsize=8)
        if i == nP - 1:
            ax.set_xlabel("Longitude", fontsize=8)
        else:
            # only the bottom map shows longitude tick labels; the maps share
            # the same extent, so hiding the others avoids title/label overlap
            ax.tick_params(labelbottom=False)
        # per-phase HUD (top-right of each map)
        is_base = p["n_planes"] <= 0
        hud = ax.text(0.985, 0.96, "", transform=ax.transAxes, ha="right",
                      va="top", fontsize=8.5, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.32",
                                fc="white" if is_base else "#fff3f7",
                                ec="#888" if is_base else "#c2185b",
                                alpha=0.93))
        map_axes.append(ax)
        fire_ims.append(fim)
        plane_artist_lists.append([])
        hud_txts.append(hud)

    ax_saidi = fig.add_subplot(gs[0 * met_span:1 * met_span, 1])
    ax_volt = fig.add_subplot(gs[1 * met_span:2 * met_span, 1])
    ax_mw = fig.add_subplot(gs[2 * met_span:3 * met_span, 1])
    ax_tie = fig.add_subplot(gs[3 * met_span:4 * met_span, 1])

    fig.suptitle(
        title or "SoCal Eaton Fire — firefighter response comparison",
        fontsize=15, fontweight="bold", y=0.975)

    # --- metric axes scaffolding: one line per phase ----------------------
    def _setup_axis(ax, key, ylabel, ttl, ylim_keys=None):
        lines = []
        all_vals = []
        for p in phases:
            (ln,) = ax.plot([], [], color=p["color"], lw=2.0,
                            label=_disp_short(p))
            lines.append(ln)
        srcs = ylim_keys or [key]
        for p in phases:
            for k in srcs:
                v = p[k]
                all_vals.append(v[np.isfinite(v)])
        finite = np.concatenate(all_vals) if all_vals else np.array([])
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            pad = 0.05 * (hi - lo) + 1e-6
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, days[-1] if days.size else 1.0)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", fontsize=7, framealpha=0.9, ncol=min(nP, 3))
        return lines

    saidi_lines = _setup_axis(ax_saidi, "saidi", "SAIDI (min/cust)",
                              "Cumulative SAIDI")
    # voltage: min (dashed) + mean (solid) per phase
    vmin_lines, vmean_lines = [], []
    for p in phases:
        (lmin,) = ax_volt.plot([], [], color=p["color"], lw=1.5, ls="--",
                               label=f"{_disp_short(p)} min")
        (lmean,) = ax_volt.plot([], [], color=p["color"], lw=2.0,
                                label=f"{_disp_short(p)} mean")
        vmin_lines.append(lmin)
        vmean_lines.append(lmean)
    vfin = np.concatenate([a[np.isfinite(a)] for p in phases
                           for a in (p["vmin"], p["vmean"])]
                          ) if phases else np.array([])
    if vfin.size:
        lo, hi = float(vfin.min()), float(vfin.max())
        pad = 0.05 * (hi - lo) + 1e-6
        ax_volt.set_ylim(lo - pad, hi + pad)
    ax_volt.set_xlim(0, days[-1] if days.size else 1.0)
    ax_volt.set_ylabel("Voltage (p.u.)", fontsize=8)
    ax_volt.set_title("Bus voltage (min / mean)", fontsize=9.5)
    ax_volt.grid(alpha=0.3)
    ax_volt.tick_params(labelsize=7)
    ax_volt.legend(loc="best", fontsize=6, ncol=2, framealpha=0.9)

    mw_lines = _setup_axis(ax_mw, "mw", "Served (MW)",
                           "Total served / consumed power")
    tie_ttl = "Intertie flow (MW)" + ("  [proxy]" if proxy else "")
    tie_lines = _setup_axis(ax_tie, "tie", "MW", tie_ttl)
    ax_tie.set_xlabel("Simulation time (days)", fontsize=8)

    # progress markers riding each (single-line) metric curve, per phase
    dot_groups = {}
    for ax, key in ((ax_saidi, "saidi"), (ax_mw, "mw"), (ax_tie, "tie")):
        ds = []
        for p in phases:
            (d,) = ax.plot([], [], "o", color=p["color"], ms=5, zorder=5)
            ds.append(d)
        dot_groups[ax] = (key, ds)

    def _update(fi):
        artists = []
        for pi, p in enumerate(phases):
            s = p["snaps"][fi]
            fire_ims[pi].set_data(s["fire_code"])
            artists.append(fire_ims[pi])
            # fading plane icons on firefighter maps only
            for art in plane_artist_lists[pi]:
                art.remove()
            plane_artist_lists[pi] = []
            if p["n_planes"] > 0:
                for (lon, lat, alpha) in _active_planes(
                        p["events"], fi, extent, nr, nc):
                    t = map_axes[pi].text(lon, lat, "✈", fontsize=13,
                                          ha="center", va="center",
                                          color="#101010", alpha=alpha,
                                          zorder=7)
                    plane_artist_lists[pi].append(t)
            # per-phase HUD
            if p["n_planes"] <= 0:
                hud_txts[pi].set_text(
                    f"Day {s['day']:.2f} (h{int(s['hour'])})\n"
                    f"Wind {s['wind_speed']:.0f} m/s")
            else:
                supp_n = int(s.get("suppressed_n", 0))
                cont_n = int(s.get("contained_n", 0))
                grounded = s["wind_speed"] >= 18.0
                # surface both tactics: air retardant (SUPPRESSED) and ground
                # containment line / point protection (CONTAINED). The ground
                # line is only shown when present, so tankers-only phases read
                # exactly as the v0.3 HUD did.
                cont_line = f"\nGround line: {cont_n:,} cells" if cont_n else ""
                hud_txts[pi].set_text(
                    f"Day {s['day']:.2f} (h{int(s['hour'])})\n"
                    f"Wind {s['wind_speed']:.0f} m/s\n"
                    f"Retardant: {supp_n:,} cells"
                    f"{cont_line}"
                    f"{'  (GROUNDED)' if grounded else ''}")
            artists.append(hud_txts[pi])

        upto = fi + 1
        for pi, p in enumerate(phases):
            saidi_lines[pi].set_data(days[:upto], p["saidi"][:upto])
            vmin_lines[pi].set_data(days[:upto], p["vmin"][:upto])
            vmean_lines[pi].set_data(days[:upto], p["vmean"][:upto])
            mw_lines[pi].set_data(days[:upto], p["mw"][:upto])
            tie_lines[pi].set_data(days[:upto], p["tie"][:upto])
        for ax, (key, ds) in dot_groups.items():
            for pi, p in enumerate(phases):
                ds[pi].set_data([days[fi]], [p[key][fi]])
        return artists

    ani = animation.FuncAnimation(fig, _update, frames=frames,
                                  interval=1000.0 / fps, blit=False)

    gif_path = os.path.join(outdir, "comparison_timelapse.gif")
    print(f"[compare] writing GIF -> {gif_path}")
    ani.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=90)
    print("[compare] GIF done")

    mp4_path = os.path.join(outdir, "comparison_timelapse.mp4")
    try:
        ani.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
                 dpi=120)
        print(f"[compare] MP4 done -> {mp4_path}")
    except Exception as e:
        print("[compare] MP4 skipped:", e)
        mp4_path = None

    plt.close(fig)
    return gif_path, mp4_path


def _parse_phase_spec(store, args):
    """Build the ordered phase list from CLI args.

    Priority: --phases (comma list of uid or uid:n) > legacy --phase-a/-b/-c.
    When neither names a phase, fall back to the first phases discovered, with
    plane counts inferred from the uid (``no_ff`` -> 0, ``ff7`` -> 7, else 3).
    """
    def infer_planes(uid):
        u = uid.lower()
        if "no_ff" in u or "n0" in u:
            return 0
        # trailing digits after 'ff' e.g. ff7 -> 7
        import re
        m = re.search(r"ff(\d+)", u)
        if m:
            return int(m.group(1))
        return 3

    # each spec is (uid, n_planes, label-or-None)
    specs = []
    if args.phases:
        for tok in args.phases.split(","):
            tok = tok.strip()
            if not tok:
                continue
            # grammar: uid[:n_planes[:label]] (label may contain spaces)
            parts = tok.split(":")
            uid = parts[0].strip()
            n = int(parts[1]) if len(parts) > 1 and parts[1].strip() \
                else infer_planes(uid)
            label = ":".join(parts[2:]).strip() if len(parts) > 2 else None
            specs.append((uid, n, label or None))
        return specs

    legacy = [(args.phase_a, 0), (args.phase_b, args.n_planes),
              (getattr(args, "phase_c", None), getattr(args, "n_planes_c", 7))]
    named = [(u, n, None) for (u, n) in legacy if u]
    if named:
        return named

    # nothing named: discover
    phases = list_phases(store)
    if len(phases) < 2:
        raise ValueError(
            f"store has {len(phases)} phase(s); need >=2. Found: {phases}")
    return [(p["uid"], infer_planes(p["uid"]), None) for p in phases]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True,
                    help="palaestrAI store URI / sqlite path (one run)")
    ap.add_argument("--phases", default=None,
                    help="comma list of phase uids (optionally uid:n_planes), "
                         "rendered top-to-bottom; e.g. "
                         "phase_0_no_ff:0,phase_1_with_ff:3,phase_2_with_ff7:7")
    # legacy two/three-phase flags (still supported)
    ap.add_argument("--phase-a", default=None,
                    help="[legacy] NO-firefighter phase uid (n_planes=0)")
    ap.add_argument("--phase-b", default=None,
                    help="[legacy] WITH-firefighter phase uid")
    ap.add_argument("--phase-c", default=None,
                    help="[legacy] 2nd WITH-firefighter phase uid")
    ap.add_argument("--n-planes", type=int, default=3, dest="n_planes",
                    help="[legacy] plane count for --phase-b (default 3)")
    ap.add_argument("--n-planes-c", type=int, default=7, dest="n_planes_c",
                    help="[legacy] plane count for --phase-c (default 7)")
    ap.add_argument("--gis-uid", default="gis_world")
    ap.add_argument("--grid-uid", default="socal_grid")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    specs = _parse_phase_spec(args.store, args)
    print("[compare] phases:", ", ".join(f"{u}(n={n})" for u, n, _ in specs))
    phases = []
    for uid, n, label in specs:
        snaps, meta = read_run(args.store, gis_uid=args.gis_uid,
                               grid_uid=args.grid_uid, phase_uid=uid)
        phases.append({"snaps": snaps, "meta": meta, "n_planes": n,
                       "uid": uid, "label": label})
    render_comparison(phases, outdir=args.outdir, stride=args.stride,
                      fps=args.fps, title=args.title)


if __name__ == "__main__":
    main()
