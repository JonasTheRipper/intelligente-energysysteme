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
import matplotlib.patheffects as pe  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run, list_phases  # noqa: E402
from analysis.make_timelapse import _topo_basemap, _hillshade  # noqa: E402
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
    """Per-tactic fire colormap (v0.4/v0.5).

    Codes: 0 UNBURNED (transparent), 1 BURNING (orange), 2 BURNED_OUT (grey),
    3 SUPPRESSED retardant/wetline (pink), 4 FLOODED (blue, reserved/unused),
    5 CONTAINED ground line / point protection (dozer brown). The pink vs brown
    split colours retardant-air vs ground-containment tactics apart.

    v0.5: every overlay alpha is reduced so the hillshaded California terrain
    reads THROUGH the fire/suppression layers (handoff requirement). Burning
    stays the most opaque (it is the live front); burned-out is the most
    translucent so the terrain under the scar is clearly visible.

    v0.6: alphas are nudged up slightly. The real Esri satellite basemap is
    darker and busier than the pale synthetic terrain the v0.5 alphas were
    tuned against, so the fire/scar layers need a touch more opacity to remain
    the clear focal content over urban/mountain imagery.
    """
    cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.30, 0.0, 0.85),
                           (0.14, 0.14, 0.14, 0.55),
                           (0.96, 0.36, 0.55, 0.80),
                           (0.20, 0.45, 0.85, 0.72),
                           (0.55, 0.36, 0.18, 0.82)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    return cmap, norm


def _topo_basemap_muted(dem, delta_m, ocean_mask=None):
    """v0.5 basemap: hypsometric terrain + hillshade tuned so relief is CLEARLY
    visible yet muted enough not to fight the semi-transparent fire overlays.

    Differs from analysis.make_timelapse._topo_basemap by (a) a stronger
    hillshade contribution (relief reads better) and (b) a mild desaturation
    toward a neutral base so the orange/pink fire layers pop against it.
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    lo, hi = np.percentile(dem, 2), np.percentile(dem, 98)
    norm = Normalize(vmin=lo, vmax=max(hi, lo + 1.0))
    tint = mpl.colormaps["terrain"](norm(dem))[:, :, :3]
    hs = _hillshade(dem, delta_m, z_factor=2.6)[:, :, None]
    # stronger multiplicative relief (0.30..1.0 vs the 0.45..1.0 base map)
    rgb = tint * (0.30 + 0.70 * hs)
    # mute: pull 32%% toward a light neutral grey so overlays stay legible
    grey = np.array([0.86, 0.86, 0.84])
    rgb = 0.68 * rgb + 0.32 * grey
    if ocean_mask is not None:
        rgb[ocean_mask] = np.array([0.80, 0.87, 0.93])  # pale blue ocean/non-fuel
    return np.clip(rgb, 0, 1)


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


def _draw_real_perimeter(ax, perimeter_polys, extent):
    """Overlay the official CAL FIRE perimeter as a bright dashed outline.

    ``perimeter_polys`` is the list-of-polygons structure returned by
    analysis.perimeter_validation.load_perimeter_polygons: each polygon is a
    list of rings, each ring an (N, 2) array of (lon, lat) vertices. Drawn at a
    high zorder so it sits above the fire raster but is thin/dashed so it never
    obscures the burn evolution. A soft dark halo underneath keeps it legible
    over both the pale terrain and the bright fire colours.
    """
    if not perimeter_polys:
        return
    for poly in perimeter_polys:
        if not poly:
            continue
        ring = poly[0]  # exterior ring only (holes are rare for fire footprints)
        lons, lats = ring[:, 0], ring[:, 1]
        # dark halo (wider, low alpha) then the bright dashed line on top
        ax.plot(lons, lats, color="#101010", lw=3.4, alpha=0.55,
                solid_capstyle="round", zorder=5.8)
        ax.plot(lons, lats, color="#00e5ff", lw=1.7, ls=(0, (6, 3)),
                alpha=0.95, zorder=6.0)


def _draw_city_labels(ax, cities, extent):
    """Annotate place names so the viewer can orient on the terrain.

    ``cities`` is a list of (name, lon, lat) tuples. Only points inside the
    map extent are drawn. A small marker dot plus a haloed label keep the text
    readable over terrain and fire alike, at a zorder above the perimeter.
    """
    if not cities:
        return
    minlon, maxlon, minlat, maxlat = extent
    for name, lon, lat in cities:
        if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
            continue
        ax.plot(lon, lat, marker="o", ms=3.6, mfc="#ffffff", mec="#111111",
                mew=0.7, zorder=7.0)
        txt = ax.annotate(
            name, xy=(lon, lat), xytext=(4, 3), textcoords="offset points",
            fontsize=7.6, fontweight="bold", color="#f7f7f7",
            ha="left", va="bottom", zorder=7.1,
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=2.1, foreground="#111111"),
            pe.Normal(),
        ])


# module-level basemap cache so we fetch satellite tiles once per extent even
# when the same extent is drawn on several map rows.
_SAT_CACHE: dict = {}


def _basemap_rgb(meta, use_satellite=True):
    """Return the RGB backdrop for a map row.

    v0.6: prefer a real Esri World Imagery satellite mosaic (shows the coastline,
    urban street grid, canyons -- fixes the v0.5 'LA basin looks like ocean'
    problem and the missing-landmark problem). Falls back to the v0.5 synthetic
    hillshade if tiles can't be fetched (offline).
    """
    extent = tuple(meta["extent"])
    if use_satellite:
        if extent in _SAT_CACHE:
            return _SAT_CACHE[extent]
        try:
            from analysis.satellite_basemap import satellite_rgb
            rgb = satellite_rgb(list(extent), with_labels=True)
            _SAT_CACHE[extent] = rgb
            return rgb
        except Exception as e:  # offline / tile server down
            print(f"[compare] satellite basemap unavailable ({e}); "
                  f"falling back to synthetic hillshade")
    dem = meta["dem"]
    delta_m = meta["delta_m"]
    return _topo_basemap_muted(dem, delta_m, ocean_mask=(dem <= 0.0))


def _draw_map(ax, meta, title, perimeter_polys=None, cities=None,
              use_satellite=True):
    """Static basemap (satellite imagery) + an empty fire raster; return im."""
    extent = meta["extent"]
    base_rgb = _basemap_rgb(meta, use_satellite=use_satellite)
    ax.imshow(base_rgb, origin="upper", extent=extent, zorder=0,
              interpolation="bilinear", aspect="auto")
    cmap, norm = _fire_cmap()
    fire_im = ax.imshow(np.zeros((2, 2), dtype=np.int8), cmap=cmap, norm=norm,
                        origin="upper", extent=extent, zorder=3,
                        interpolation="nearest", aspect="auto")
    # v0.5: static orientation overlays (real perimeter + place names).
    _draw_real_perimeter(ax, perimeter_polys, extent)
    _draw_city_labels(ax, cities, extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    # v0.5: keep the lon/lat tick labels legible (avoid the cramped, overlapping
    # default ticks on the ~0.2-degree extents used for these fires).
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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
    perimeter_polys=None, cities=None,
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

    # v0.6: size the figure so each satellite map row fills its rectangle with
    # little whitespace. The fires are wide-and-short (lon span >> lat span), so
    # we (a) give the map column a much larger width share and (b) make each row
    # only as tall as the map's aspect needs. We approximate the per-row map
    # aspect (width/height) from the extent, accounting for the cos(lat)
    # longitude foreshortening, then set the figure height from it.
    minlon, maxlon, minlat, maxlat = extent
    midlat = 0.5 * (minlat + maxlat)
    map_w_deg = (maxlon - minlon) * math.cos(math.radians(midlat))
    map_h_deg = (maxlat - minlat)
    map_aspect = max(0.6, map_w_deg / max(map_h_deg, 1e-6))  # width / height

    # left (map) column width vs right (metrics) column width, in inches.
    map_col_w = 12.6
    met_col_w = 5.2
    fig_w = map_col_w + met_col_w
    # Ideal row height to show the map at true aspect would be map_col_w /
    # map_aspect, but with nP rows stacked that gets very tall for wide-short
    # fires. Cap the per-row height so the whole figure stays reasonable; the
    # map uses aspect='auto' so a mild vertical stretch is acceptable and still
    # eliminates the big v0.5 whitespace bands above/below each map.
    row_map_h = min(map_col_w / map_aspect, 4.3)
    row_map_h = max(row_map_h, 3.0)
    # small inter-row gap; total height driven by the map rows (metrics reflow)
    row_gap = 0.5
    fig_h = max(6.0, nP * row_map_h + (nP - 1) * row_gap + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    hspace = row_gap / max(row_map_h, 1e-6) * 1.15
    gs = fig.add_gridspec(
        nrows, 2, width_ratios=[map_col_w, met_col_w],
        hspace=hspace, wspace=0.16,
        left=0.045, right=0.988, top=0.93, bottom=0.055,
    )

    map_axes = []
    fire_ims = []
    plane_artist_lists = []
    hud_txts = []
    for i, p in enumerate(phases):
        ax = fig.add_subplot(gs[i * map_span:(i + 1) * map_span, 0])
        lbl = _disp_label(p)
        tag = "Baseline" if p["n_planes"] <= 0 else "Firefighters"
        fim = _draw_map(ax, p["meta"], f"{tag} — {lbl.upper()}",
                        perimeter_polys=perimeter_polys, cities=cities)
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

    # v0.5: orientation legend (real perimeter + place names) on the top map.
    if (perimeter_polys or cities) and map_axes:
        from matplotlib.lines import Line2D
        leg_handles = []
        if perimeter_polys:
            leg_handles.append(Line2D([0], [0], color="#00e5ff", lw=1.7,
                                      ls=(0, (6, 3)),
                                      label="Official CAL FIRE perimeter"))
        if cities:
            leg_handles.append(Line2D([0], [0], marker="o", color="none",
                                      mfc="#ffffff", mec="#111111", mew=0.7,
                                      ms=5, label="Place / city"))
        map_axes[0].legend(handles=leg_handles, loc="upper left", fontsize=7,
                           framealpha=0.88, borderpad=0.4, handlelength=1.8)

    ax_saidi = fig.add_subplot(gs[0 * met_span:1 * met_span, 1])
    ax_volt = fig.add_subplot(gs[1 * met_span:2 * met_span, 1])
    ax_mw = fig.add_subplot(gs[2 * met_span:3 * met_span, 1])
    ax_tie = fig.add_subplot(gs[3 * met_span:4 * met_span, 1])

    fig.suptitle(
        title or "SoCal Eaton Fire — firefighter response comparison",
        fontsize=15, fontweight="bold", y=0.975)

    # v0.6: attribution for the Esri World Imagery basemap (terms of use).
    fig.text(0.045, 0.012,
             "Basemap imagery: Esri, Maxar, Earthstar Geographics",
             fontsize=6.5, color="#555555", ha="left", va="bottom")

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
    ap.add_argument("--perimeter", default=None,
                    help="GeoJSON of the official fire perimeter to overlay "
                         "(dashed line) on every map row for orientation")
    ap.add_argument("--cities", default=None,
                    help="semicolon list of place labels 'Name,lon,lat'; e.g. "
                         "'Altadena,-118.131,34.190;Pasadena,-118.145,34.156'")
    args = ap.parse_args()

    # optional orientation overlays
    perimeter_polys = None
    if args.perimeter:
        from analysis.perimeter_validation import load_perimeter_polygons
        perimeter_polys = load_perimeter_polygons(args.perimeter)
        print(f"[compare] perimeter overlay: {args.perimeter} "
              f"({len(perimeter_polys)} polygon(s))")
    cities = None
    if args.cities:
        cities = []
        for token in args.cities.split(";"):
            token = token.strip()
            if not token:
                continue
            parts = token.split(",")
            if len(parts) != 3:
                raise ValueError(f"bad --cities entry (want Name,lon,lat): {token}")
            name = parts[0].strip()
            cities.append((name, float(parts[1]), float(parts[2])))
        print(f"[compare] city labels: {', '.join(c[0] for c in cities)}")

    specs = _parse_phase_spec(args.store, args)
    print("[compare] phases:", ", ".join(f"{u}(n={n})" for u, n, _ in specs))
    phases = []
    for uid, n, label in specs:
        snaps, meta = read_run(args.store, gis_uid=args.gis_uid,
                               grid_uid=args.grid_uid, phase_uid=uid)
        phases.append({"snaps": snaps, "meta": meta, "n_planes": n,
                       "uid": uid, "label": label})
    render_comparison(phases, outdir=args.outdir, stride=args.stride,
                      fps=args.fps, title=args.title,
                      perimeter_polys=perimeter_polys, cities=cities)


if __name__ == "__main__":
    main()
