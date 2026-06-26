#!/usr/bin/env python3
"""
Build an interactive geo-referenced folium map of the SoCal pandapower model
and its power-flow results.

  - Buses are drawn as circles, coloured by voltage level, sized by load.
  - Lines are drawn with their real geometry, coloured by power-flow loading %.
  - Generators / interties are flagged with markers.
  - Layer control lets you toggle voltage levels, MV feeders, generation, etc.

Usage:
    python make_map.py --net socal_grid_solved.json --out socal_grid_map.html
"""
import os, sys, json, argparse
import numpy as np
import pandapower as pp
import folium
from folium.plugins import MarkerCluster

# voltage -> colour
VCOLORS = {500:"#d73027", 287:"#f46d43", 230:"#fc8d59", 220:"#fc8d59",
           161:"#fee08b", 138:"#fee08b", 115:"#d9ef8b", 92:"#a6d96a",
           70:"#66bd63", 69:"#66bd63", 66:"#1a9850", 60:"#1a9850",
           55:"#66c2a5", 34.5:"#3288bd", 34:"#3288bd", 33:"#3288bd",
           12:"#5e4fa2"}


def vcolor(kv):
    best = min(VCOLORS, key=lambda v: abs(v - kv))
    return VCOLORS[best]


def loading_color(pct):
    if pct is None or np.isnan(pct):
        return "#999999"
    if pct < 30:   return "#1a9850"
    if pct < 60:   return "#fee08b"
    if pct < 90:   return "#f46d43"
    return "#d73027"


def _bus_xy(net, b):
    g = net.bus.at[b, "geo"]
    if isinstance(g, str) and g:
        c = json.loads(g)["coordinates"]
        return c[0], c[1]
    return None


def build_map(net, out_html):
    solved = len(net.res_bus) > 0 and net.res_bus["vm_pu"].notna().any()

    # center
    xs, ys = [], []
    for b in net.bus.index:
        xy = _bus_xy(net, b)
        if xy: xs.append(xy[0]); ys.append(xy[1])
    center = [float(np.median(ys)), float(np.median(xs))]
    m = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron",
                   control_scale=True)

    # group buses by voltage tier for the layer control
    tiers = {"EHV (>=287 kV)": lambda v: v >= 287,
             "HV (115-230 kV)": lambda v: 115 <= v < 287,
             "Sub-transmission (33-115 kV)": lambda v: 33 <= v < 115,
             "MV feeders (~12 kV)": lambda v: v < 33}
    bus_layers = {name: folium.FeatureGroup(name=f"Buses: {name}", show=(v(500) or "MV" not in name))
                  for name, v in tiers.items()}

    # ---- lines ----
    line_layer = folium.FeatureGroup(name="Lines (by loading %)", show=True)
    mv_line_layer = folium.FeatureGroup(name="MV feeders", show=False)
    for li in net.line.index:
        g = net.line.at[li, "geo"]
        if not (isinstance(g, str) and g):
            continue
        coords = json.loads(g)["coordinates"]
        latlon = [[c[1], c[0]] for c in coords]
        is_mv = bool(net.line.at[li, "mv_layer"]) if "mv_layer" in net.line.columns else False
        if solved:
            pct = float(net.res_line.at[li, "loading_percent"]) if li in net.res_line.index else None
            col = loading_color(pct)
            tip = f"{net.line.at[li,'name']}<br>loading {pct:.1f}%" if pct is not None else net.line.at[li,'name']
        else:
            kv = net.bus.at[net.line.at[li,"from_bus"], "vn_kv"]
            col = vcolor(kv); tip = f"{net.line.at[li,'name']} ({kv:.0f} kV)"
        pl = folium.PolyLine(latlon, color=col, weight=2 if not is_mv else 1,
                             opacity=0.8, tooltip=tip)
        pl.add_to(mv_line_layer if is_mv else line_layer)

    # ---- buses ----
    for b in net.bus.index:
        xy = _bus_xy(net, b)
        if not xy: continue
        lon, lat = xy
        kv = float(net.bus.at[b, "vn_kv"])
        load_p = net.load[(net.load.bus == b) & (net.load.in_service)]["p_mw"].sum()
        r = 2 + min(load_p / 40.0, 8)
        if solved and b in net.res_bus.index and not np.isnan(net.res_bus.at[b,"vm_pu"]):
            vm = net.res_bus.at[b, "vm_pu"]
            popup = (f"<b>{net.bus.at[b,'name']}</b><br>{kv:.0f} kV<br>"
                     f"V = {vm:.3f} pu<br>load = {load_p:.1f} MW")
        else:
            popup = f"<b>{net.bus.at[b,'name']}</b><br>{kv:.0f} kV<br>load = {load_p:.1f} MW"
        # pick tier layer
        for name, test in tiers.items():
            if test(kv):
                layer = bus_layers[name]; break
        folium.CircleMarker([lat, lon], radius=r, color=vcolor(kv),
                            fill=True, fill_opacity=0.7, weight=0.5,
                            popup=folium.Popup(popup, max_width=250)).add_to(layer)

    # ---- generation / interties ----
    gen_layer = folium.FeatureGroup(name="Generation & interties", show=True)
    for i in net.ext_grid.index:
        b = net.ext_grid.at[i, "bus"]; xy = _bus_xy(net, b)
        if xy:
            folium.Marker([xy[1], xy[0]], icon=folium.Icon(color="red", icon="bolt", prefix="fa"),
                          tooltip=f"Intertie/slack: {net.ext_grid.at[i,'name']}").add_to(gen_layer)
    # large plant reference sgens
    if "type" in net.sgen.columns:
        plant = net.sgen[net.sgen["type"].isin(["plant_ref"]) | (net.sgen["sn_mva"] > 200)]
        for i in plant.index:
            b = net.sgen.at[i, "bus"]; xy = _bus_xy(net, b)
            if xy:
                folium.CircleMarker([xy[1], xy[0]], radius=4, color="#000000",
                                    fill=True, fill_color="#ffd700", fill_opacity=0.9,
                                    tooltip=f"{net.sgen.at[i,'name']} ({net.sgen.at[i,'sn_mva']:.0f} MVA)"
                                    ).add_to(gen_layer)

    for layer in bus_layers.values(): layer.add_to(m)
    line_layer.add_to(m); mv_line_layer.add_to(m); gen_layer.add_to(m)

    # legend
    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border:1px solid #888;
                border-radius:6px; font: 12px sans-serif; box-shadow:0 1px 4px rgba(0,0,0,.3)">
    <b>Voltage level</b><br>
    <span style="color:#d73027">&#9679;</span> 500 kV &nbsp;
    <span style="color:#fc8d59">&#9679;</span> 220-230 kV<br>
    <span style="color:#d9ef8b">&#9679;</span> 115 kV &nbsp;
    <span style="color:#1a9850">&#9679;</span> 60-69 kV<br>
    <span style="color:#3288bd">&#9679;</span> 33-34.5 kV &nbsp;
    <span style="color:#5e4fa2">&#9679;</span> 12 kV<br>
    <b>Line loading</b> (if solved)<br>
    <span style="color:#1a9850">&#8212;</span> &lt;30% &nbsp;
    <span style="color:#fee08b">&#8212;</span> 30-60% &nbsp;
    <span style="color:#f46d43">&#8212;</span> 60-90% &nbsp;
    <span style="color:#d73027">&#8212;</span> &gt;90%
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"  map -> {out_html}  ({len(net.bus)} buses, {len(net.line)} lines, solved={solved})")
    return out_html


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(here, "socal_grid_solved.json"))
    ap.add_argument("--out", default=os.path.join(here, "socal_grid_map.html"))
    a = ap.parse_args()
    net = pp.from_json(a.net)
    build_map(net, a.out)
