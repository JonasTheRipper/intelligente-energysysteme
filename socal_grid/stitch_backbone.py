#!/usr/bin/env python3
"""
Backbone stitching: the CEC line dataset leaves each voltage layer broken into
many disconnected islands (because line endpoints don't meet at substations).
Electrically the real grid is a connected mesh at each transmission voltage.

This module reconnects the islands of a given voltage layer by repeatedly adding
an equivalent line between the two geographically-closest buses that belong to
different islands of that voltage, until the layer is a single component
(or no pair is closer than max_link_km). These synthetic links represent the
real transmission corridors the GIS data failed to stitch at shared substations.

Each added line uses the standard per-voltage parameters and its true geodesic
length, and is flagged net.line.synthetic = True so it can be inspected/toggled.
"""
import os, sys, math, json
import numpy as np
import networkx as nx
import pandapower as pp

sys.path.insert(0, os.path.dirname(__file__))
from electrical_params import get_line_params

M = (88000.0, 111000.0)  # m per deg lon/lat ~34N
EARTH_R = 6371.0

def _haversine_km(a, b):
    p = math.pi/180
    dlat = (b[1]-a[1])*p; dlon=(b[0]-a[0])*p
    h = math.sin(dlat/2)**2 + math.cos(a[1]*p)*math.cos(b[1]*p)*math.sin(dlon/2)**2
    return 2*EARTH_R*math.asin(math.sqrt(h))

def _bus_coords(net):
    coords = {}
    for b in net.bus.index:
        g = net.bus.at[b, "geo"]
        if isinstance(g, str) and g:
            coords[b] = tuple(json.loads(g)["coordinates"])
    return coords

def _layer_graph(net, buses):
    g = nx.Graph(); g.add_nodes_from(buses)
    bset = set(buses)
    for _, r in net.line.iterrows():
        if r.from_bus in bset and r.to_bus in bset:
            g.add_edge(r.from_bus, r.to_bus)
    # transformers don't count (different voltage) — layer connectivity only via same-V lines
    return g

def stitch_voltage_layer(net, kv, xy, max_link_km=80.0, verbose=True):
    buses = list(net.bus[net.bus.vn_kv.round() == kv].index)
    buses = [b for b in buses if b in xy]
    if len(buses) < 2:
        return 0
    params = get_line_params(int(round(kv)), underground=False)
    added = 0
    while True:
        g = _layer_graph(net, buses)
        comps = [c for c in nx.connected_components(g)]
        if len(comps) <= 1:
            break
        comps.sort(key=len, reverse=True)
        main = comps[0]
        # find closest pair (main-island bus, other-island bus)
        main_arr = np.array([xy[b] for b in main]); main_ids = list(main)
        best = None
        for c in comps[1:]:
            for b in c:
                lon, lat = xy[b]
                d2 = (main_arr[:,0]-lon)**2 + (main_arr[:,1]-lat)**2
                j = int(np.argmin(d2))
                dkm = _haversine_km(xy[b], xy[main_ids[j]])
                if best is None or dkm < best[0]:
                    best = (dkm, b, main_ids[j])
        if best is None:
            break
        dkm, b1, b2 = best
        if dkm > max_link_km:
            # remaining islands are too far -> likely genuinely separate; stop
            if verbose:
                print(f"    {kv}kV: stop, nearest island {dkm:.0f} km > {max_link_km} km "
                      f"({len(comps)-1} islands left)")
            break
        li = pp.create_line_from_parameters(
            net, from_bus=b1, to_bus=b2, length_km=max(dkm, 0.3),
            r_ohm_per_km=params["r"], x_ohm_per_km=params["x"],
            c_nf_per_km=params["c"], max_i_ka=params["imax"],
            name=f"SYNTH {kv}kV link", type="ol")
        net.line.at[li, "kv_class"] = kv
        net.line.at[li, "synthetic"] = True
        net.line.at[li, "geo"] = json.dumps(
            {"type": "LineString", "coordinates": [list(xy[b1]), list(xy[b2])]})
        added += 1
    if verbose and added:
        print(f"    {kv}kV: added {added} synthetic links")
    return added

def stitch_all(net, voltages=(500, 230, 220, 115, 92, 66, 69, 70), max_link_km=80.0):
    if "synthetic" not in net.line.columns:
        net.line["synthetic"] = False
    xy = _bus_coords(net)
    total = 0
    for kv in voltages:
        total += stitch_voltage_layer(net, kv, xy, max_link_km=max_link_km)
    print(f"  backbone stitching added {total} synthetic links")
    return total
