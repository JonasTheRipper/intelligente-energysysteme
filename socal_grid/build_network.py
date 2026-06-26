#!/usr/bin/env python3
"""
Build a geo-referenced pandapower model of the Southern California grid
(transmission + sub-transmission backbone) from CEC GIS data + OSM substations,
with population-weighted loads and generators snapped to buses.

KEY IDEA — substation-anchored node clustering
-----------------------------------------------
The CEC transmission-line dataset is drawn for display, so line endpoints that
should meet at a substation are often tens-to-hundreds of metres apart and do
NOT share coordinates. A naive exact-coordinate snap therefore yields thousands
of disconnected fragments.

To recover a connected electrical topology we build a single set of "nodes"
(physical sites) by spatially clustering ALL line endpoints together with the
OSM substations, using a distance threshold (CLUSTER_RADIUS_M). Every line
endpoint is then snapped to its cluster centroid. A node may host several
voltage levels; each (node, voltage) pair becomes one pandapower bus, and
co-located voltage levels at a node are coupled with transformers.

Geo-referencing: bus coordinates are stored via pandapower's `geodata=` (GeoJSON
Point in net.bus.geo) AND a classic net.bus_geodata table; line geometry is
stored in net.line.geo (GeoJSON LineString) and net.line_geodata.
"""
import json, math, os, sys
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pandapower as pp

sys.path.insert(0, os.path.dirname(__file__))
from electrical_params import (nearest_voltage, get_line_params,
                               transformer_rating_mva, STD_VOLTAGES)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "model")

# ---- tunables -------------------------------------------------------------
CLUSTER_RADIUS_M = 800.0    # endpoints/substations within this are one node
MIN_VOLTAGE = 33            # keep >=33 kV in the backbone layer
REGIONAL_PEAK_MW = 35000.0  # SoCal (SCE+LADWP+SDG&E) representative coincident peak
LOAD_POWER_FACTOR = 0.98
EARTH_R = 6371.0
M_PER_DEG_LAT = 111000.0
M_PER_DEG_LON = 88000.0     # ~ at 34 N

# ---- helpers --------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    p = math.pi / 180
    a = (math.sin((lat2-lat1)*p/2)**2 +
         math.cos(lat1*p)*math.cos(lat2*p)*math.sin((lon2-lon1)*p/2)**2)
    return 2*EARTH_R*math.asin(math.sqrt(a))

def line_length_km(coords):
    L = 0.0
    for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]):
        L += haversine(x1,y1,x2,y2)
    return L

def iter_line_coords(geom):
    if geom["type"] == "LineString":
        yield geom["coordinates"]
    elif geom["type"] == "MultiLineString":
        for part in geom["coordinates"]:
            yield part

def load_geojson(path):
    return json.load(open(path))["features"]


# ---- spatial clustering of nodes -----------------------------------------
class NodeClusters:
    """Grid-bucketed union of points into clusters within CLUSTER_RADIUS_M.
    Assigns each query point to an existing cluster centroid or creates a new one."""
    def __init__(self, radius_m=CLUSTER_RADIUS_M):
        self.r = radius_m
        self.cell_lon = radius_m / M_PER_DEG_LON
        self.cell_lat = radius_m / M_PER_DEG_LAT
        self.grid = defaultdict(list)   # cell -> list of cluster ids
        self.centroids = []             # cluster id -> [lon, lat]
        self.counts = []                # cluster id -> n points

    def _cell(self, lon, lat):
        return (int(lon // self.cell_lon), int(lat // self.cell_lat))

    def _dist_m(self, a, lon, lat):
        return math.hypot((a[0]-lon)*M_PER_DEG_LON, (a[1]-lat)*M_PER_DEG_LAT)

    def assign(self, lon, lat):
        cx, cy = self._cell(lon, lat)
        best, bestd = None, self.r
        for i in (cx-1, cx, cx+1):
            for j in (cy-1, cy, cy+1):
                for cid in self.grid.get((i, j), ()):
                    d = self._dist_m(self.centroids[cid], lon, lat)
                    if d <= bestd:
                        best, bestd = cid, d
        if best is None:
            cid = len(self.centroids)
            self.centroids.append([lon, lat]); self.counts.append(1)
            self.grid[(cx, cy)].append(cid)
            return cid
        # update running centroid
        n = self.counts[best]
        c = self.centroids[best]
        c[0] = (c[0]*n + lon)/(n+1); c[1] = (c[1]*n + lat)/(n+1)
        self.counts[best] = n+1
        return best


# ---- main build -----------------------------------------------------------
def build():
    print("Loading data...")
    lines = load_geojson(os.path.join(DATA, "socal_transmission_lines.geojson"))
    plants = load_geojson(os.path.join(DATA, "socal_power_plants.geojson"))
    cities = load_geojson(os.path.join(DATA, "ca_cities_population.geojson"))
    subs = load_geojson(os.path.join(DATA, "socal_substations_osm.geojson"))

    # 1) Seed clusters with substations (authoritative sites), then add line endpoints.
    print(f"Clustering nodes (radius {CLUSTER_RADIUS_M:.0f} m)...")
    clusters = NodeClusters()
    for f in subs:
        lon, lat = f["geometry"]["coordinates"][:2]
        clusters.assign(lon, lat)
    n_after_sub = len(clusters.centroids)

    # gather usable line segments first (filter status/voltage), remember endpoints
    segs = []  # (kv_std, ug, coords, name, owner)
    for f in lines:
        p = f["properties"]
        status = (p.get("Status") or "").lower()
        if status not in ("operational", "", "unknown"):
            continue
        kv_std = nearest_voltage(p.get("kV"))
        if kv_std is None or kv_std < MIN_VOLTAGE:
            continue
        ug = (p.get("Type") or "").strip().lower() == "ug"
        nm = p.get("Name") or p.get("TLine_Name")
        ow = p.get("Owner")
        for coords in iter_line_coords(f["geometry"]):
            if len(coords) >= 2:
                segs.append((kv_std, ug, [c[:2] for c in coords], nm, ow))

    # assign endpoints to clusters (creates new clusters where no substation nearby)
    for kv_std, ug, coords, nm, ow in segs:
        clusters.assign(*coords[0])
        clusters.assign(*coords[-1])
    print(f"  {n_after_sub} clusters from substations -> {len(clusters.centroids)} total nodes")

    net = pp.create_empty_network(name="SoCal Grid (CEC/OSM approx)", sn_mva=100.0)

    node_buses = {}                    # (cluster_id, kv_std) -> bus idx
    node_voltages = defaultdict(set)   # cluster_id -> {kv_std}

    def cluster_of(lon, lat):
        return clusters.assign(lon, lat)  # idempotent: returns existing nearby cluster

    def get_bus(cid, kv_std):
        key = (cid, kv_std)
        if key not in node_buses:
            lon, lat = clusters.centroids[cid]
            b = pp.create_bus(net, vn_kv=float(kv_std),
                              name=f"node{cid}@{kv_std}kV",
                              geodata=(lon, lat), type="b")
            node_buses[key] = b
            node_voltages[cid].add(kv_std)
        return node_buses[key]

    # 2) create lines + buses
    print("Building buses & lines...")
    n_lines = 0
    line_geo = {}
    for kv_std, ug, coords, nm, ow in segs:
        ca = cluster_of(*coords[0]); cb = cluster_of(*coords[-1])
        if ca == cb:
            continue  # both ends in same site -> internal, skip
        ba = get_bus(ca, kv_std); bb = get_bus(cb, kv_std)
        params = get_line_params(kv_std, underground=ug)
        length = max(line_length_km(coords), 0.1)
        li = pp.create_line_from_parameters(
            net, from_bus=ba, to_bus=bb, length_km=length,
            r_ohm_per_km=params["r"], x_ohm_per_km=params["x"],
            c_nf_per_km=params["c"], max_i_ka=params["imax"],
            name=nm or f"line_{n_lines}", type="ol" if not ug else "cs")
        net.line.at[li, "kv_class"] = kv_std
        net.line.at[li, "owner"] = ow
        line_geo[li] = [[float(x), float(y)] for x, y in coords]
        n_lines += 1
    print(f"  created {len(net.bus)} buses, {n_lines} lines")

    # 3) transformers between co-located voltage levels
    print("Adding transformers...")
    n_trafo = 0
    for cid, volts in node_voltages.items():
        if len(volts) < 2:
            continue
        vs = sorted(volts, reverse=True)
        for hv, lv in zip(vs[:-1], vs[1:]):
            hb = node_buses[(cid, hv)]; lb = node_buses[(cid, lv)]
            sn = transformer_rating_mva(hv, lv)
            pp.create_transformer_from_parameters(
                net, hv_bus=hb, lv_bus=lb, sn_mva=sn,
                vn_hv_kv=float(hv), vn_lv_kv=float(lv),
                vkr_percent=0.4, vk_percent=12.0, pfe_kw=0.0, i0_percent=0.05,
                name=f"trafo {hv}/{lv} node{cid}")
            n_trafo += 1
    print(f"  created {n_trafo} transformers")

    _attach_line_geodata(net, line_geo)

    # 3b) stitch fragmented voltage layers into connected backbones
    print("Stitching fragmented transmission backbone...")
    from stitch_backbone import stitch_all
    stitch_all(net)

    # 4) largest connected component
    print("Extracting largest connected component...")
    import pandapower.topology as top
    import networkx as nx
    mg = top.create_nxgraph(net, respect_switches=False)
    comps = sorted(nx.connected_components(mg), key=len, reverse=True)
    sizes = [len(c) for c in comps[:8]]
    print(f"  {len(comps)} components; top sizes {sizes}")
    keep = comps[0]
    drop = [b for b in net.bus.index if b not in keep]
    pp.drop_buses(net, drop)
    print(f"  kept {len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} trafos")

    # 5) generators, slack, loads
    print("Attaching generators...")
    _attach_generators(net, plants)
    _add_slack(net)
    print("Disaggregating population-weighted loads...")
    _add_loads(net, cities)
    return net, line_geo


def _attach_line_geodata(net, line_geo):
    import json as _json
    for li, coords in line_geo.items():
        if li in net.line.index:
            net.line.at[li, "geo"] = _json.dumps({"type": "LineString", "coordinates": coords})


def _bus_coords(net):
    import json as _json
    coords = {}
    for b in net.bus.index:
        g = net.bus.at[b, "geo"]
        if isinstance(g, str) and g:
            coords[b] = tuple(_json.loads(g)["coordinates"])
    return coords


def _attach_generators(net, plants):
    bus_xy = _bus_coords(net)
    bus_ids = list(bus_xy.keys())
    bus_arr = np.array([bus_xy[b] for b in bus_ids])
    vn = net.bus.loc[bus_ids, "vn_kv"].values
    DISPATCHABLE = {"NG","WAT","NUC","GEO","BIT","DFO","GAS","OG","OGW",
                    "PC","MSW","LFG","OBG","WDS","HBD","WH","OIL","SUB"}
    agg = defaultdict(lambda: {"p":0.0,"n":0,"fuel":Counter()})
    for f in plants:
        p = f["properties"]
        if p.get("Retired_Plant"):
            continue
        try: cap = float(p.get("Capacity_Latest") or 0)
        except (TypeError, ValueError): cap = 0.0
        if cap <= 0:
            continue
        lon, lat = f["geometry"]["coordinates"][:2]
        d2 = (bus_arr[:,0]-lon)**2 + (bus_arr[:,1]-lat)**2
        order = np.argsort(d2)
        chosen = None
        for idx in order[:25]:
            if vn[idx] >= 66:
                chosen = idx; break
        if chosen is None:
            chosen = order[0]
        b = bus_ids[chosen]
        a = agg[b]; a["p"] += cap; a["n"] += 1; a["fuel"][p.get("PriEnergySource")] += cap
    total_all = total_disp = 0.0
    for b, a in agg.items():
        fuel = a["fuel"].most_common(1)[0][0]; total_all += a["p"]
        if fuel in DISPATCHABLE:
            pp.create_gen(net, bus=b, p_mw=0.0, vm_pu=1.0,
                          max_p_mw=a["p"], min_p_mw=0.0, sn_mva=max(a["p"],1.0),
                          name=f"GEN {fuel} ({a['n']})", controllable=True)
            total_disp += a["p"]
        else:
            pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0, sn_mva=max(a["p"],1.0),
                           name=f"SGEN {fuel} ({a['n']})", type=fuel)
    net["_gen_capacity_mw"] = total_all
    net["_disp_capacity_mw"] = total_disp
    print(f"  {total_all:.0f} MW total ({total_disp:.0f} MW dispatchable), "
          f"{len(net.gen)} gen / {len(net.sgen)} sgen buses")


def _add_slack(net, n_interties=6):
    """Place several external grids at the strongest, geographically-spread
    500/230 kV hub buses. SoCal imports bulk power via multiple interties
    (Palo Verde, Pacific DC Intertie, etc.); several ext_grids spread the slack
    burden and make the power flow physically realistic instead of forcing all
    power through one node."""
    import pandapower.topology as top
    mg = top.create_nxgraph(net, respect_switches=False)
    deg = dict(mg.degree())
    hv = net.bus[net.bus.vn_kv >= 230]
    if len(hv) == 0:
        hv = net.bus[net.bus.vn_kv >= net.bus.vn_kv.max()-1]
    ranked = sorted(hv.index, key=lambda b: deg.get(b, 0), reverse=True)
    xy = _bus_coords(net)
    chosen = []
    for b in ranked:
        if b not in xy:
            continue
        ok = all(((xy[b][0]-xy[c][0])**2 + (xy[b][1]-xy[c][1])**2) ** 0.5 >= 0.5
                 for c in chosen)
        if ok:
            chosen.append(b)
        if len(chosen) >= n_interties:
            break
    for i, b in enumerate(chosen):
        pp.create_ext_grid(net, bus=b, vm_pu=1.02, va_degree=0.0,
                           name=f"Intertie {i+1}" + (" (ref slack)" if i == 0 else ""))
    net["_slack_bus"] = int(chosen[0])
    net["_interties"] = [int(b) for b in chosen]
    print(f"  {len(chosen)} interties/slacks at buses {chosen} "
          f"({[float(net.bus.at[b,'vn_kv']) for b in chosen]} kV)")


def _add_loads(net, cities):
    """Population-weighted disaggregation with per-bus capping + redistribution.
    Each city's population is shared (inverse-distance) among its NEAREST_K
    nearest load buses, then any bus over MAX_BUS_LOAD_MW is capped and the
    excess redistributed, keeping the regional total fixed."""
    MAX_BUS_LOAD_MW = 250.0
    NEAREST_K = 4
    bus_xy = _bus_coords(net)
    load_buses = [b for b in net.bus.index if 33 <= net.bus.at[b,"vn_kv"] <= 115]
    if not load_buses: load_buses = list(net.bus.index)
    bxy = np.array([bus_xy[b] for b in load_buses])
    city_xy = np.array([f["geometry"]["coordinates"][:2] for f in cities])
    city_pop = np.array([f["properties"]["population"] for f in cities], dtype=float)
    weights = np.zeros(len(load_buses))
    for (lon, lat), pop in zip(city_xy, city_pop):
        d2 = (bxy[:,0]-lon)**2 + (bxy[:,1]-lat)**2
        order = np.argsort(d2)[:NEAREST_K]
        if d2[order[0]] > 0.4**2:
            continue
        inv = 1.0 / (np.sqrt(d2[order]) + 1e-3)
        inv = inv / inv.sum()
        for j, w in zip(order, inv):
            weights[j] += pop * w
    if weights.sum() == 0: weights[:] = 1.0
    floor = max(weights[weights>0].mean()*0.03, 1.0) if (weights>0).any() else 1.0
    weights = np.where(weights>0, weights, floor)
    weights = weights / weights.sum()
    p = REGIONAL_PEAK_MW * weights
    for _ in range(50):
        over = p > MAX_BUS_LOAD_MW
        if not over.any():
            break
        excess = (p[over] - MAX_BUS_LOAD_MW).sum()
        p[over] = MAX_BUS_LOAD_MW
        room = ~over
        if not room.any():
            break
        headroom = MAX_BUS_LOAD_MW - p[room]
        p[room] += excess * (headroom / headroom.sum())
    q_factor = math.tan(math.acos(LOAD_POWER_FACTOR))
    n_load = 0
    for b, pmw in zip(load_buses, p):
        if pmw < 0.05: continue
        pp.create_load(net, bus=b, p_mw=float(pmw), q_mvar=float(pmw*q_factor), name=f"load@{b}")
        n_load += 1
    print(f"  {n_load} loads, total {net.load.p_mw.sum():.0f} MW "
          f"(target {REGIONAL_PEAK_MW:.0f}), max bus load {net.load.p_mw.max():.0f} MW")


if __name__ == "__main__":
    net, line_geo = build()
    out = os.path.join(OUT, "socal_grid.json")
    pp.to_json(net, out)
    print(f"\nSaved -> {out}")
    print(net)
