#!/usr/bin/env python3
"""
Toggleable synthetic medium-voltage (MV, ~12 kV) feeder layer.

The backbone model (build_network.py) carries load on sub-transmission buses
(33-115 kV).  In reality that power steps down at distribution substations onto
~12 kV feeders that fan out to customers.  This module synthesises that layer so
the model can be studied with realistic distribution-level buses, losses and
voltage drop -- without needing real (proprietary, mostly unpublished) feeder GIS.

DESIGN
------
For every backbone bus that currently carries load (the "source" bus, typically
33-115 kV) we:
  1) create a 12 kV MV head bus co-located with the source bus,
  2) connect them with a distribution transformer (source kV / 12 kV),
  3) fan out N_FEEDERS synthetic 12 kV feeder buses at small geographic offsets
     around the source (radial cables, FEEDER_KM long),
  4) move the source bus's load DOWN onto the feeder buses, split evenly.

Everything created here is TAGGED so the layer is fully reversible:
  net.bus['mv_layer']  == True   for MV head + feeder buses
  net.line['mv_layer'] == True   for MV feeder cables
  net.trafo['mv_layer']== True   for distribution transformers
  net.load['mv_layer'] == True   for loads relocated onto feeders
The original backbone loads are deactivated (in_service=False) and tagged
'_backbone_load_of' so remove_mv_layer() can restore them exactly.

USAGE
-----
    from mv_feeders import add_mv_layer, remove_mv_layer
    add_mv_layer(net)        # build & enable the feeder layer
    remove_mv_layer(net)     # drop it and restore backbone loads

Geo-referencing is preserved: every MV bus gets a GeoJSON Point in net.bus.geo
and every feeder cable a GeoJSON LineString in net.line.geo.
"""
import json, math
import numpy as np
import pandapower as pp

# ---- tunables ---------------------------------------------------------------
MV_KV          = 12.0     # nominal MV voltage
N_FEEDERS      = 3        # synthetic feeders per distribution substation
FEEDER_KM      = 4.0      # representative feeder length (km)
FEEDER_R       = 0.30     # ohm/km  (typical 12 kV overhead/cable equivalent)
FEEDER_X       = 0.35     # ohm/km
FEEDER_C       = 10.0     # nF/km
FEEDER_IMAX    = 0.40     # kA  (~7 MVA at 12 kV)
DIST_TRAFO_VK  = 6.0      # distribution transformer short-circuit voltage %
DIST_TRAFO_VKR = 0.8      # resistive part %
OFFSET_M       = 1500.0   # geographic spread of feeder ends around the substation
M_PER_DEG_LAT  = 111000.0
M_PER_DEG_LON  = 88000.0


def _xy(net, b):
    g = net.bus.at[b, "geo"]
    if isinstance(g, str) and g:
        return tuple(json.loads(g)["coordinates"])
    return None


def _ensure_cols(net):
    for tbl in ("bus", "line", "trafo", "load"):
        if "mv_layer" not in net[tbl].columns:
            net[tbl]["mv_layer"] = False
    if "_backbone_load_of" not in net.load.columns:
        net.load["_backbone_load_of"] = -1


def add_mv_layer(net, n_feeders=N_FEEDERS, feeder_km=FEEDER_KM):
    """Build the synthetic 12 kV feeder layer and relocate load onto it.

    Returns a dict of summary counts.
    """
    if net.get("_mv_layer_present"):
        print("  MV layer already present; skipping.")
        return {}
    _ensure_cols(net)

    # source buses = buses that currently carry an active backbone load
    src_loads = net.load[(net.load["in_service"]) &
                         (net.load["mv_layer"] == False)]
    sources = sorted(set(int(b) for b in src_loads["bus"]))

    n_bus0, n_line0, n_trafo0, n_load0 = (len(net.bus), len(net.line),
                                          len(net.trafo), len(net.load))

    for sb in sources:
        sxy = _xy(net, sb)
        if sxy is None:
            continue
        slon, slat = sxy
        svn = float(net.bus.at[sb, "vn_kv"])

        # 1) MV head bus co-located with the source
        head = pp.create_bus(net, vn_kv=MV_KV, name=f"MV_head@{sb}",
                             geodata=(slon, slat), type="b")
        net.bus.at[head, "mv_layer"] = True

        # 2) distribution transformer source -> 12 kV
        sn = max(net.load[net.load["bus"] == sb]["p_mw"].sum() * 1.3, 5.0)
        ti = pp.create_transformer_from_parameters(
            net, hv_bus=sb, lv_bus=head, sn_mva=float(sn),
            vn_hv_kv=svn, vn_lv_kv=MV_KV,
            vkr_percent=DIST_TRAFO_VKR, vk_percent=DIST_TRAFO_VK,
            pfe_kw=0.0, i0_percent=0.05, name=f"dist_trafo@{sb}")
        net.trafo.at[ti, "mv_layer"] = True

        # 3) fan out N feeders at small geographic offsets
        feeder_buses = []
        for k in range(n_feeders):
            ang = 2 * math.pi * k / n_feeders
            dlon = (OFFSET_M * math.cos(ang)) / M_PER_DEG_LON
            dlat = (OFFSET_M * math.sin(ang)) / M_PER_DEG_LAT
            flon, flat = slon + dlon, slat + dlat
            fb = pp.create_bus(net, vn_kv=MV_KV, name=f"MV_feeder{sb}_{k}",
                              geodata=(flon, flat), type="b")
            net.bus.at[fb, "mv_layer"] = True
            li = pp.create_line_from_parameters(
                net, from_bus=head, to_bus=fb, length_km=feeder_km,
                r_ohm_per_km=FEEDER_R, x_ohm_per_km=FEEDER_X,
                c_nf_per_km=FEEDER_C, max_i_ka=FEEDER_IMAX,
                name=f"feeder{sb}_{k}", type="cs")
            net.line.at[li, "mv_layer"] = True
            net.line.at[li, "geo"] = json.dumps(
                {"type": "LineString", "coordinates": [[slon, slat], [flon, flat]]})
            feeder_buses.append(fb)

        # 4) move the source bus load down onto the feeders (split evenly),
        #    deactivate the backbone load(s) but keep them for reversibility.
        sl = net.load[(net.load["bus"] == sb) & (net.load["in_service"]) &
                      (net.load["mv_layer"] == False)]
        for li_idx in sl.index:
            p = float(net.load.at[li_idx, "p_mw"])
            q = float(net.load.at[li_idx, "q_mvar"])
            net.load.at[li_idx, "in_service"] = False  # disable backbone load
            per_p, per_q = p / len(feeder_buses), q / len(feeder_buses)
            for fb in feeder_buses:
                nl = pp.create_load(net, bus=fb, p_mw=per_p, q_mvar=per_q,
                                    name=f"mvload@{fb}")
                net.load.at[nl, "mv_layer"] = True
                net.load.at[nl, "_backbone_load_of"] = int(li_idx)

    net["_mv_layer_present"] = True
    summary = {
        "source_substations": len(sources),
        "mv_buses_added": len(net.bus) - n_bus0,
        "feeders_added": len(net.line) - n_line0,
        "dist_trafos_added": len(net.trafo) - n_trafo0,
        "mv_loads_added": len(net.load) - n_load0,
    }
    print(f"  MV layer: {summary['source_substations']} substations -> "
          f"+{summary['mv_buses_added']} buses, +{summary['feeders_added']} feeders, "
          f"+{summary['dist_trafos_added']} dist-trafos, "
          f"+{summary['mv_loads_added']} relocated loads")
    return summary


def remove_mv_layer(net):
    """Drop everything tagged mv_layer and re-activate the backbone loads."""
    if not net.get("_mv_layer_present"):
        print("  no MV layer present.")
        return
    # re-activate original backbone loads
    restore = set(int(x) for x in net.load["_backbone_load_of"] if x is not None and x >= 0)
    for li_idx in restore:
        if li_idx in net.load.index:
            net.load.at[li_idx, "in_service"] = True
    # drop MV loads
    mv_loads = net.load.index[net.load["mv_layer"] == True]
    net.load.drop(mv_loads, inplace=True)
    # drop MV feeders & dist trafos
    mv_lines = net.line.index[net.line["mv_layer"] == True]
    pp.drop_lines(net, mv_lines)
    mv_trafos = net.trafo.index[net.trafo["mv_layer"] == True]
    net.trafo.drop(mv_trafos, inplace=True)
    # drop MV buses
    mv_buses = net.bus.index[net.bus["mv_layer"] == True]
    pp.drop_buses(net, mv_buses)
    net["_mv_layer_present"] = False
    print(f"  MV layer removed; backbone loads restored ({len(restore)}).")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dispatch_and_run import run, report
    here = os.path.dirname(__file__)
    net = pp.from_json(os.path.join(here, "socal_grid.json"))
    print("Adding MV feeder layer...")
    add_mv_layer(net)
    print("Running power flow WITH MV layer...")
    if run(net):
        report(net)
        out = os.path.join(here, "socal_grid_mv_solved.json")
        pp.to_json(net, out)
        print(f"Saved -> {out}")
