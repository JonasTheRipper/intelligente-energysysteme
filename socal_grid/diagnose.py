#!/usr/bin/env python3
"""Diagnose convergence: fix numerical issues, then find the load level that solves
and inspect what limits it."""
import os, numpy as np, pandapower as pp
import pandapower.topology as top
import networkx as nx
from dispatch_and_run import dispatch

OUT = os.path.dirname(__file__)

def sanitize(net):
    """Fix numerical pathologies that cause singular Jacobians."""
    # 1) enforce a minimum series impedance per line (avoid near-zero -> singular)
    min_len = 0.2  # km
    net.line.loc[net.line.length_km < min_len, "length_km"] = min_len
    # absolute floor on total reactance
    xtot = net.line.x_ohm_per_km * net.line.length_km
    bad = xtot < 0.05
    if bad.any():
        net.line.loc[bad, "x_ohm_per_km"] = 0.05 / net.line.loc[bad, "length_km"]
    rtot = net.line.r_ohm_per_km * net.line.length_km
    badr = rtot < 0.01
    if badr.any():
        net.line.loc[badr, "r_ohm_per_km"] = 0.01 / net.line.loc[badr, "length_km"]
    # 2) drop in-service isolated buses not connected to any ext_grid island
    return net

def connectivity(net):
    mg = top.create_nxgraph(net, respect_switches=False)
    comps = list(nx.connected_components(mg))
    eg = set(net.ext_grid.bus)
    energized = set()
    for c in comps:
        if c & eg:
            energized |= c
    dead = [b for b in net.bus.index if b not in energized]
    return comps, dead

def main():
    net = pp.from_json(os.path.join(OUT, "socal_grid.json"))
    sanitize(net)
    comps, dead = connectivity(net)
    print(f"components: {len(comps)}; dead (no ext_grid in island): {len(dead)} buses")
    # set generation
    dispatch(net)
    # scale loads to find feasible ceiling
    base_p = net.load.p_mw.copy()
    base_q = net.load.q_mvar.copy()
    base_sgen = net.sgen.p_mw.copy()
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]:
        net.load.p_mw = base_p * scale
        net.load.q_mvar = base_q * scale
        net.sgen.p_mw = base_sgen * scale  # scale gen with load to keep balance
        try:
            pp.runpp(net, algorithm="nr", init="dc", calculate_voltage_angles=True,
                     max_iteration=60, numba=False)
            vm = net.res_bus.vm_pu.dropna()
            ll = net.res_line.loading_percent.dropna()
            print(f"  scale {scale:.2f}: CONVERGED  vmin={vm.min():.3f} vmax={vm.max():.3f} "
                  f"lines>100%={int((ll>100).sum())} maxload={ll.max():.0f}%")
        except Exception as e:
            print(f"  scale {scale:.2f}: FAILED ({type(e).__name__})")

if __name__ == "__main__":
    main()
