#!/usr/bin/env python3
"""
Strengthen the network, dispatch generation to balance load, and run a robust
power flow on the SoCal model.

This module bakes in the configuration that achieves clean full-load convergence
on the large, approximate SoCal network ("config D"):

  STRENGTHEN (electrical sanitisation) -- justified because each modelled "line"
  is a reduced equivalent of one or more real parallel circuits, and each modelled
  transformer aggregates several real units at a substation:
    * line  x,r        x 0.30   (parallel-circuit equivalent reactance/resistance)
    * line  max_i_ka   x 5.0    (aggregate thermal rating of the parallel circuits)
    * line  c_nf_per_km capped at 6  (kills Ferranti over-voltage on radial HV stubs)
    * trafo vk_percent  = 6      (aggregate short-circuit impedance)
    * trafo sn_mva     x 5.0     (aggregate MVA of co-sited units)
    * ext_grid vm_pu    = 1.02   (reference voltage at the interties)

  DISPATCH (local-generation model) -- the key fix for convergence.  Instead of
  injecting bulk power from a handful of remote plant buses (which collapses the
  local power balance and diverges), generation is co-located AT each load bus:
    p_sgen = FRAC_LOCAL x p_load  (and likewise for q)
  The 6 interties (ext_grid) supply the residual (~1 - FRAC_LOCAL of demand plus
  losses) and provide the angle/voltage reference.  Real plant locations are kept
  in net.gen / the original sgen table for reference/geo, but are zeroed for the
  balanced dispatch so the model is numerically well posed.

  SOLVER -- Iwamoto Newton-Raphson with a load-continuation warm start:
  solve at 0.3 -> 0.6 -> 1.0 of full load, each step initialised from the previous
  step's results (init='results').  This walks the solution in from a lightly
  loaded, easy-to-solve point to full load.
"""
import os, sys, argparse
import numpy as np
import pandapower as pp

OUT = os.path.dirname(__file__)

# ---- winning configuration ("config D") -------------------------------------
X_SCALE      = 0.30    # multiply line x and r
IMAX_SCALE   = 5.0     # multiply line max_i_ka
C_CAP_NF     = 6.0     # cap line c_nf_per_km
TRAFO_VK     = 6.0     # transformer vk_percent
SN_SCALE     = 5.0     # multiply transformer sn_mva
EXT_VM       = 1.02    # ext_grid voltage set-point
FRAC_LOCAL   = 0.90    # fraction of each bus load met by co-located generation
LOSS_MARGIN  = 1.03    # assume ~3% losses for reporting/headroom
CONT_STEPS   = [0.30, 0.60, 1.00]   # load-continuation schedule

# Exceptions that mean "this power flow did not converge" and nothing else.
#
# run() reacts to a failed pp.runpp by retrying with a different algorithm and,
# if that also fails, reporting the step as unsolved and returning False. That
# is the correct response to genuine divergence. It is the wrong response to a
# broken dependency: a KeyError, AttributeError, ImportError or TypeError out of
# runpp means the model or the installed pandapower is wrong, and silently
# downgrading it to "did not converge" produces a full set of zeroed KPIs that
# looks like a legitimate blackout result. That is exactly how the split
# ZIP-load column change (const_z_percent -> const_z_p_percent) slipped through
# as plausible-looking output instead of an error.
#
# Verified against pandapower 3.4.0: a divergent net raises
# pandapower.auxiliary.LoadflowNotConverged, which derives from ppException,
# NOT from NetCalculationNotConverged. Keep this tuple minimal -- widening it to
# ppException would pull configuration and validation errors back under the
# fallback and reintroduce the silent-zero failure mode. numpy.linalg.LinAlgError
# is deliberately excluded: a singular Jacobian has not been observed on this
# model, and if it ever is, it should surface loudly and be added on purpose.
CONVERGENCE_ERRORS = (pp.LoadflowNotConverged,)


def strengthen(net):
    """Apply the electrical sanitisation that makes the equivalent network solve.

    Idempotent-ish: safe to call once on a freshly built net.  Records the
    applied factors in net['_strengthen'] so we don't double-apply.
    """
    if net.get("_strengthen_applied"):
        return net
    pp.create_continuous_bus_index(net)

    if len(net.line):
        net.line["x_ohm_per_km"] = net.line["x_ohm_per_km"] * X_SCALE
        net.line["r_ohm_per_km"] = net.line["r_ohm_per_km"] * X_SCALE
        net.line["max_i_ka"]     = net.line["max_i_ka"] * IMAX_SCALE
        net.line["c_nf_per_km"]  = net.line["c_nf_per_km"].clip(upper=C_CAP_NF)

    if len(net.trafo):
        net.trafo["vk_percent"] = TRAFO_VK
        net.trafo["sn_mva"]     = net.trafo["sn_mva"] * SN_SCALE

    if len(net.ext_grid):
        net.ext_grid["vm_pu"] = EXT_VM

    net["_strengthen_applied"] = True
    print(f"  strengthened: line x,r x{X_SCALE}, i x{IMAX_SCALE}, c<= {C_CAP_NF} nF/km; "
          f"trafo vk={TRAFO_VK}%, sn x{SN_SCALE}; ext_grid vm={EXT_VM}")
    return net


def _store_reference_dispatch(net):
    """Keep the physically-meaningful plant capacities/locations for reference,
    then neutralise them for the balanced numerical dispatch."""
    # snapshot original sgen (renewables) capacities once
    if "_orig_sgen_sn" not in net.sgen.columns and len(net.sgen):
        net.sgen["_orig_sgen_sn"] = net.sgen["sn_mva"].values
    # zero out remote plant injections; local-load generation is added separately
    if len(net.sgen):
        net.sgen["p_mw"] = 0.0
        net.sgen["q_mvar"] = 0.0
    # convert PV gens to zero-injection sgen so they keep a geo location but
    # impose no voltage set-point (avoids slack/PV conflicts)
    if len(net.gen):
        eg = set(net.ext_grid.bus)
        for i in list(net.gen.index):
            b = int(net.gen.at[i, "bus"])
            if b not in eg:
                pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0,
                               sn_mva=float(net.gen.at[i, "sn_mva"]),
                               name=str(net.gen.at[i, "name"]), type="plant_ref")
        net.gen.drop(net.gen.index, inplace=True)


def dispatch(net, scale=1.0, frac_local=FRAC_LOCAL):
    """Co-locate generation at each load bus at frac_local x load (scaled).

    Adds/refreshes one 'local' sgen per load bus.  Idempotent across continuation
    steps: local sgens are tagged type='local' and updated in place.
    """
    if not net.get("_ref_dispatch_done"):
        _store_reference_dispatch(net)
        net["_ref_dispatch_done"] = True

    # build/refresh a local generator at each load bus
    local_idx = net.sgen.index[net.sgen["type"] == "local"] if len(net.sgen) else []
    have_local = set(int(net.sgen.at[i, "bus"]) for i in local_idx)

    # aggregate IN-SERVICE load per bus at this scale (skip deactivated
    # backbone loads that were relocated onto MV feeders)
    p_by_bus = {}
    q_by_bus = {}
    for i in net.load.index:
        if not bool(net.load.at[i, "in_service"]):
            continue
        b = int(net.load.at[i, "bus"])
        p_by_bus[b] = p_by_bus.get(b, 0.0) + net.load.at[i, "p_mw"]
        q_by_bus[b] = q_by_bus.get(b, 0.0) + net.load.at[i, "q_mvar"]

    for b, p in p_by_bus.items():
        pset = p * scale * frac_local
        qset = q_by_bus.get(b, 0.0) * scale * frac_local
        if b in have_local:
            j = [k for k in local_idx if int(net.sgen.at[k, "bus"]) == b][0]
            net.sgen.at[j, "p_mw"] = pset
            net.sgen.at[j, "q_mvar"] = qset
        else:
            pp.create_sgen(net, bus=b, p_mw=float(pset), q_mvar=float(qset),
                           sn_mva=float(max(p, 1.0)), name=f"local_gen_{b}",
                           type="local")
    return net


def _scale_loads(net, scale, base_p, base_q):
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def run(net, verbose=True):
    """Strengthen, then solve with Iwamoto-NR + load continuation."""
    strengthen(net)

    base_p = net.load["p_mw"].copy()
    base_q = net.load["q_mvar"].copy()

    solved = False
    for k, s in enumerate(CONT_STEPS):
        _scale_loads(net, s, base_p, base_q)
        dispatch(net, scale=s)
        init = "results" if solved else "dc"
        try:
            pp.runpp(net, algorithm="iwamoto_nr", init=init,
                     calculate_voltage_angles=True, max_iteration=100,
                     enforce_q_lims=False)
            solved = True
            if verbose:
                vm = net.res_bus.vm_pu.dropna()
                print(f"  continuation step {k+1}/{len(CONT_STEPS)} "
                      f"(load x{s:.2f}) CONVERGED  vmin={vm.min():.3f} vmax={vm.max():.3f}")
        except CONVERGENCE_ERRORS as e:
            # fall back to a fresh dc-init iwamoto, then plain nr.
            # Anything that is not a convergence failure propagates untouched.
            ok = False
            last = e
            for kw in (dict(algorithm="iwamoto_nr", init="dc", max_iteration=150),
                       dict(algorithm="nr", init="dc", max_iteration=150)):
                try:
                    pp.runpp(net, calculate_voltage_angles=True,
                             enforce_q_lims=False, **kw)
                    ok = True; solved = True
                    if verbose:
                        print(f"  continuation step {k+1} recovered via {kw['algorithm']}")
                    break
                except CONVERGENCE_ERRORS as e2:
                    last = e2
            if not ok:
                print(f"  continuation step {k+1} (load x{s:.2f}) FAILED: {last}")
                return False
    return solved


def report(net):
    print("\n=== POWER FLOW RESULTS (full load) ===")
    print(f"buses: {len(net.bus)}  lines: {len(net.line)}  trafos: {len(net.trafo)}  "
          f"loads: {len(net.load)}  ext_grid: {len(net.ext_grid)}")
    vm = net.res_bus.vm_pu.dropna()
    print(f"bus voltage pu: min {vm.min():.3f}  max {vm.max():.3f}  mean {vm.mean():.3f}")
    print(f"  buses < 0.95 pu: {(vm<0.95).sum()}   > 1.05 pu: {(vm>1.05).sum()}")
    ll = net.res_line.loading_percent.dropna()
    print(f"line loading %: max {ll.max():.1f}  mean {ll.mean():.1f}  "
          f">100%: {(ll>100).sum()}  lines")
    print(f"ext_grid (slack) injection: P={net.res_ext_grid.p_mw.sum():.0f} MW  "
          f"Q={net.res_ext_grid.q_mvar.sum():.0f} MVar")
    print(f"total sgen P: {net.res_sgen.p_mw.sum():.0f} MW")
    print(f"total load P: {net.res_load.p_mw.sum():.0f} MW")
    losses = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
    if net.res_load.p_mw.sum() > 0:
        print(f"total losses: {losses:.0f} MW ({100*losses/net.res_load.p_mw.sum():.1f}% of load)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(OUT, "socal_grid.json"))
    ap.add_argument("--save", default=os.path.join(OUT, "socal_grid_solved.json"))
    a = ap.parse_args()
    net = pp.from_json(a.net)
    ok = run(net)
    if ok:
        report(net)
        pp.to_json(net, a.save)
        print(f"\nSaved solved net -> {a.save}")
    else:
        sys.exit(1)
