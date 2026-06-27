"""Power-flow-invariant global voltage rescale for the SoCal pandapower net.

WHY: ``midas_powergrid`` hard-caps the bus ``vn_kv`` sensor Box at 440 kV
(``midas_powergrid/meta.py``: ``("vn_kv", "float", 0, 440)``). The SoCal grid
has 50 buses (and the matching trafo HV windings) at 500 kV, so the real MIDAS
descriptor raises ``Value "500.0" not contained in space Box(0,440)`` before the
world can step. Patching site-packages is forbidden, so instead we relabel every
voltage *base* by a single global factor ``f`` (default 0.8 -> 500 kV becomes
400 kV) and rescale the impedances so the per-unit power flow is **identical**.

Invariance (per-unit system, ``z_pu = z_ohm * S_base / V_base**2``):
    bus.vn_kv        *= f
    trafo.vn_hv_kv   *= f      (not a sensor, but needed for a correct tap ratio)
    trafo.vn_lv_kv   *= f
    line.r/x_ohm_per_km *= f**2   -> z_pu unchanged
    line.c_nf/g_us      /= f**2   -> b_pu/g_pu unchanged
    line.max_i_ka       /= f      -> loading_percent unchanged (I ~ 1/V at fixed P)
    loads / sgen p_mw,q_mvar, vk_percent, sn_mva, ext_grid/gen vm_pu : UNCHANGED

So vm_pu, served MW, loading %, and in_service all stay bit-for-bit equal; only
the absolute kV labels shrink under the 440 cap.

Run:  python midas_socal/rescale_grid_voltage.py
"""

from __future__ import annotations

import os

import numpy as np
import pandapower as pp

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "socal_grid_midas.json")
DST = os.path.join(_HERE, "socal_grid_midas_rescaled.json")
VN_CAP = 440.0


def rescale(net, f: float) -> None:
    net.bus["vn_kv"] = net.bus["vn_kv"] * f
    if len(net.trafo):
        net.trafo["vn_hv_kv"] = net.trafo["vn_hv_kv"] * f
        net.trafo["vn_lv_kv"] = net.trafo["vn_lv_kv"] * f
    if len(net.line):
        net.line["r_ohm_per_km"] = net.line["r_ohm_per_km"] * f * f
        net.line["x_ohm_per_km"] = net.line["x_ohm_per_km"] * f * f
        net.line["c_nf_per_km"] = net.line["c_nf_per_km"] / (f * f)
        if "g_us_per_km" in net.line:
            net.line["g_us_per_km"] = net.line["g_us_per_km"] / (f * f)
        net.line["max_i_ka"] = net.line["max_i_ka"] / f
    if len(net.trafo3w):
        for col in ("vn_hv_kv", "vn_mv_kv", "vn_lv_kv"):
            if col in net.trafo3w:
                net.trafo3w[col] = net.trafo3w[col] * f


def main() -> None:
    base = pp.from_json(SRC)
    pp.runpp(base)
    vmax = float(base.bus.vn_kv.max())
    # pick f so the largest base drops just under the cap (with a margin)
    f = (VN_CAP - 1.0) / vmax
    f = min(f, 0.8)
    print(f"max vn_kv={vmax} -> factor f={f:.6f} (new max {vmax * f:.2f} kV)")

    net = pp.from_json(SRC)
    rescale(net, f)
    assert float(net.bus.vn_kv.max()) < VN_CAP, "rescale did not clear the cap"
    pp.runpp(net)

    dvm = float(np.max(np.abs(base.res_bus.vm_pu.values - net.res_bus.vm_pu.values)))
    dserved = abs(base.res_load.p_mw.sum() - net.res_load.p_mw.sum())
    dload = float(np.max(np.abs(
        base.res_line.loading_percent.values - net.res_line.loading_percent.values)))
    print(f"max |dvm_pu| = {dvm:.3e}")
    print(f"|d served MW| = {dserved:.3e}")
    print(f"max |d line loading%| = {dload:.3e}")
    assert dvm < 1e-6, "vm_pu changed -- rescale not power-flow invariant"
    assert dserved < 1e-3, "served MW changed"
    assert dload < 1e-3, "line loading changed"

    pp.to_json(net, DST)
    print(f"wrote {DST}  (factor f={f:.6f})")


if __name__ == "__main__":
    main()
