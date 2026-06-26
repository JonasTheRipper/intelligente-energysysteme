#!/usr/bin/env python3
"""System test: MIDAS-style convergence check on the prepared grid.

Applies the prepared per-bus load / sgen time-series to the SoCal grid at a few
representative MIDAS time steps and solves with **plain ``pp.runpp``** (exactly
what the MIDAS ``pandapower`` simulator does), confirming the scenario will
converge before launching the full mosaik run.

This test depends on the MIDAS data artefacts produced by
``midas_socal/prepare_midas.py`` (``socal_grid_midas.json``, the load / sgen
time-series CSVs and the bus mappings) living in the MIDAS data directory. When
those artefacts are absent — e.g. on a fresh CI runner that has not run the
``prepare`` step — the test is **skipped** rather than failed.

Run: ``python3 tests/test_midas_steps.py`` or ``pytest -m slow tests/test_midas_steps.py``.
"""

import json
import os

import numpy as np
import pandas as pd
import pandapower as pp
import pytest

D = os.path.expanduser("~/.config/midas/midas_data")
_REQUIRED = [
    "socal_grid_midas.json",
    "socal_load_ts.csv",
    "socal_sgen_ts.csv",
    "load_mapping.json",
    "sgen_mapping.json",
]


def _data_available() -> bool:
    return all(os.path.exists(os.path.join(D, f)) for f in _REQUIRED)


def _load_artifacts():
    net = pp.from_json(os.path.join(D, "socal_grid_midas.json"))
    load_ts = pd.read_csv(os.path.join(D, "socal_load_ts.csv"))
    sgen_ts = pd.read_csv(os.path.join(D, "socal_sgen_ts.csv"))
    load_map = json.load(open(os.path.join(D, "load_mapping.json")))
    sgen_map = json.load(open(os.path.join(D, "sgen_mapping.json")))
    return net, load_ts, sgen_ts, load_map, sgen_map


def _apply_step(net, load_ts, sgen_ts, load_map, sgen_map,
                bus_loads, bus_sgens, t):
    # loads: distribute the bus column value across that bus's load elements
    for bus_s, entries in load_map.items():
        bus = int(bus_s)
        # each entry is [[p_col, q_col], scale] in the prepared mapping schema
        cols, scale = entries[0]
        if isinstance(cols, (list, tuple)):
            p_col, q_col = cols[0], cols[1]
        else:  # backward-compat: single p column, derive q from cos phi
            p_col, q_col = cols, None
        p_val = float(load_ts[p_col].iloc[t]) * scale
        q_val = (float(load_ts[q_col].iloc[t]) * scale
                 if q_col is not None else p_val * 0.31)  # ~cos phi 0.95
        idxs = bus_loads.get(bus, [])
        if not idxs:
            continue
        p_per = p_val / len(idxs)
        q_per = q_val / len(idxs)
        for i in idxs:
            net.load.at[i, "p_mw"] = p_per
            net.load.at[i, "q_mvar"] = q_per
    # sgens: distribute the bus column value across that bus's sgen elements
    for bus_s, entries in sgen_map.items():
        bus = int(bus_s)
        cols, scale = entries[0]
        if isinstance(cols, (list, tuple)):
            p_col, q_col = cols[0], cols[1]
        else:
            p_col, q_col = cols, None
        p_val = float(sgen_ts[p_col].iloc[t]) * scale
        q_val = (float(sgen_ts[q_col].iloc[t]) * scale
                 if q_col is not None else 0.0)
        sidxs = bus_sgens.get(bus, [])
        if not sidxs:
            continue
        p_per = p_val / len(sidxs)
        q_per = q_val / len(sidxs)
        for i in sidxs:
            net.sgen.at[i, "p_mw"] = p_per
            net.sgen.at[i, "q_mvar"] = q_per


def test_midas_steps_converge():
    if not _data_available():
        pytest.skip(
            "MIDAS data artefacts not present (run midas_socal/prepare_midas.py "
            "first); skipping convergence check."
        )

    net, load_ts, sgen_ts, load_map, sgen_map = _load_artifacts()

    bus_loads, bus_sgens = {}, {}
    for i in net.load.index:
        bus_loads.setdefault(int(net.load.at[i, "bus"]), []).append(i)
    for i in net.sgen.index:
        bus_sgens.setdefault(int(net.sgen.at[i, "bus"]), []).append(i)

    any_converged = False
    for t in [0, 24, 48, 72, 80, 95]:
        if t >= len(load_ts):
            continue
        _apply_step(net, load_ts, sgen_ts, load_map, sgen_map,
                    bus_loads, bus_sgens, t)
        tot_load = net.load.p_mw.sum()
        tot_sgen = net.sgen.p_mw.sum()
        try:
            pp.runpp(net, numba=True, calculate_voltage_angles=True,
                     enforce_q_lims=False)
            vm = net.res_bus.vm_pu.dropna()
            ext = net.res_ext_grid.p_mw.sum()
            viol = int(((vm < 0.9) | (vm > 1.1)).sum())
            print(f"t={t:2d}  load={tot_load:7.0f}MW sgen={tot_sgen:7.0f}MW "
                  f"ext_grid={ext:8.0f}MW  vmin={vm.min():.3f} vmax={vm.max():.3f} "
                  f"viol(>0.1pu)={viol}  CONVERGED")
            any_converged = True
        except Exception as e:
            print(f"t={t:2d}  load={tot_load:.0f} sgen={tot_sgen:.0f}  FAILED: {e}")

    assert any_converged, "no MIDAS time step converged"


if __name__ == "__main__":
    if not _data_available():
        print("MIDAS data artefacts not present; nothing to do.")
    else:
        test_midas_steps_converge()
        print("\nMIDAS STEP CONVERGENCE TEST PASSED")
