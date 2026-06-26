#!/usr/bin/env python3
"""
Prepare the Southern California pandapower model for a MIDAS co-simulation
driven by REAL CAISO actuals.

This script bakes the *verified-converging* dispatch recipe into the generated
time series so that MIDAS -- which calls a bare ``pp.runpp(net, numba=...)``
with a flat start every step -- converges on all 96 steps of the day.

Outputs (written into the MIDAS data path so the powerseries module can find
them, with reference copies next to this script):
  1. socal_grid_midas.json   - the strengthened grid. PV ``gen`` rows are
                               converted to zero-injection sgens (so no gen
                               shares a bus with an ext_grid -> no voltage
                               control conflict). Pre-existing load/sgen
                               elements are what MIDAS drives each step.
  2. socal_load_ts.csv        - per load-bus active (p_*) AND reactive (q_*)
                               columns, the CAISO demand *shape* scaled to the
                               bus peak. 96 rows (15-min steps over 24 h).
  3. socal_sgen_ts.csv        - per renewable sgen p_*/q_* columns (CAISO
                               solar/wind/battery shapes, capped per bus) plus
                               per load-bus LOCAL generator p_*/q_* columns
                               (local active follows residual demand; local
                               reactive supplies the bus's reactive demand).
  4. load_mapping.json        - {bus: [[[p_col, q_col], 1.0], ...]}  (combined)
  5. sgen_mapping.json        - {bus: [[[p_col, q_col], 1.0], ...]}  (combined)

WHY THIS CONVERGES (verified, see project notes):
  * Loads: p = demand_shape[t] * bus_peak;  q = p * 0.31  (cos phi ~ 0.95).
  * Renewables: p = min(shape[t] * nameplate, CAP) where
      CAP = 2.0 * bus_peak_load + 5 MW  -- a per-bus hosting cap that prevents
      radial degree-1 renewable stubs from causing a voltage-rise divergence.
      q = 0.
  * Local generators (one per load bus): p = max(0.95*bus_load - renew@bus, 0);
      q = 0.85 * bus_load_q  -- LOCAL REACTIVE SUPPORT, the critical ingredient
      that keeps vmin ~ 0.96-0.99 instead of collapsing to ~0.73.
  The interties (6 ext_grid) carry only losses + the small residual.

The CAISO actuals provide the real day *dynamics*; the pandapower model
provides the geographic/topological structure and the per-bus *magnitudes*.
"""
import os, sys, json, collections
import numpy as np
import pandas as pd
import pandapower as pp

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "socal_grid")
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.join(HERE, "weather"))
from dispatch_and_run import strengthen  # noqa: E402
from noaa_provider import NOAAWeatherConfig, build_noaa_weather  # noqa: E402

# MIDAS data path (where powerseries looks for filenames)
MIDAS_DATA = os.path.expanduser("~/.config/midas/midas_data")

N_STEPS = 96               # 15-min steps over 24 h
STEP_SECONDS = 15 * 60

# capacity factors used to turn nameplate into a realistic peak contribution
CF = {"SUN": 1.0, "WND": 1.0, "BAT": 0.5}  # solar/wind already shaped by CAISO

# ----- WINNING DISPATCH CONSTANTS (verified to converge all 96 steps) -----
LOAD_PF_TAN = 0.31         # q/p for loads (cos phi ~ 0.95)
RENEW_CAP_MULT = 2.0       # per-bus renewable hosting cap = MULT*bus_peak + ADD
RENEW_CAP_ADD = 5.0        # MW
LOCAL_LOAD_FRAC = 0.95     # local active gen serves 95% of bus residual demand
LOCAL_Q_FRAC = 0.85        # local reactive gen supplies 85% of bus reactive load


def load_caiso(date):
    f = os.path.join(DATA, f"caiso_{date}.csv")
    df = pd.read_csv(f, parse_dates=["time"])
    # resample the 5-min CAISO actuals to 96 x 15-min steps
    df = df.set_index("time").resample(f"{STEP_SECONDS}s").mean().interpolate()
    df = df.iloc[:N_STEPS].reset_index()
    demand = df["demand_mw"].to_numpy()
    solar = df["solar_mw"].clip(lower=0).to_numpy()
    wind = df["wind_mw"].clip(lower=0).to_numpy()
    demand_shape = demand / demand.max()
    solar_shape = solar / max(solar.max(), 1e-6)
    wind_shape = wind / max(wind.max(), 1e-6)
    return df, demand_shape, solar_shape, wind_shape


def build_weather(date, days, weather_source, station):
    """Generate the NOAA weather CSV that replaces DWD Bremen.

    Writes ``socal_noaa_weather.csv`` into both the MIDAS data dir and the
    local scenario dir, in the schema the MIDAS weather simulator expects.
    """
    from datetime import datetime, timezone
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cfg = NOAAWeatherConfig(
        source=weather_source, station=station, start=start, days=days,
        allow_fallback=True,
    )
    fname = "socal_noaa_weather.csv"
    out_data = os.path.join(MIDAS_DATA, fname)
    build_noaa_weather(cfg, out_data)
    # mirror to the scenario dir for provenance / offline use
    try:
        import shutil
        shutil.copy(out_data, os.path.join(HERE, fname))
    except Exception:
        pass
    print(f"  NOAA weather ({weather_source}) -> {out_data}")
    return fname


def main(date="2024-07-16", days=1, weather_source="isd", station="72295023174"):
    os.makedirs(MIDAS_DATA, exist_ok=True)
    print(f"Preparing MIDAS scenario from CAISO {date} ...")

    # --- NOAA weather (replaces DWD Bremen, generated INSIDE the scenario) ---
    build_weather(date, days, weather_source, station)

    df, demand_shape, solar_shape, wind_shape = load_caiso(date)
    print(f"  CAISO shapes: demand min {demand_shape.min():.2f}, "
          f"solar midday {solar_shape.max():.2f}, wind mean {wind_shape.mean():.2f}")

    net = pp.from_json(os.path.join(MODEL, "socal_grid.json"))
    strengthen(net)

    # --- resolve voltage-control conflicts: convert PV gens to zero-injection
    # sgens (keep geo location) and drop the gen table so no gen shares a bus
    # with an ext_grid. ---
    if len(net.gen):
        for i in list(net.gen.index):
            b = int(net.gen.at[i, "bus"])
            pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0,
                           sn_mva=float(net.gen.at[i, "sn_mva"]),
                           name=str(net.gen.at[i, "name"]), type="plant_ref")
        net.gen.drop(net.gen.index, inplace=True)
        print(f"  converted PV gens -> zero-injection sgens, dropped gen table")

    # ----- per-bus peak active & reactive load (population-weighted full load)
    bus_peak = collections.defaultdict(float)
    bus_q_peak = collections.defaultdict(float)
    for i in net.load.index:
        b = int(net.load.at[i, "bus"])
        bus_peak[b] += float(net.load.at[i, "p_mw"])
        bus_q_peak[b] += float(net.load.at[i, "q_mvar"])

    # ===== LOAD time series (combined p + q) =====
    # each load bus follows CAISO demand shape; q = p * LOAD_PF_TAN
    load_cols = {}
    load_mapping = {}
    for b, p in bus_peak.items():
        pcol = f"load_p_bus_{b}"
        qcol = f"load_q_bus_{b}"
        p_series = demand_shape * p                # MW
        load_cols[pcol] = p_series
        load_cols[qcol] = p_series * LOAD_PF_TAN   # MVar (cos phi ~ 0.95)
        load_mapping[int(b)] = [[[pcol, qcol], 1.0]]
    load_df = pd.DataFrame(load_cols)
    load_df.to_csv(os.path.join(MIDAS_DATA, "socal_load_ts.csv"), index=False)
    p_total_peak = load_df[[c for c in load_df if c.startswith("load_p")]].sum(axis=1).max()
    print(f"  load TS: {load_df.shape[1]} cols x {load_df.shape[0]} steps "
          f"(active peak {p_total_peak:.0f} MW)")

    demand_total = demand_shape * sum(bus_peak.values())

    sgen_cols = {}
    sgen_mapping = collections.defaultdict(list)

    # ===== RENEWABLE generation (solar/wind/battery), per-bus capped =====
    renew_total = np.zeros(N_STEPS)
    renew_by_bus = collections.defaultdict(lambda: np.zeros(N_STEPS))
    for i in net.sgen.index:
        if str(net.sgen.at[i, "type"]) in ("local", "plant_ref"):
            continue
        b = int(net.sgen.at[i, "bus"])
        sn = float(net.sgen.at[i, "sn_mva"])
        typ = str(net.sgen.at[i, "type"])
        if typ == "SUN":
            series = solar_shape * sn * CF["SUN"]
        elif typ == "WND":
            series = wind_shape * sn * CF["WND"]
        elif typ == "BAT":
            t = np.arange(N_STEPS)
            bat = np.where((t >= 72) & (t < 88), 1.0, 0.0)  # 18:00-22:00
            series = bat * sn * CF["BAT"]
        else:
            series = np.zeros(N_STEPS)
        # per-bus hosting cap: prevents radial degree-1 stubs from diverging
        cap = RENEW_CAP_MULT * bus_peak.get(b, 0.0) + RENEW_CAP_ADD
        series = np.clip(series, 0.0, cap)
        pcol = f"sgen_{typ}_{i}_p_bus_{b}"
        qcol = f"sgen_{typ}_{i}_q_bus_{b}"
        sgen_cols[pcol] = series
        sgen_cols[qcol] = np.zeros(N_STEPS)        # renewables at unity PF
        sgen_mapping[b].append([[pcol, qcol], 1.0])
        renew_total += series
        renew_by_bus[b] += series

    # ===== LOCAL (conventional/dispatchable) generation, one per load bus =====
    # active: serves LOCAL_LOAD_FRAC of bus residual demand (load - renew@bus)
    # reactive: supplies LOCAL_Q_FRAC of bus reactive load (LOCAL REACTIVE SUPPORT)
    local_total = np.zeros(N_STEPS)
    localq_total = np.zeros(N_STEPS)
    for b, p in bus_peak.items():
        sidx = pp.create_sgen(net, bus=int(b), p_mw=0.0, q_mvar=0.0,
                              sn_mva=float(max(2.0 * p, 1.0)),
                              name=f"local_gen_{b}", type="local")
        pcol = f"sgen_LOCAL_{sidx}_p_bus_{b}"
        qcol = f"sgen_LOCAL_{sidx}_q_bus_{b}"
        bus_demand = demand_shape * p
        bus_demand_q = demand_shape * bus_q_peak[b]
        p_series = np.clip(LOCAL_LOAD_FRAC * bus_demand - renew_by_bus[b], 0.0, None)
        q_series = LOCAL_Q_FRAC * bus_demand_q
        sgen_cols[pcol] = p_series
        sgen_cols[qcol] = q_series
        sgen_mapping[int(b)].append([[pcol, qcol], 1.0])
        local_total += p_series
        localq_total += q_series
    print(f"  renew peak {renew_total.max():.0f} MW (capped); "
          f"local P peak {local_total.max():.0f} MW, "
          f"local Q peak {localq_total.max():.0f} MVar; "
          f"gen/demand {(renew_total+local_total).max()/demand_total.max():.2f}")

    sgen_df = pd.DataFrame(sgen_cols)
    sgen_df.to_csv(os.path.join(MIDAS_DATA, "socal_sgen_ts.csv"), index=False)
    p_sgen_peak = sgen_df[[c for c in sgen_df if "_p_bus_" in c]].sum(axis=1).max()
    print(f"  sgen TS: {sgen_df.shape[1]} cols x {sgen_df.shape[0]} steps "
          f"(active peak {p_sgen_peak:.0f} MW)")

    # ----- pre-seed grid load/sgen p,q at step-0 series values (clean first solve)
    for i in net.load.index:
        b = int(net.load.at[i, "bus"])
        # split the bus's step-0 value equally across its load elements
        n_at_bus = sum(1 for j in net.load.index if int(net.load.at[j, "bus"]) == b)
        net.load.at[i, "p_mw"] = (demand_shape[0] * bus_peak[b]) / n_at_bus
        net.load.at[i, "q_mvar"] = (demand_shape[0] * bus_peak[b] * LOAD_PF_TAN) / n_at_bus

    # sgen step-0 pre-seed (match the p column the powerseries will drive)
    p_step0 = {c: float(v[0]) for c, v in sgen_cols.items() if "_p_bus_" in c}
    q_step0 = {c: float(v[0]) for c, v in sgen_cols.items() if "_q_bus_" in c}
    sidx_to_cols = {}
    for b, entries in sgen_mapping.items():
        for (cols, _scale) in entries:
            pcol, qcol = cols
            sidx = int(pcol.split("_")[2])
            sidx_to_cols[sidx] = (pcol, qcol)
    for i in net.sgen.index:
        cols = sidx_to_cols.get(int(i))
        if cols:
            net.sgen.at[i, "p_mw"] = p_step0.get(cols[0], 0.0)
            net.sgen.at[i, "q_mvar"] = q_step0.get(cols[1], 0.0)
        else:
            net.sgen.at[i, "p_mw"] = 0.0
            net.sgen.at[i, "q_mvar"] = 0.0

    grid_out_local = os.path.join(HERE, "socal_grid_midas.json")
    grid_out_data = os.path.join(MIDAS_DATA, "socal_grid_midas.json")
    pp.to_json(net, grid_out_local)
    pp.to_json(net, grid_out_data)

    # ----- mappings (combined: each entry is [[p_col, q_col], scale]) -----
    for path in (MIDAS_DATA, HERE):
        with open(os.path.join(path, "load_mapping.json"), "w") as f:
            json.dump({str(k): v for k, v in load_mapping.items()}, f)
        with open(os.path.join(path, "sgen_mapping.json"), "w") as f:
            json.dump({str(k): v for k, v in sgen_mapping.items()}, f)

    print(f"  grid -> {grid_out_data}")
    print(f"  mappings (combined p+q) -> {MIDAS_DATA}/(load|sgen)_mapping.json")
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepare MIDAS inputs for the SoCal scenario")
    p.add_argument("date", nargs="?", default="2024-07-16", help="CAISO date YYYY-MM-DD")
    p.add_argument("--days", type=int, default=1, help="weather horizon in days")
    p.add_argument("--weather-source", default="isd", choices=["isd", "hrrr", "synthetic"])
    p.add_argument("--station", default="72295023174", help="NOAA ISD station id")
    a = p.parse_args()
    main(a.date, a.days, a.weather_source, a.station)
