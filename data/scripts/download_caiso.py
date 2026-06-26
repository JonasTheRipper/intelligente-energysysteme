#!/usr/bin/env python3
"""
Download real CAISO actuals for a representative day to drive the MIDAS
co-simulation of the Southern California grid:

  - system-wide actual demand (5-minute)
  - actual solar generation (5-minute)
  - actual wind generation   (5-minute)

CAISO operates ~80% of California's load; Southern California (SP15) is roughly
half of the CAISO footprint. We download the CAISO system actuals and, in the
scenario builder, scale them to the SoCal model's 35 GW peak so the *shape*
(day/night demand curve, solar belly, wind ramp) is real CAISO data while the
magnitude matches our model.

Output: data/caiso_<date>.csv with columns
  time, demand_mw, solar_mw, wind_mw  (5-minute resolution, local CA time)
"""
import os, sys, argparse
import pandas as pd
import gridstatus

HERE = os.path.dirname(__file__)


def fetch(date="2024-07-16", out=None):
    iso = gridstatus.CAISO()
    print(f"Fetching CAISO actuals for {date} ...")

    load = iso.get_load(date, verbose=False)          # 5-min, 'Load'
    load = load[["Time", "Load"]].rename(columns={"Load": "demand_mw"})

    # solar & wind actuals come from the fuel mix
    fm = iso.get_fuel_mix(date, verbose=False)
    fmt = fm.rename(columns={c: c.strip() for c in fm.columns})
    cols = {c.lower(): c for c in fmt.columns}
    solar_col = cols.get("solar")
    wind_col = cols.get("wind")
    sw = fmt[["Time", solar_col, wind_col]].rename(
        columns={solar_col: "solar_mw", wind_col: "wind_mw"})

    df = pd.merge_asof(load.sort_values("Time"), sw.sort_values("Time"),
                       on="Time", direction="nearest")
    df = df.rename(columns={"Time": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["demand_mw"]).reset_index(drop=True)

    if out is None:
        out = os.path.join(HERE, f"caiso_{date}.csv")
    df.to_csv(out, index=False)
    print(f"  rows: {len(df)}  span: {df.time.min()} .. {df.time.max()}")
    print(f"  demand MW: {df.demand_mw.min():.0f}..{df.demand_mw.max():.0f}")
    print(f"  solar  MW: {df.solar_mw.min():.0f}..{df.solar_mw.max():.0f}")
    print(f"  wind   MW: {df.wind_mw.min():.0f}..{df.wind_mw.max():.0f}")
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-07-16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fetch(a.date, a.out)
