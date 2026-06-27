"""Eaton-scenario sanity check + KPI report read entirely from the palaestrAI store.

Validates the no-suppression Eaton run:
  * fire spreads (burned cells grow monotonically) and moves SOUTH-WEST from the
    Eaton Canyon ignition toward Altadena/foothill communities (Santa Ana wind);
  * the DamageMapperAgent sheds load on fire-reached buses (served MW drops,
    customers disconnect, SAIDI accumulates);
  * reports final KPIs for the V02_STORE_EATON_RESULT writeup.

Run:
  PYTHONPATH=$PWD python analysis/eaton_sanity.py \
    --store postgresql://palaestrai:...@127.0.0.1:5433/palaestrai_eaton \
    --ignition-lon -118.09358 --ignition-lat 34.18604
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run, CUSTOMERS_PER_MW  # noqa: E402


def _lonlat_to_rc(lon, lat, bounds, nr, nc):
    minlon, minlat, maxlon, maxlat = bounds
    c = int((lon - minlon) / (maxlon - minlon) * (nc - 1))
    # row 0 is the NORTH edge (maxlat); raster grows southward
    r = int((maxlat - lat) / (maxlat - minlat) * (nr - 1))
    return r, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--gis-uid", default="gis_world")
    ap.add_argument("--grid-uid", default="socal_grid")
    ap.add_argument("--ignition-lon", type=float, default=-118.09358)
    ap.add_argument("--ignition-lat", type=float, default=34.18604)
    args = ap.parse_args()

    snaps, meta = read_run(args.store, gis_uid=args.gis_uid,
                           grid_uid=args.grid_uid)
    extent = meta["extent"]  # [minlon, maxlon, minlat, maxlat]
    bounds = (extent[0], extent[2], extent[1], extent[3])
    nr, nc = snaps[0]["fire_code"].shape
    ig_r, ig_c = _lonlat_to_rc(args.ignition_lon, args.ignition_lat,
                               bounds, nr, nc)

    n_steps = len(snaps)
    burned_n = [int((s["fire_code"] > 0).sum()) for s in snaps]
    cell_area_km2 = (meta["delta_m"] ** 2) / 1e6
    burned_km2 = [b * cell_area_km2 for b in burned_n]
    acres = [k * 247.105 for k in burned_km2]

    # fire centroid drift (in cells) to verify SW spread
    def centroid(fc):
        ys, xs = np.where(fc > 0)
        if len(ys) == 0:
            return (ig_r, ig_c)
        return (float(ys.mean()), float(xs.mean()))

    # measure drift from the FIRST step that has fire (just after ignition),
    # so we capture the early wind-driven SW push before the front hits the
    # non-burnable coastal/ocean boundary to the SW and growth saturates.
    first_fire = next((k for k, s in enumerate(snaps)
                       if (s["fire_code"] > 0).any()), 0)
    c_first = centroid(snaps[first_fire]["fire_code"])
    # peak-extent step (growth saturates once fuel boundary reached)
    peak = int(np.argmax(burned_n))
    c_last = centroid(snaps[peak]["fire_code"])
    d_row = c_last[0] - c_first[0]   # +row => SOUTH
    d_col = c_last[1] - c_first[1]   # -col => WEST

    served0 = meta["base_served"]
    served_last = snaps[-1]["served_mw"]
    cust_disc_last = snaps[-1]["cust_disc"]
    saidi_last = snaps[-1]["saidi"]
    shed_mw = max(0.0, served0 - served_last)

    print("=" * 64)
    print("EATON FIRE -- v0.2 no-suppression sanity check (from store)")
    print("=" * 64)
    print(f"steps stored          : {n_steps}")
    print(f"grid shape            : {nr} x {nc}  (cell {meta['delta_m']:.0f} m,"
          f" {cell_area_km2*1e6:.0f} m^2)")
    print(f"ignition cell (r,c)   : ({ig_r},{ig_c})  "
          f"lon/lat ({args.ignition_lon},{args.ignition_lat})")
    print("-" * 64)
    print("FIRE SPREAD")
    print(f"  burned cells   : {burned_n[0]} -> {burned_n[-1]}")
    print(f"  burned area    : {burned_km2[0]:.1f} -> {burned_km2[-1]:.1f} km^2"
          f"  ({acres[-1]:,.0f} acres)")
    print(f"  monotonic grow : {all(b2 >= b1 for b1, b2 in zip(burned_n, burned_n[1:]))}")
    print(f"  centroid drift : d_row={d_row:+.1f} (+=S), d_col={d_col:+.1f} (-=W)")
    sw = (d_row > 0) and (d_col < 0)
    print(f"  spreads SW     : {sw}  "
          f"({'SOUTH-WEST toward Altadena -- OK' if sw else 'CHECK direction'})")
    print("-" * 64)
    print("GRID IMPACT (load-shed)")
    print(f"  base served MW    : {served0:,.1f}")
    print(f"  final served MW   : {served_last:,.1f}")
    print(f"  shed MW           : {shed_mw:,.1f}  "
          f"({100*shed_mw/max(1e-9,served0):.1f}% of base)")
    print(f"  customers/MW      : {CUSTOMERS_PER_MW}")
    print(f"  customers disc.   : {cust_disc_last:,.0f}")
    print(f"  cumulative SAIDI  : {saidi_last:,.1f} customer-min/customer")
    print("=" * 64)

    # crude PASS/FAIL
    ok_spread = burned_n[-1] > burned_n[0] and sw
    ok_grid = shed_mw > 0 and cust_disc_last > 0
    print(f"SANITY: fire_spread={'PASS' if ok_spread else 'FAIL'}  "
          f"grid_impact={'PASS' if ok_grid else 'FAIL'}")
    return 0 if (ok_spread and ok_grid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
