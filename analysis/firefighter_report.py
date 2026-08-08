"""Acres-saved report for the v0.3 FirefighterAgent (store-only).

Compares the final burned footprint of one or more firefighter runs against a
zero-plane baseline, to quantify how much area the aero-tanker fleet protects as
``n_planes`` increases. Like :mod:`analysis.make_timelapse`, it is **store-only**
-- it re-reads each run's ``world_states`` rows via
:func:`analysis.store_readers.read_run` and never re-runs a simulation.

Workflow (the coding session does NOT run these; the user does, after review):

  # baseline + a small n_planes sweep, each into its own store db:
  for n in 0 1 3 5 7; do
    # edit experiment_eaton_local.yml firefighter n_planes: $n  (or template it)
    env PYTHONPATH=$PWD palaestrai -c runtime_pg_eaton.conf.yaml start \
      palaestrai_socal/experiment_eaton_local.yml
    # ... copy/point the resulting store to e.g. _outputs/eaton_local_n$n.db
  done

  python analysis/firefighter_report.py \
    --run 0=postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai \
    --run 1=postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai \
    --run 3=postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai \
    --run 5=postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai \
    --run 7=postgresql://palaestrai:socal_local@127.0.0.1:5433/palaestrai

The lowest ``n_planes`` supplied is the baseline; acres-saved for every other
run is ``baseline_burned_acres - run_burned_acres``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run  # noqa: E402

SQM_PER_ACRE = 4046.8564224


def burned_acres(store_uri: str, gis_uid: str = "gis_world",
                 grid_uid: str = "socal_grid",
                 env_step_min: float = 60.0) -> Tuple[float, int, float]:
    """Final burned area for one run: ``(acres, burned_cells, cell_size_m)``.

    Burned = cells BURNING or BURNED_OUT in the last reconstructed frame
    (``fire_code`` 1 or 2; SUPPRESSED retardant cells are *not* counted as
    burned -- that is the whole point). Acres = ``cells * delta_m^2 / 4046.86``.
    """
    snaps, meta = read_run(store_uri, gis_uid=gis_uid, grid_uid=grid_uid,
                           env_step_min=env_step_min)
    delta_m = float(meta["delta_m"])
    last = snaps[-1]["fire_code"]
    burned_cells = int(np.count_nonzero((last == 1) | (last == 2)))
    acres = burned_cells * (delta_m * delta_m) / SQM_PER_ACRE
    return acres, burned_cells, delta_m


def build_report(runs: Dict[int, str], **read_kw) -> List[dict]:
    """Compute the acres-saved table for ``{n_planes: store_uri}``.

    The smallest ``n_planes`` key is the baseline. Returns one row per run with
    burned acres and acres/percent saved vs the baseline, sorted by n_planes.
    """
    rows: List[dict] = []
    for n in sorted(runs):
        acres, cells, delta_m = burned_acres(runs[n], **read_kw)
        rows.append({"n_planes": n, "burned_acres": acres,
                     "burned_cells": cells, "cell_size_m": delta_m})
    base = rows[0]["burned_acres"]
    for r in rows:
        r["acres_saved"] = base - r["burned_acres"]
        r["pct_saved"] = (100.0 * r["acres_saved"] / base) if base > 0 else 0.0
    return rows


def format_report(rows: List[dict]) -> str:
    base_n = rows[0]["n_planes"]
    out = [
        f"Firefighter acres-saved report (baseline = n_planes={base_n})",
        f"cell size ~ {rows[0]['cell_size_m']:.0f} m",
        "",
        f"{'n_planes':>8} {'burned_acres':>14} {'burned_cells':>13} "
        f"{'acres_saved':>13} {'pct_saved':>10}",
    ]
    for r in rows:
        out.append(
            f"{r['n_planes']:>8d} {r['burned_acres']:>14,.1f} "
            f"{r['burned_cells']:>13,d} {r['acres_saved']:>13,.1f} "
            f"{r['pct_saved']:>9.1f}%")
    return "\n".join(out)


def _parse_run(arg: str) -> Tuple[int, str]:
    """Parse a ``N=URI`` CLI argument into ``(n_planes, store_uri)``."""
    if "=" not in arg:
        raise ValueError(f"--run expects N=URI, got {arg!r}")
    n_str, uri = arg.split("=", 1)
    return int(n_str), uri


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[], metavar="N=URI",
                    help="n_planes=store_uri for one run (repeatable)")
    ap.add_argument("--gis-uid", default="gis_world")
    ap.add_argument("--grid-uid", default="socal_grid")
    ap.add_argument("--env-step-min", type=float, default=60.0)
    args = ap.parse_args()

    if not args.run:
        ap.error("supply at least one --run N=URI (e.g. --run 0=postgresql://...)")
    runs = dict(_parse_run(a) for a in args.run)
    rows = build_report(runs, gis_uid=args.gis_uid, grid_uid=args.grid_uid,
                        env_step_min=args.env_step_min)
    print(format_report(rows))
