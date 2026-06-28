"""Extract per-step fire-spread timeline from the Eaton store for real-vs-sim validation."""
from __future__ import annotations
import os, sys, datetime as dt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.store_readers import read_run  # noqa: E402

STORE = "postgresql://palaestrai:socal_local_1782561794@127.0.0.1:5433/palaestrai_eaton"
START = dt.datetime(2025, 1, 8, 2, 18, 0, tzinfo=dt.timezone.utc)

snaps, meta = read_run(STORE, gis_uid="gis_world", grid_uid="socal_grid")
cell_area_km2 = (meta["delta_m"] ** 2) / 1e6
nr, nc = snaps[0]["fire_code"].shape

print(f"# grid {nr}x{nc}, cell {meta['delta_m']:.0f} m, cell_area {cell_area_km2:.3f} km^2")
print(f"# n snapshots: {len(snaps)}")
print(f"# extent (minlon,maxlon,minlat,maxlat): {meta['extent']}")
print()
print(f"{'step':>4} {'sim_hours':>9} {'UTC_time':>20} {'burned_cells':>12} {'acres':>14} {'km2':>10}")

# snapshots may be 2 per step (one per env). dedup by burned-cell value progression.
prev = None
rows = []
for i, s in enumerate(snaps):
    bn = int((s["fire_code"] > 0).sum())
    rows.append(bn)

# Determine cadence: if 2 snaps per step, take every other or unique burned counts over time.
# We'll just index snapshots in order; map to hours by proportion to 120 steps.
n = len(snaps)
for i, s in enumerate(snaps):
    bn = int((s["fire_code"] > 0).sum())
    acres = bn * cell_area_km2 * 247.105
    km2 = bn * cell_area_km2
    # approximate sim hour: snapshots span 120 steps => hour ~ i/(n-1)*120
    hr = i / max(1, (n - 1)) * 120.0
    t = START + dt.timedelta(hours=hr)
    if i % 4 == 0 or i == n - 1:  # print every ~4th to keep readable
        print(f"{i:>4} {hr:>9.1f} {t.strftime('%Y-%m-%d %H:%MZ'):>20} {bn:>12} {acres:>14,.0f} {km2:>10.1f}")

# also print key milestones: 6h, 12h, 24h, 48h, 72h, peak
print()
print("# --- milestones (nearest snapshot) ---")
def nearest(hr_target):
    idx = min(range(n), key=lambda i: abs(i/max(1,(n-1))*120.0 - hr_target))
    bn = int((snaps[idx]["fire_code"] > 0).sum())
    return idx, bn * cell_area_km2 * 247.105
for h in [3, 6, 12, 24, 36, 48, 72, 96, 120]:
    idx, ac = nearest(h)
    print(f"  ~{h:>3}h : {ac:>12,.0f} acres  (snap {idx})")
peak = int(np.argmax(rows))
print(f"  peak  : {rows[peak]*cell_area_km2*247.105:>12,.0f} acres  (snap {peak}, hour {peak/max(1,(n-1))*120.0:.1f})")
print(f"  final : {rows[-1]*cell_area_km2*247.105:>12,.0f} acres")
