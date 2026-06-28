"""Quantitative validation: v0.2 Eaton sim vs. real Eaton Fire (Jan 2025).

Produces:
  * a real-vs-sim acreage growth curve (log scale) PNG
  * centroid spread distance/direction in km
  * printed comparison table
"""
from __future__ import annotations
import os, sys, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from analysis.store_readers import read_run  # noqa: E402

STORE = "postgresql://palaestrai:socal_local_1782561794@127.0.0.1:5433/palaestrai_eaton"
START = dt.datetime(2025, 1, 8, 2, 18, 0, tzinfo=dt.timezone.utc)  # sim start (= ignition+ ~step0)
IGN_REAL = dt.datetime(2025, 1, 8, 2, 18, 0, tzinfo=dt.timezone.utc)  # 6:18pm PST Jan 7

# ---- real timeline (hours since 6:18pm PST Jan 7 -> acres) from CAL FIRE / Wikipedia/LAT ----
# (time, acres)  PST timestamps converted to hours-since-ignition
real = [
    (0.13, 10),      # 6:26pm  - 10 ac
    (5.8, 400),      # ~12:07am Jan8 (>1000 soon after midnight; 400 by midnight)
    (5.9, 1000),     # 12:07am Jan8 >1000
    (10.2, 2227),    # 6:30am Jan8 - 2,227 (Wikipedia/inciweb 7am=2,227)
    (16.3, 10600),   # 10:36am Jan8 - 10,600
    (40.0, 13690),   # Jan 9 ~ midday (ABC/Wiki Jan9 13,690)
    (52.0, 14117),   # evening Jan 10 reaches 14,117 (peak)
    (120.0, 14021),  # plateau (final 14,021)
]
real_h = [r[0] for r in real]
real_ac = [r[1] for r in real]

# ---- sim timeline ----
snaps, meta = read_run(STORE, gis_uid="gis_world", grid_uid="socal_grid")
cell_area_km2 = (meta["delta_m"] ** 2) / 1e6
nr, nc = snaps[0]["fire_code"].shape
n = len(snaps)
sim_h, sim_ac = [], []
for i, s in enumerate(snaps):
    bn = int((s["fire_code"] > 0).sum())
    sim_h.append(i / max(1, (n - 1)) * 120.0)
    sim_ac.append(bn * cell_area_km2 * 247.105)

# ---- centroid spread in km ----
extent = meta["extent"]  # [minlon,maxlon,minlat,maxlat]
minlon, maxlon, minlat, maxlat = extent
def centroid(fc):
    ys, xs = np.where(fc > 0)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())
first = next((k for k in range(n) if (snaps[k]["fire_code"] > 0).any()), 0)
peak = int(np.argmax([(s["fire_code"] > 0).sum() for s in snaps]))
c0 = centroid(snaps[first]["fire_code"])
c1 = centroid(snaps[peak]["fire_code"])
# cell -> km. row spacing & col spacing
km_per_row = (maxlat - minlat) * 111.0 / (nr - 1)
mean_lat = (minlat + maxlat) / 2
km_per_col = (maxlon - minlon) * 111.0 * np.cos(np.radians(mean_lat)) / (nc - 1)
d_row = c1[0] - c0[0]   # +south
d_col = c1[1] - c0[1]   # -west
south_km = d_row * km_per_row
west_km = -d_col * km_per_col
total_km = (south_km**2 + west_km**2) ** 0.5

print("=" * 70)
print("EATON FIRE: v0.2 SIM vs REAL  (quantitative validation)")
print("=" * 70)
print(f"{'metric':<34}{'REAL':>16}{'SIM v0.2':>18}")
print("-" * 70)
print(f"{'final burned area (acres)':<34}{'14,021':>16}{sim_ac[-1]:>18,.0f}")
print(f"{'  ratio sim/real':<34}{'':>16}{sim_ac[-1]/14021:>17.1f}x")
print(f"{'peak reached at (h since ign.)':<34}{'~52 (Jan10 eve)':>16}{sim_h[peak]:>17.1f}h")
print(f"{'~10k-acre milestone at (h)':<34}{'16.3 (10,600)':>16}{'see curve':>18}")
print(f"{'spread direction':<34}{'E then strong W':>16}{'SW':>18}")
print(f"{'centroid net drift (km)':<34}{'~3-4 km W':>16}{f'{west_km:.1f}W/{south_km:.1f}S':>18}")
_res = f"{meta['delta_m']:.0f} m"
print(f"{'cell resolution':<34}{'~10-30 m (real fuels)':>16}{_res:>18}")
print("=" * 70)
print(f"sim centroid drift: {south_km:+.1f} km S, {west_km:+.1f} km W (total {total_km:.1f} km)")
print(f"sim t=0 already burned: {sim_ac[0]:,.0f} acres (1 cell = {cell_area_km2*247.105:,.0f} acres)")

# ---- plot ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(real_h, real_ac, "o-", color="#c0392b", lw=2.2, ms=7, label="Real Eaton Fire (CAL FIRE)")
ax.plot(sim_h, sim_ac, "s-", color="#2471a3", lw=2.0, ms=4, label="v0.2 simulation")
ax.axhline(14021, ls="--", color="#c0392b", alpha=0.4)
ax.text(122, 14021, "  real final\n  14,021 ac", va="center", color="#c0392b", fontsize=8)
ax.axhline(sim_ac[-1], ls="--", color="#2471a3", alpha=0.4)
ax.text(122, sim_ac[-1], f"  sim final\n  {sim_ac[-1]:,.0f} ac", va="center", color="#2471a3", fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("Hours since ignition (Jan 7, 6:18 pm PST)")
ax.set_ylabel("Burned area (acres, log scale)")
ax.set_title("Eaton Fire growth: real vs. v0.2 simulation\n(coarse 947 m grid over statewide SoCal extent)")
ax.legend(loc="lower right")
ax.grid(True, which="both", alpha=0.25)
ax.set_xlim(-3, 135)
fig.tight_layout()
out = os.path.join(_HERE, "eaton_validation_growth.png")
fig.savefig(out, dpi=130)
print(f"\nsaved: {out}")
