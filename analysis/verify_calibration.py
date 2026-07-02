"""Standalone verification of v0.5 production code path (NO monkeypatch).

For each fire, builds a WildfireDriver with the production kwargs (perimeter_path,
base_speed, boundary_gain, dead_fuel_moisture, kappa, fuel_reclass), runs the
no-firefighting CA for the experiment's max_steps (60) env steps, and prints the
max-extent (peak) Dice + area% vs the official CAL FIRE perimeter.

Per the INTEGRATION SPEC (STEP 5), for Palisades the "widest extent" metrics
are reported; the same interpretation applies to Eaton (the fire overshoots the
perimeter after step 14, so peak-Dice is the meaningful measure).

Hard bar (must pass both):
    peak Dice >= 0.80 AND |area%| at peak <= 10

Run:
    cd /home/user/workspace/socal-wildfires
    source .venv/bin/activate && export PYTHONPATH=$PWD
    python analysis/verify_calibration.py
"""
from __future__ import annotations

import math
import sys
import numpy as np

from wildfire_cma import gis
from wildfire_cma.cma import UNBURNED
from palaestrai_socal.agents.wildfire_core import WildfireDriver
from analysis.perimeter_validation import (
    load_perimeter_polygons,
    rasterize_perimeter,
    score,
    meets_bar,
)


def run_fire(
    name: str,
    perimeter_path: str,
    ignition_lonlat,
    bounds,
    nrows: int,
    ncols: int,
    max_steps: int,
    env_step_min: float,
    base_speed: float,
    boundary_gain: float,
    dead_fuel_moisture: float,
    kappa: float,
    fuel_reclass: bool,
    containment_margin: int = 2,
    seed: int = 47,
    verbose: bool = True,
):
    """Build driver via production path (no monkeypatch), advance, score peak."""
    # Build raster the same way calibration does
    raster = gis.socal_from_srtm(
        nrows=nrows, ncols=ncols, bounds=bounds, seed=seed
    )

    # Build the driver — PRODUCTION path, mirrors WildfireCmaMuscle
    driver = WildfireDriver(
        fuel=raster.fuel,
        dem=raster.dem,
        delta_m=raster.delta_m,
        bounds=bounds,
        ignition_points=[ignition_lonlat],
        ignition_step=1,
        env_step_min=env_step_min,
        dt_cma_min=5.0,
        t_burn_steps=6,
        kappa=kappa,
        dead_fuel_moisture=dead_fuel_moisture,
        wind_speed=base_speed,   # fallback scalar (overridden by wind_field)
        wind_dir_deg=45.0,
        seed=seed,
        perimeter_path=perimeter_path,
        base_speed=base_speed,
        boundary_gain=boundary_gain,
        fuel_reclass=fuel_reclass,
        containment_margin=containment_margin,
    )

    # Load real mask at the same grid resolution
    polys = load_perimeter_polygons(perimeter_path)
    real_mask = rasterize_perimeter(polys, bounds, nrows, ncols)

    # Build initial cell_state grid (all unburned)
    state = np.full((nrows, ncols), UNBURNED, dtype=np.int8)

    # Run max_steps steps and track max-extent (peak Dice) metrics
    best_m = None
    best_step = 0
    final_m = None
    print(f"\n--- {name} verification: grid={nrows}x{ncols}, bounds={bounds} ---")
    for k in range(max_steps):
        muts = driver.step(state)
        for (r, c, s, _layer) in muts:
            state[r, c] = s

        sim_mask = state != UNBURNED
        m = score(sim_mask, real_mask, raster.delta_m)
        ok = meets_bar(m)
        final_m = m  # last iteration wins

        if verbose and (k < 16 or (k + 1) % 10 == 0):
            print(
                f"  step {k+1:3d}: Dice={m['dice']:.3f}  "
                f"sim={m['sim_acres']:.0f}ac  area%={m['area_pct_err']:+.1f}%"
                + ("  <== PASS" if ok else "")
            )

        # Track peak by Dice (tie-break by area error)
        if best_m is None or m["dice"] > best_m["dice"]:
            best_m = m
            best_step = k + 1

    ok_best = meets_bar(best_m)
    ok_final = meets_bar(final_m)
    print(
        f"\n{name} PEAK (step {best_step}/{max_steps}): "
        f"Dice={best_m['dice']:.3f}  "
        f"sim={best_m['sim_acres']:.0f}ac  real={best_m['real_acres']:.0f}ac  "
        f"area%={best_m['area_pct_err']:+.1f}%  "
        f"{'PASS' if ok_best else 'FAIL'}"
    )
    print(
        f"{name} FINAL (step {max_steps}/{max_steps}): "
        f"Dice={final_m['dice']:.3f}  "
        f"sim={final_m['sim_acres']:.0f}ac  real={final_m['real_acres']:.0f}ac  "
        f"area%={final_m['area_pct_err']:+.1f}%  "
        f"{'PASS' if ok_final else 'FAIL'}  "
        f"(containment_margin={containment_margin})"
    )
    # With containment enabled, the FINAL step must also meet the bar (fire is
    # arrested at the perimeter and stays there, not just at a transient peak).
    return best_m, (ok_best and ok_final), best_step, final_m, ok_final


def main():
    results = {}
    all_pass = True

    # ------------------------------------------------------------------ EATON
    # Calibration validated: steps=14 (env), spd=14, bg=0.3, moist=0.13,
    # kappa=1.5 -> Dice=0.813, area -4.1%.
    # Grid: pad=0.015 around perimeter bbox [-118.1621,-118.0131,34.1619,34.2378].
    # Spec says to prefer calibration crop nrows=130, ncols=182 if 222x184 drops
    # below 0.8.

    EATON_BB = (-118.1621, 34.1619, -118.0131, 34.2378)
    pad = 0.015
    eaton_bounds = (
        EATON_BB[0] - pad, EATON_BB[1] - pad,
        EATON_BB[2] + pad, EATON_BB[3] + pad,
    )
    # Compute nrows/ncols the same way the prototypes did (90 m target cells)
    dlat = eaton_bounds[3] - eaton_bounds[1]
    dlon = eaton_bounds[2] - eaton_bounds[0]
    midlat = (eaton_bounds[1] + eaton_bounds[3]) / 2
    eaton_nrows = int(dlat * 111000 / 90)
    eaton_ncols = int(dlon * 111000 * math.cos(math.radians(midlat)) / 90)
    print(f"Eaton calibration grid: {eaton_nrows}x{eaton_ncols}, bounds={eaton_bounds}")

    m_eaton, ok_eaton, eaton_peak_step, m_eaton_final, ok_eaton_final = run_fire(
        name="EATON",
        perimeter_path="data/perimeters/eaton_perimeter.geojson",
        ignition_lonlat=(-118.0935761, 34.1860422),
        bounds=eaton_bounds,
        nrows=eaton_nrows,
        ncols=eaton_ncols,
        max_steps=60,
        env_step_min=60.0,
        base_speed=14.0,
        boundary_gain=0.3,
        dead_fuel_moisture=0.13,
        kappa=1.5,
        fuel_reclass=False,
        containment_margin=2,
        seed=47,
    )
    results["eaton"] = m_eaton
    results["eaton_ok"] = ok_eaton
    results["eaton_peak_step"] = eaton_peak_step
    results["eaton_final"] = m_eaton_final
    if not ok_eaton:
        all_pass = False

    # -------------------------------------------------------------- PALISADES
    # Calibration validated: steps=28 (env), spd=16, bg=0.6, moist=0.08,
    # kappa=1.5 -> Dice=0.822, area -0.4%.
    # Grid: nrows=159, ncols=219, bounds=(-118.7009, 34.0148, -118.4856, 34.1444)
    # Spec says max_steps=60 is fine; fire reaches steady state by step ~28.

    palisades_bounds = (-118.7009, 34.0148, -118.4856, 34.1444)

    m_pal, ok_pal, pal_peak_step, m_pal_final, ok_pal_final = run_fire(
        name="PALISADES",
        perimeter_path="data/perimeters/palisades_perimeter.geojson",
        ignition_lonlat=(-118.5426, 34.0781),
        bounds=palisades_bounds,
        nrows=159,
        ncols=219,
        max_steps=60,
        env_step_min=60.0,
        base_speed=16.0,
        boundary_gain=0.6,
        dead_fuel_moisture=0.08,
        kappa=1.5,
        fuel_reclass=True,
        containment_margin=2,
        seed=47,
    )
    results["palisades"] = m_pal
    results["palisades_ok"] = ok_pal
    results["palisades_peak_step"] = pal_peak_step
    results["palisades_final"] = m_pal_final
    if not ok_pal:
        all_pass = False

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print(
        f"  Eaton     peak(step {results['eaton_peak_step']}): "
        f"Dice={results['eaton']['dice']:.3f} area%={results['eaton']['area_pct_err']:+.1f}%  |  "
        f"final(60): Dice={results['eaton_final']['dice']:.3f} "
        f"area%={results['eaton_final']['area_pct_err']:+.1f}%  "
        f"{'PASS' if results['eaton_ok'] else 'FAIL'}"
    )
    print(
        f"  Palisades peak(step {results['palisades_peak_step']}): "
        f"Dice={results['palisades']['dice']:.3f} area%={results['palisades']['area_pct_err']:+.1f}%  |  "
        f"final(60): Dice={results['palisades_final']['dice']:.3f} "
        f"area%={results['palisades_final']['area_pct_err']:+.1f}%  "
        f"{'PASS' if results['palisades_ok'] else 'FAIL'}"
    )
    print(f"  OVERALL: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
