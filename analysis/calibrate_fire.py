"""Standalone CA calibration harness (v0.5).

Drives the pure :class:`wildfire_cma.cma.WildfireCMA` (no palaestrAI / mosaik) at
speed to calibrate a fire's *no-firefighting baseline* against the real CAL FIRE
perimeter. Supports a **time-varying wind schedule** (list of per-env-step
(speed, dir) applied by mutating ``theta.wind_speed`` / ``theta.wind_dir_deg``
between ``advance()`` calls) -- this mirrors exactly the ``wind_schedule`` param
added to :class:`palaestrai_socal.gis_world_env.GisWorldEnvironment`, so params
found here transfer to the full palaestrAI runs bit-for-bit (same CA kernel).

Usage as a library: :func:`simulate` runs one fire; :func:`grid_search` sweeps
parameters and returns the best config meeting the Dice>=0.8 / area+-10% bar.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wildfire_cma.cma import WildfireCMA, Theta, BURNING, BURNED_OUT
from wildfire_cma import gis
from analysis.perimeter_validation import (
    load_perimeter_polygons, rasterize_perimeter, score, meets_bar,
)


# --------------------------------------------------------------------------- #
# Wind schedules (env-step resolution)
# --------------------------------------------------------------------------- #
def constant_wind(speed: float, direction: float, n: int) -> List[Tuple[float, float]]:
    return [(speed, direction)] * n


def santa_ana_schedule(n_steps: int, env_step_min: float,
                       peak_speed: float, base_speed: float,
                       direction: float,
                       peak_hours: float = 12.0,
                       decay_hours: float = 36.0) -> List[Tuple[float, float]]:
    """A realistic Santa-Ana wind envelope: strong early plateau then decay.

    Wind holds near ``peak_speed`` for ``peak_hours``, then decays exponentially
    toward ``base_speed`` over ``decay_hours``. Direction is constant (offshore
    NE for Santa-Ana); a time-varying direction can be layered on separately.
    """
    sched = []
    for k in range(n_steps):
        t_h = k * env_step_min / 60.0
        if t_h <= peak_hours:
            spd = peak_speed
        else:
            frac = np.exp(-(t_h - peak_hours) / decay_hours)
            spd = base_speed + (peak_speed - base_speed) * frac
        sched.append((float(spd), float(direction)))
    return sched


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class FireConfig:
    name: str
    perimeter_path: str
    ignition_lonlat: Tuple[float, float]
    bounds: Tuple[float, float, float, float]  # grid extent (padded around fire)
    nrows: int
    ncols: int
    n_steps: int = 60                # env steps
    env_step_min: float = 60.0       # minutes per env step
    dt_cma_min: float = 5.0          # CMA sub-step
    t_burn_steps: int = 6
    seed: int = 47
    official_acres: float = 0.0


def build_raster(cfg: FireConfig, dem_npz: Optional[str] = None):
    return gis.socal_from_srtm(nrows=cfg.nrows, ncols=cfg.ncols,
                               bounds=cfg.bounds, dem_npz=dem_npz,
                               seed=cfg.seed or 7)


def simulate(cfg: FireConfig, kappa: float, moisture: float,
             wind_schedule: Sequence[Tuple[float, float]],
             raster=None, dem_npz: Optional[str] = None) -> np.ndarray:
    """Run one no-firefighting fire, return final burned boolean mask.

    ``wind_schedule`` has one (speed, dir) per env step. Between env steps the
    CA advances ``env_step_min`` minutes under the current wind.
    """
    if raster is None:
        raster = build_raster(cfg, dem_npz=dem_npz)
    theta = Theta(
        ignition_points=[cfg.ignition_lonlat],
        wind_speed=wind_schedule[0][0],
        wind_dir_deg=wind_schedule[0][1],
        dead_fuel_moisture=moisture,
        kappa=kappa,
    )
    ca = WildfireCMA(raster, theta, dt_cma_min=cfg.dt_cma_min,
                     t_burn_steps=cfg.t_burn_steps, seed=cfg.seed)
    for k in range(cfg.n_steps):
        spd, ddeg = wind_schedule[min(k, len(wind_schedule) - 1)]
        ca.theta.wind_speed = float(spd)
        ca.theta.wind_dir_deg = float(ddeg)
        ca.advance(cfg.env_step_min)
    return ca.fire_mask()


def evaluate(cfg: FireConfig, kappa: float, moisture: float,
             wind_schedule: Sequence[Tuple[float, float]],
             real_mask: np.ndarray, raster) -> Dict[str, float]:
    sim = simulate(cfg, kappa, moisture, wind_schedule, raster=raster)
    m = score(sim, real_mask, raster.delta_m)
    m["kappa"] = kappa
    m["moisture"] = moisture
    return m


def load_real_mask(cfg: FireConfig) -> np.ndarray:
    polys = load_perimeter_polygons(cfg.perimeter_path)
    return rasterize_perimeter(polys, cfg.bounds, cfg.nrows, cfg.ncols)


# --------------------------------------------------------------------------- #
# Parameter search
# --------------------------------------------------------------------------- #
def grid_search(cfg: FireConfig,
                kappas: Sequence[float],
                moistures: Sequence[float],
                schedule_fn,
                schedule_grid: Sequence[dict],
                dem_npz: Optional[str] = None,
                verbose: bool = True) -> Tuple[dict, List[dict]]:
    """Sweep (kappa, moisture, wind-schedule) and return (best, all_results).

    ``schedule_fn(cfg, **kw)`` builds a wind schedule from each dict in
    ``schedule_grid``. Best = meets the bar with Dice highest; else closest to
    the bar by a combined penalty.
    """
    raster = build_raster(cfg, dem_npz=dem_npz)
    real_mask = load_real_mask(cfg)
    results: List[dict] = []
    for skw in schedule_grid:
        sched = schedule_fn(cfg, **skw)
        for kappa in kappas:
            for moisture in moistures:
                m = evaluate(cfg, kappa, moisture, sched, real_mask, raster)
                m["schedule"] = skw
                m["passes"] = meets_bar(m)
                results.append(m)
                if verbose:
                    print(f"  k={kappa:.2f} m={moisture:.3f} {skw} -> "
                          f"Dice={m['dice']:.3f} area={m['sim_acres']:.0f}ac "
                          f"({m['area_pct_err']:+.1f}%) pass={m['passes']}")

    def penalty(m):
        area_pen = abs(m["area_pct_err"]) / 10.0
        dice_pen = max(0.0, 0.8 - m["dice"]) * 10.0
        return dice_pen + area_pen

    passing = [m for m in results if m["passes"]]
    if passing:
        best = max(passing, key=lambda m: m["dice"])
    else:
        best = min(results, key=penalty)
    return best, results
