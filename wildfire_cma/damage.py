"""Damage mapper D: S x G -> dG for the wildfire CMA.

Given the wildfire cellular state ``S`` (from :class:`~wildfire_cma.cma.WildfireCMA`)
and a pandapower grid ``G`` co-registered with the raster, the damage mapper
produces the grid mutation ``dG`` per the GUARDIAN spec:

* A **bus/node** ``v`` is failed (out of service) if the cell containing it is
  ``BURNING`` or ``BURNED_OUT``.
* An **overhead line** ``e`` is failed if the fire front comes within the
  radiant-heat clearance buffer ``d_clear`` of the line footprint.

Per timestep the mapper exposes the mutated grid ``G_t = G (+) dG_t`` so the
power flow runs on a non-stationary topology. The mapper caches the
(bus -> cell) and (line -> cells) co-registration so per-step evaluation is
cheap even for the full 2,294-bus SoCal grid.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .cma import BURNED_OUT, BURNING, RasterStack, WildfireCMA


def _bus_lonlat(net, bus_idx) -> Optional[Tuple[float, float]]:
    geo = net.bus.at[bus_idx, "geo"] if "geo" in net.bus.columns else None
    if geo is None or (isinstance(geo, float) and math.isnan(geo)):
        return None
    try:
        g = json.loads(geo) if isinstance(geo, str) else geo
        lon, lat = g["coordinates"][0], g["coordinates"][1]
        return float(lon), float(lat)
    except Exception:
        return None


def _line_coords(net, line_idx) -> List[Tuple[float, float]]:
    geo = net.line.at[line_idx, "geo"] if "geo" in net.line.columns else None
    if geo is None or (isinstance(geo, float) and not isinstance(geo, str) and math.isnan(geo)):
        return []
    try:
        g = json.loads(geo) if isinstance(geo, str) else geo
        return [(float(x), float(y)) for x, y in g["coordinates"]]
    except Exception:
        return []


@dataclass
class DamageState:
    """Result of one damage-mapper evaluation."""

    failed_buses: Set[int] = field(default_factory=set)
    failed_lines: Set[int] = field(default_factory=set)
    newly_failed_buses: Set[int] = field(default_factory=set)
    newly_failed_lines: Set[int] = field(default_factory=set)


class DamageMapper:
    """Co-registers a pandapower grid with the CMA raster and fails assets.

    Parameters
    ----------
    net:
        pandapower network with ``geo`` GeoJSON columns on buses and lines.
    raster:
        the raster stack used by the CMA (for lon/lat <-> row/col mapping).
    clearance_m:
        radiant-heat clearance buffer ``d_clear`` (metres) around the fire
        front within which an overhead line is failed.
    """

    def __init__(self, net, raster: RasterStack, clearance_m: float = 90.0):
        self.net = net
        self.raster = raster
        self.clearance_m = float(clearance_m)
        self._clear_cells = max(1, int(round(clearance_m / raster.delta_m)))

        # cache bus -> cell
        self.bus_cell: Dict[int, Tuple[int, int]] = {}
        for b in net.bus.index:
            ll = _bus_lonlat(net, b)
            if ll is not None and self._in_bounds(*ll):
                self.bus_cell[int(b)] = raster.lonlat_to_rc(*ll)

        # cache line -> set of cells it passes through (rasterised footprint)
        self.line_cells: Dict[int, List[Tuple[int, int]]] = {}
        for ln in net.line.index:
            coords = _line_coords(net, ln)
            cells = []
            for (lon, lat) in coords:
                if self._in_bounds(lon, lat):
                    cells.append(raster.lonlat_to_rc(lon, lat))
            if cells:
                self.line_cells[int(ln)] = cells

        self._failed_buses: Set[int] = set()
        self._failed_lines: Set[int] = set()

    def _in_bounds(self, lon: float, lat: float) -> bool:
        minlon, minlat, maxlon, maxlat = self.raster.bounds
        return (minlon <= lon <= maxlon) and (minlat <= lat <= maxlat)

    def _fire_distance_field(self, fire_mask: np.ndarray) -> np.ndarray:
        """Chebyshev distance (in cells) from each cell to the nearest fire cell.

        Uses a fast iterative dilation up to the clearance radius; cells beyond
        the buffer get a large sentinel value.
        """
        from scipy.ndimage import binary_dilation
        try:
            struct = np.ones((3, 3), bool)
            reached = fire_mask.copy()
            dist = np.full(fire_mask.shape, self._clear_cells + 1, dtype=np.int16)
            dist[fire_mask] = 0
            cur = fire_mask
            for d in range(1, self._clear_cells + 1):
                nxt = binary_dilation(cur, structure=struct)
                ring = nxt & ~reached
                dist[ring] = d
                reached = nxt
                cur = nxt
            return dist
        except Exception:
            # scipy missing: fall back to fire cells only (distance 0/inf)
            dist = np.full(fire_mask.shape, self._clear_cells + 1, dtype=np.int16)
            dist[fire_mask] = 0
            return dist

    def evaluate(self, cma: WildfireCMA) -> DamageState:
        """Compute dG: which buses/lines are failed by the current fire state."""
        fire = (cma.state == BURNING) | (cma.state == BURNED_OUT)

        # --- buses: failed if their cell is on fire -----------------------
        newly_buses: Set[int] = set()
        for b, (r, c) in self.bus_cell.items():
            if fire[r, c] and b not in self._failed_buses:
                newly_buses.add(b)
        self._failed_buses |= newly_buses

        # --- lines: failed if footprint within clearance of the front -----
        dist = self._fire_distance_field(fire)
        newly_lines: Set[int] = set()
        for ln, cells in self.line_cells.items():
            if ln in self._failed_lines:
                continue
            for (r, c) in cells:
                if dist[r, c] <= self._clear_cells:
                    newly_lines.add(ln)
                    break
        self._failed_lines |= newly_lines

        return DamageState(
            failed_buses=set(self._failed_buses),
            failed_lines=set(self._failed_lines),
            newly_failed_buses=newly_buses,
            newly_failed_lines=newly_lines,
        )

    def apply(self, net=None) -> DamageState:
        """Evaluate is separate; this applies the accumulated dG to the grid.

        Sets failed buses and lines ``in_service = False`` (the topology
        mutation ``G_t = G (+) dG_t``). Returns the current damage state.
        """
        net = net or self.net
        ds = DamageState(failed_buses=set(self._failed_buses),
                         failed_lines=set(self._failed_lines))
        if self._failed_buses:
            idx = [b for b in self._failed_buses if b in net.bus.index]
            net.bus.loc[idx, "in_service"] = False
            # take out elements attached to dead buses
            for tbl in ("load", "sgen", "gen"):
                if hasattr(net, tbl) and len(getattr(net, tbl)):
                    t = getattr(net, tbl)
                    t.loc[t["bus"].isin(idx), "in_service"] = False
        if self._failed_lines:
            lidx = [l for l in self._failed_lines if l in net.line.index]
            net.line.loc[lidx, "in_service"] = False
        return ds

    def reset(self) -> None:
        self._failed_buses.clear()
        self._failed_lines.clear()
