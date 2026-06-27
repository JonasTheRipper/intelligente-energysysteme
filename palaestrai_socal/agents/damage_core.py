"""Numpy-only damage driver behind the :class:`DamageMapperMuscle`.

The v0.2 design moves the wildfire->grid *damage mapping* out of the monolithic
environment and into an agent. This module is the agent-side brain, kept free of
any palaestrAI / mosaik import so it is unit-testable with numpy alone
(``tests/test_damage_agent.py``).

It co-registers power-grid buses with the GIS raster (lon/lat -> row/col) and,
each step, reads the authoritative GIS cell-state grid ``S`` to decide which
buses are *fire-affected* (their cell is ``BURNING`` or ``BURNED_OUT``). The
muscle then sheds the loads attached to those buses by driving their MIDAS
``...load-<bus>-<idx>.p_mw`` actuators to ``0``.

Important v0.2 deviation from v0.1
----------------------------------
v0.1 tripped buses/lines by flipping ``in_service=False`` directly on the
pandapower net. Under the REAL MIDAS/mosaik co-simulation the only writable grid
control surface is the powergrid simulator's *load* (and ``sgen``) ``p_mw``
actuators -- mosaik exposes no ``bus.in_service`` / ``line.in_service``
actuator. So v0.2 realises the same de-energisation as a **load-shed trip**:
the load on a fire-affected bus is set to ``0 MW``. This reproduces the served-
load shortfall KPI while staying within the native MIDAS actuator set.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# cell-state codes (mirror palaestrai_socal.spaces / wildfire_cma.cma)
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2

# Powergrid-0.0-load-<bus>-<idx>.p_mw  (optionally env-prefixed). The first
# integer after ``load-`` is the bus the load is attached to.
_LOAD_RE = re.compile(r"load-(\d+)-\d+\.p_mw$")


def load_actuator_bus(uid: str) -> Optional[int]:
    """Extract the bus index from a ``...load-<bus>-<idx>.p_mw`` actuator uid.

    Returns ``None`` for any uid that is not a load real-power actuator (e.g. a
    ``q_mvar`` actuator or a bus/line sensor), so the caller can skip it.
    """
    m = _LOAD_RE.search(uid)
    return int(m.group(1)) if m else None


def _lonlat_to_rc(
    lon: float,
    lat: float,
    bounds: Tuple[float, float, float, float],
    nrows: int,
    ncols: int,
) -> Tuple[int, int]:
    """Map a geographic point to a raster ``(row, col)``.

    Matches :meth:`wildfire_cma.cma.RasterStack.lonlat_to_rc` *exactly* (it uses
    ``ncols - 1`` / ``nrows - 1`` as the span divisor); co-registration must
    agree cell-for-cell with the GIS substrate, so this is not an approximation.
    Row 0 is the northern (max-lat) edge, column 0 the western (min-lon) edge.
    """
    minlon, minlat, maxlon, maxlat = bounds
    fx = (lon - minlon) / (maxlon - minlon) if maxlon > minlon else 0.0
    fy = (maxlat - lat) / (maxlat - minlat) if maxlat > minlat else 0.0
    c = int(np.clip(int(fx * (ncols - 1)), 0, ncols - 1))
    r = int(np.clip(int(fy * (nrows - 1)), 0, nrows - 1))
    return r, c


class DamageMapperDriver:
    """Maps GIS fire state to fire-affected buses (load-shed targets).

    Parameters
    ----------
    bus_lonlat:
        ``{bus_index: (lon, lat)}`` for every controllable bus.
    bounds:
        raster ``(minlon, minlat, maxlon, maxlat)`` (``gis.bounds`` sensor).
    shape:
        raster ``(nrows, ncols)`` (``gis.grid_shape`` sensor).

    The driver *latches* shed buses: once a bus is fire-affected it stays shed
    for the rest of the episode (a burned-out feeder is not re-energised), which
    matches the monotonic v0.1 damage accumulation.
    """

    def __init__(
        self,
        bus_lonlat: Dict[int, Tuple[float, float]],
        bounds: Tuple[float, float, float, float],
        shape: Tuple[int, int],
    ):
        self.bounds = tuple(float(x) for x in bounds)
        self.nrows, self.ncols = int(shape[0]), int(shape[1])
        self.bus_cell: Dict[int, Tuple[int, int]] = {}
        for b, ll in bus_lonlat.items():
            if ll is None:
                continue
            lon, lat = float(ll[0]), float(ll[1])
            if self._in_bounds(lon, lat):
                self.bus_cell[int(b)] = _lonlat_to_rc(
                    lon, lat, self.bounds, self.nrows, self.ncols
                )
        self._shed: Set[int] = set()

    def _in_bounds(self, lon: float, lat: float) -> bool:
        minlon, minlat, maxlon, maxlat = self.bounds
        return (minlon <= lon <= maxlon) and (minlat <= lat <= maxlat)

    @property
    def shed_buses(self) -> Set[int]:
        """Buses shed so far this episode (latched, monotonic)."""
        return set(self._shed)

    def evaluate(
        self,
        cell_state: np.ndarray,
        buses: Optional[Iterable[int]] = None,
    ) -> Set[int]:
        """Return the latched set of fire-affected buses given the GIS state.

        ``cell_state`` is the 2-D authoritative ``S`` grid (decoded from the
        ``gis.cell_state`` sensor). ``buses`` optionally restricts evaluation to
        a subset (e.g. only the buses this agent actually controls); when
        ``None`` all co-registered buses are considered.
        """
        S = np.asarray(cell_state).reshape(self.nrows, self.ncols)
        fire = (S == BURNING) | (S == BURNED_OUT)
        candidates = self.bus_cell if buses is None else {
            int(b): self.bus_cell[int(b)]
            for b in buses if int(b) in self.bus_cell
        }
        for b, (r, c) in candidates.items():
            if fire[r, c]:
                self._shed.add(int(b))
        return set(self._shed)

    def reset(self) -> None:
        self._shed.clear()


def coerce_to_actuator_space(value, actuator) -> np.ndarray:
    """Cast ``value`` to exactly match an actuator's space dtype and shape.

    palaestrAI's space containment check (``_space_contains``) wraps the written
    value into ``np.ndarray`` inferring its dtype dynamically. A value whose
    dtype is python ``float`` / ``np.float64`` is **not** contained in a
    ``Box(..., dtype=np.float32)`` -- it raises ``OutOfActionSpaceError``. We
    therefore coerce every actuator write to the actuator's own
    ``space.dtype`` and ``space.shape`` so containment always holds.

    For scalar Box spaces (``shape == ()``) this returns a 0-d array of the
    right dtype; for vector spaces it reshapes/broadcasts to the space shape.
    """
    space = getattr(actuator, "space", None)
    dtype = getattr(space, "dtype", np.float32)
    shape = getattr(space, "shape", None)
    arr = np.asarray(value, dtype=dtype)
    if shape is not None:
        if arr.shape == ():
            arr = np.full(shape, arr, dtype=dtype)
        else:
            arr = arr.reshape(shape).astype(dtype, copy=False)
    return arr
