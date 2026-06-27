"""Encoding helpers for the v0.2 two-environment palaestrAI design.

palaestrAI actuators/sensors are fixed-shape :class:`palaestrai.types.Box`
vectors, but the GIS world needs to exchange **variable-length** data with the
agents:

* the ``gis.cell_mutations`` actuator carries a list of cell edits
  ``(row, col, state_code, layer_code)`` whose length changes every step (the
  fire front grows); and
* the ``world_state`` dump persisted to the store must embed full 2-D cell-state
  / fuel grids as JSON-serialisable strings (no numpy arrays).

This module solves both with pure-numpy codecs so the unit tests need neither
palaestrai nor pandapower:

* :func:`encode_mutations` / :func:`decode_mutations` -- a fixed-capacity
  "padded set": a flat float vector ``[N, r0,c0,s0,l0, r1,c1,s1,l1, ...]`` of
  capacity ``CAP`` mutations, round-trip exact for ``N <= CAP``.
* :func:`encode_grid` / :func:`decode_grid` -- a compact ``zlib+base64`` codec
  for an ``int8`` grid, returning a JSON-serialisable dict (lists + str only).

The :class:`palaestrai.types.Box` factories are imported lazily so importing
this module stays numpy-only (the CI unit stage has no palaestrai).
"""

from __future__ import annotations

import base64
import zlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

# -- cell state codes (mirror wildfire_cma.cma; SUPPRESSED/FLOODED reserved) --
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2
SUPPRESSED = 3        # reserved for a future FirefighterAgent (exposed, unused)
FLOODED = 4           # reserved for a future flood hazard (exposed, unused)

# -- mutation layers (which hazard wrote the cell) ---------------------------
LAYER_FIRE = 0
LAYER_SUPPRESSION = 1
LAYER_FLOOD = 2

VALID_STATES = (UNBURNED, BURNING, BURNED_OUT, SUPPRESSED, FLOODED)
VALID_LAYERS = (LAYER_FIRE, LAYER_SUPPRESSION, LAYER_FLOOD)

# Padded-set capacity: one Santa-Ana hour on the 600x760 raster ignites far
# fewer than this many *new* cells, so CAP=8192 never truncates in practice.
CAP = 8192
WIDTH = 4   # (row, col, state, layer)


# --------------------------------------------------------------------------
# padded-set codec for the gis.cell_mutations actuator
# --------------------------------------------------------------------------
def mutation_vector_size(cap: int = CAP) -> int:
    """Length of the flat float vector that holds up to ``cap`` mutations."""
    return 1 + cap * WIDTH


def encode_mutations(
    muts: Sequence[Tuple[int, int, int, int]], cap: int = CAP
) -> np.ndarray:
    """Pack a list of ``(row, col, state, layer)`` into a fixed-size vector.

    Layout: ``[N, r0,c0,s0,l0, r1,c1,s1,l1, ...]`` zero-padded to ``cap``
    entries. Extra mutations beyond ``cap`` are dropped (and ``N`` is clamped).
    """
    arr = np.zeros(mutation_vector_size(cap), dtype=np.float64)
    n = min(len(muts), cap)
    arr[0] = float(n)
    for i in range(n):
        r, c, s, lyr = muts[i]
        base = 1 + i * WIDTH
        arr[base + 0] = float(r)
        arr[base + 1] = float(c)
        arr[base + 2] = float(s)
        arr[base + 3] = float(lyr)
    return arr


def decode_mutations(
    arr: Sequence[float], cap: int = CAP
) -> List[Tuple[int, int, int, int]]:
    """Inverse of :func:`encode_mutations` -> list of int 4-tuples."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return []
    n = int(round(float(a[0])))
    n = max(0, min(n, cap, (a.size - 1) // WIDTH))
    out: List[Tuple[int, int, int, int]] = []
    for i in range(n):
        base = 1 + i * WIDTH
        out.append((
            int(round(float(a[base + 0]))),
            int(round(float(a[base + 1]))),
            int(round(float(a[base + 2]))),
            int(round(float(a[base + 3]))),
        ))
    return out


# --------------------------------------------------------------------------
# compact JSON-serialisable codec for an int8 cell grid (world_state dump)
# --------------------------------------------------------------------------
def encode_grid(grid: np.ndarray, dtype: str = "int8") -> Dict[str, object]:
    """Compress a grid to a JSON-serialisable dict (zlib + base64).

    Returns ``{"shape": [...], "dtype": "<dtype>", "codec": "zlib+b64",
    "data": "<base64>"}`` -- only lists and a str, so it round-trips through
    ``json.dumps`` straight into the store ``world_states.state_dump`` column.
    ``dtype`` defaults to ``int8`` (cell grids); pass ``"float32"`` for a DEM.
    """
    dt = np.dtype(dtype)
    g = np.ascontiguousarray(grid, dtype=dt)
    comp = zlib.compress(g.tobytes(), 6)
    return {
        "shape": [int(x) for x in g.shape],
        "dtype": str(dt),
        "codec": "zlib+b64",
        "data": base64.b64encode(comp).decode("ascii"),
    }


def decode_grid(obj: Dict[str, object]) -> np.ndarray:
    """Inverse of :func:`encode_grid` -> a fresh writable int8 ndarray."""
    raw = zlib.decompress(base64.b64decode(obj["data"]))  # type: ignore[arg-type]
    dt = np.dtype(str(obj.get("dtype", "int8")))
    arr = np.frombuffer(raw, dtype=dt).reshape(tuple(obj["shape"]))  # type: ignore[arg-type]
    return arr.copy()


# --------------------------------------------------------------------------
# lazy palaestrai.types.Box factories (kept out of the numpy-only import path)
# --------------------------------------------------------------------------
def mutation_space(cap: int = CAP):
    """Box for the ``gis.cell_mutations`` actuator (lazy palaestrai import)."""
    from palaestrai.types import Box

    size = mutation_vector_size(cap)
    return Box(low=-1.0, high=1.0e9, shape=(size,), dtype=np.float64)


def scalar_box(low: float, high: float):
    """1-D Box for a single scalar sensor/actuator (lazy palaestrai import)."""
    from palaestrai.types import Box

    return Box(low=float(low), high=float(high), shape=(1,), dtype=np.float64)


def vector_box(low: float, high: float, n: int):
    """1-D Box of length ``n`` (lazy palaestrai import)."""
    from palaestrai.types import Box

    return Box(low=float(low), high=float(high), shape=(int(n),), dtype=np.float64)
