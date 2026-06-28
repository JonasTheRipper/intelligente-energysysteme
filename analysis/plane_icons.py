"""Derive aero-tanker plane-icon positions from the stored SUPPRESSED grid.

Firefighter telemetry (planes_in_service, drops_this_step, ...) travels on the
palaestrAI MUSCLE return channel, NOT as an environment sensor, so it is *not*
in ``world_states`` and the store-only timelapse cannot read it. What IS stored
is the authoritative ``gis.cell_state`` grid, including the firefighter's
``SUPPRESSED`` (code 3) retardant cells.

So we visualise the planes indirectly: the cells that became SUPPRESSED *this*
step (the diff of the SUPPRESSED set between consecutive frames) are the
retardant the fleet just laid, and a plane is drawn at the leading edge of that
freshly-laid line. This module is the pure, numpy-only geometry helper; the
matplotlib glyph/fade lives in :mod:`analysis.make_comparison_timelapse`.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def new_suppressed_cells(
    prev: np.ndarray, curr: np.ndarray
) -> np.ndarray:
    """Boolean grid of cells that became SUPPRESSED between ``prev`` and ``curr``.

    Both inputs are boolean (or 0/1) grids of identical shape marking the
    SUPPRESSED cells in the previous and current frame.
    """
    p = np.asarray(prev, dtype=bool)
    c = np.asarray(curr, dtype=bool)
    if p.shape != c.shape:
        raise ValueError(f"shape mismatch: prev {p.shape} vs curr {c.shape}")
    return c & ~p


def plane_positions(
    prev_suppressed: np.ndarray,
    curr_suppressed: np.ndarray,
    n_planes: int,
) -> List[Tuple[int, int]]:
    """Up to ``n_planes`` ``(row, col)`` plane positions along the new line.

    Given the SUPPRESSED boolean grids of two consecutive frames and the fleet
    size, return the cells where plane icons should be drawn this step:

    * the *newly* SUPPRESSED cells are the retardant laid this step;
    * if there are none (grounded / no fire head / line ageing out), return ``[]``
      so the renderer shows no planes;
    * the new cells are ordered deterministically (by ``(row, col)``) and, when
      there are more new cells than planes, ``n_planes`` of them are sampled at
      evenly spaced positions along that ordering so the icons spread along the
      freshly-laid line rather than clustering.

    Deterministic: the same inputs always yield the same list, so it is unit
    testable without any simulation.
    """
    if n_planes <= 0:
        return []
    new = new_suppressed_cells(prev_suppressed, curr_suppressed)
    rows, cols = np.nonzero(new)
    if rows.size == 0:
        return []
    # deterministic order along the line
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    k = int(rows.size)
    n = min(int(n_planes), k)
    if n >= k:
        idx = np.arange(k)
    else:
        # evenly spaced sample indices across [0, k-1], inclusive of both ends.
        idx = np.linspace(0, k - 1, n).round().astype(int)
        idx = np.unique(idx)
    return [(int(rows[i]), int(cols[i])) for i in idx]
