"""Tactic primitives -- map a resource budget to ``(row, col, state, layer)``.

Each primitive is numpy-only and deterministic given its inputs. They reuse the
v0.3 selectors in :mod:`palaestrai_socal.agents.firefighter_core` so the
indirect-line tactic is byte-identical to the shipped retardant line.

Every primitive returns a list of ``(row, col, state, layer)`` mutation tuples
(the same shape the muscle encodes onto ``gis.cell_mutations``) and degrades to
``[]`` when its budget is ``<= 0`` -- preserving the no-op identity (a grounded
or crewless resource emits zero edits, so the fire CA is unchanged).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents.firefighter_core import (
    BURNING,
    UNBURNED,
    downwind_offset,
    fire_head,
    select_retardant_line,
)

Mutation = Tuple[int, int, int, int]


def indirect_line(
    state: np.ndarray,
    fuel: Optional[np.ndarray],
    wind_dir_deg: float,
    budget: int,
    line_state: int = spaces.SUPPRESSED,
    layer: int = spaces.LAYER_SUPPRESSION,
) -> List[Mutation]:
    """Build a contiguous line one cell downwind of the fire head.

    This is the v0.3 retardant tactic. With ``line_state=SUPPRESSED`` and
    ``layer=LAYER_SUPPRESSION`` (the defaults) the cell list is exactly
    :func:`firefighter_core.select_retardant_line`, mapped to mutation tuples in
    the same order -- the basis of the v0.3 identity regression. Ground crews /
    dozers reuse the same geometry with ``line_state=CONTAINED``.
    """
    cells = select_retardant_line(state, fuel, wind_dir_deg, budget)
    return [(r, c, line_state, layer) for (r, c) in cells]


def direct_attack(
    state: np.ndarray,
    budget: int,
    wind_dir_deg: float = 0.0,
    layer: int = spaces.LAYER_SUPPRESSION,
) -> List[Mutation]:
    """Drop water/foam directly ON burning cells (extinguish -> wetline).

    Targets the active fire **head** first (the downwind-advancing front, where
    knockdown does the most good), then any remaining burning cells, both in
    deterministic row/col order. Each becomes ``SUPPRESSED`` (a short-hold
    wetline that ages like retardant). Returns ``[]`` for a zero budget or no
    fire.
    """
    if budget <= 0:
        return []
    S = np.asarray(state)
    drow, dcol = downwind_offset(wind_dir_deg)
    head = set(fire_head(S, drow, dcol))
    ordered_head = sorted(head)
    rest = sorted(
        (int(r), int(c)) for (r, c) in np.argwhere(S == BURNING)
        if (int(r), int(c)) not in head
    )
    targets = ordered_head + rest
    return [(r, c, spaces.SUPPRESSED, layer) for (r, c) in targets[:budget]]


def containment_line(
    state: np.ndarray,
    fuel: Optional[np.ndarray],
    wind_dir_deg: float,
    budget: int,
    layer: int = spaces.LAYER_SUPPRESSION,
) -> List[Mutation]:
    """Build a ground containment line (handline / dozer) -> ``CONTAINED``.

    Same ahead-of-head geometry as :func:`indirect_line`, but the cells are
    ``CONTAINED`` (permanent within the episode, outranking SUPPRESSED).
    """
    return indirect_line(state, fuel, wind_dir_deg, budget,
                         line_state=spaces.CONTAINED, layer=layer)


# handline / dozer_line are the same primitive under different doctrine labels;
# the productivity difference lives entirely in the resource capacity.
handline = containment_line
dozer_line = containment_line


def burnout(
    state: np.ndarray,
    line_cells: Sequence[Tuple[int, int]],
    wind_dir_deg: float,
    budget: int,
) -> List[Mutation]:
    """Intentionally ignite UNBURNED fuel between a line and the fire.

    For each containment/retardant cell in ``line_cells`` the candidate is the
    UNBURNED cell one step *upwind* (toward the fire), which is set ``BURNING``
    (a controlled backfire that removes fuel ahead of the main front). Returns
    ``[]`` for a zero budget or when there is nothing to ignite.
    """
    if budget <= 0 or len(line_cells) == 0:
        return []
    S = np.asarray(state)
    nr, nc = S.shape
    drow, dcol = downwind_offset(wind_dir_deg)
    # upwind = toward the fire = opposite the downwind advance direction.
    seen = set()
    out: List[Mutation] = []
    for (r, c) in sorted((int(a), int(b)) for (a, b) in line_cells):
        ur, uc = r - drow, c - dcol
        if 0 <= ur < nr and 0 <= uc < nc and S[ur, uc] == UNBURNED:
            key = (ur, uc)
            if key not in seen:
                seen.add(key)
                out.append((ur, uc, BURNING, spaces.LAYER_FIRE))
                if len(out) >= budget:
                    break
    return out


def point_protect(
    state: np.ndarray,
    value_raster: Optional[np.ndarray],
    budget: int,
    layer: int = spaces.LAYER_SUPPRESSION,
) -> List[Mutation]:
    """Harden the highest-value UNBURNED grid-asset cells (-> ``CONTAINED``).

    ``value_raster`` holds each cell's worth (served MW lost if the asset there
    trips; see :func:`planner.value_raster_from_buses`). The highest-value
    not-yet-burned cells are made non-ignitable, directly preventing the
    fire->damage->load-shed trip the DamageMapper would otherwise apply. Ties
    break by row/col for determinism. Returns ``[]`` with no budget or no map.
    """
    if budget <= 0 or value_raster is None:
        return []
    S = np.asarray(state)
    V = np.asarray(value_raster, dtype=float)
    if V.shape != S.shape:
        return []
    rs, cs = np.where((V > 0.0) & (S == UNBURNED))
    if rs.size == 0:
        return []
    order = sorted(
        zip(rs.tolist(), cs.tolist()),
        key=lambda rc: (-float(V[rc[0], rc[1]]), rc[0], rc[1]),
    )
    return [(r, c, spaces.CONTAINED, layer) for (r, c) in order[:budget]]
