"""Doctrine helpers -- how tactics are chosen (numpy-only, deterministic).

Three doctrine elements from DESIGN §2.3 are expressed as pure functions the
planner calls:

* **direct vs. indirect attack** -- chosen by a fireline-intensity proxy
  (fuel x wind x slope). Low intensity -> direct attack on the edge is safe;
  high intensity -> build indirect line ahead of the front.
* **anchor-and-flank** -- order candidate line cells from a defensible anchor
  (burned ground / existing line / grid edge) rather than scattering them.
* **triage-by-value** -- prefer protecting high-value (grid-critical) cells.

These are advisory orderings/flags; the planner remains the single allocator.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Fireline-intensity proxy threshold above which direct attack on the edge is
# deemed unsafe and the planner builds indirect line instead. The proxy is a
# dimensionless fuel x wind x slope product (see :func:`fireline_intensity`);
# the cutoff is tuned so an 8 m/s chaparral run stays "direct-capable" while a
# Santa-Ana 25 m/s run forces indirect. NOT a runtime parameter.
DIRECT_ATTACK_MAX_INTENSITY = 600.0


def fireline_intensity(
    fuel_mean: float,
    wind_speed: float,
    slope_deg: float = 0.0,
) -> float:
    """A coarse, monotone fireline-intensity proxy (higher = more intense).

    ``fuel_mean`` is the mean fuel class over the active front (richer fuel =>
    more intense), scaled by a wind factor (quadratic-ish in wind) and a slope
    factor. Deterministic and unit-free; used only for the direct/indirect
    threshold, so absolute calibration is unimportant -- ordering is.
    """
    f = max(float(fuel_mean), 0.0)
    wind_factor = 1.0 + (max(float(wind_speed), 0.0) / 5.0) ** 2
    slope_factor = 1.0 + abs(float(slope_deg)) / 30.0
    return f * wind_factor * slope_factor


def choose_attack(
    fuel_mean: float,
    wind_speed: float,
    slope_deg: float = 0.0,
    requested: Optional[str] = None,
) -> str:
    """Return ``"direct"`` or ``"indirect"``.

    If ``requested`` pins a doctrine (``"direct"`` / ``"indirect"``) it wins;
    otherwise the fireline-intensity proxy decides (direct when below
    :data:`DIRECT_ATTACK_MAX_INTENSITY`).
    """
    if requested in ("direct", "indirect"):
        return requested
    intensity = fireline_intensity(fuel_mean, wind_speed, slope_deg)
    return "direct" if intensity < DIRECT_ATTACK_MAX_INTENSITY else "indirect"


def is_anchored(
    state: np.ndarray,
    cell: Tuple[int, int],
    anchor_states: Sequence[int],
) -> bool:
    """True if ``cell`` touches a defensible anchor (8-neighbour).

    An anchor is any neighbouring cell already in one of ``anchor_states``
    (e.g. BURNED_OUT ground, an existing SUPPRESSED/CONTAINED line, or the grid
    edge). Used to order anchor-and-flank line construction.
    """
    S = np.asarray(state)
    nr, nc = S.shape
    r, c = int(cell[0]), int(cell[1])
    if r == 0 or c == 0 or r == nr - 1 or c == nc - 1:
        return True                       # grid edge is a natural anchor
    anchors = set(int(a) for a in anchor_states)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < nr and 0 <= cc < nc and int(S[rr, cc]) in anchors:
                return True
    return False


def anchor_and_flank_order(
    state: np.ndarray,
    cells: Sequence[Tuple[int, int]],
    anchor_states: Sequence[int],
) -> List[Tuple[int, int]]:
    """Order ``cells`` so anchored cells come first, then by row/col.

    Deterministic: a stable key of ``(not anchored, row, col)``.
    """
    S = np.asarray(state)
    return sorted(
        ((int(r), int(c)) for (r, c) in cells),
        key=lambda rc: (0 if is_anchored(S, rc, anchor_states) else 1,
                        rc[0], rc[1]),
    )
