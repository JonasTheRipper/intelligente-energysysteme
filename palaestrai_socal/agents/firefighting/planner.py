"""IncidentCommand -- the deterministic fleet-mix allocator (v0.4).

The :class:`~palaestrai_socal.agents.firefighter_agent.FirefighterMuscle`
delegates its per-step decision to :meth:`IncidentCommand.propose`, which spends
each resource's budget on a tactic under the active doctrine and merges the
results into a single ``(row, col, state, layer)`` list -- the same list the
muscle already encodes onto ``gis.cell_mutations``.

Two contracts are preserved (DESIGN §1, hard requirements 1-2):

* **v0.3 identity** -- a command with only a ``TankerFleet`` and ``"indirect"``
  doctrine returns exactly ``select_retardant_line`` cells as SUPPRESSED edits,
  in the same order (no other resource, no reordering).
* **No-op identity** -- when every resource's budget is 0 (grounded, no crews)
  the proposal is ``[]``, so the fire CA is bit-for-bit the v0.2 baseline.

All allocation is deterministic given the inputs (sorted tie-breaks).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents.firefighting import doctrine as _doc
from palaestrai_socal.agents.firefighting import tactics as _tac
from palaestrai_socal.agents.firefighting.resources import (
    Dozers,
    Engines,
    HandCrews,
    HeloFleet,
    TankerFleet,
)

Mutation = Tuple[int, int, int, int]

# anchor states for anchor-and-flank ordering: burned ground and existing line.
_ANCHOR_STATES = (spaces.BURNED_OUT, spaces.SUPPRESSED, spaces.CONTAINED)


def value_raster_from_buses(
    shape: Tuple[int, int],
    bus_cell: Dict[int, Tuple[int, int]],
    bus_value: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Build a per-cell value raster from a bus->cell registration.

    The inverse of :class:`damage_core.DamageMapperDriver`: where the damage
    mapper reads which buses a *fire cell* trips, this writes each bus's worth
    (served MW lost if it trips, default 1.0) into its raster cell, so the
    planner can triage lines toward grid-critical ground. Colliding buses sum.
    Returns a ``float`` array of ``shape`` (zeros where no asset).
    """
    nr, nc = int(shape[0]), int(shape[1])
    V = np.zeros((nr, nc), dtype=float)
    for b, rc in bus_cell.items():
        r, c = int(rc[0]), int(rc[1])
        if 0 <= r < nr and 0 <= c < nc:
            V[r, c] += float(bus_value[b]) if bus_value and b in bus_value else 1.0
    return V


@dataclass
class IncidentCommand:
    """Allocate a fleet mix across tactics each env step.

    Parameters
    ----------
    resources:
        ordered list of resource dataclasses (see :mod:`.resources`).
    doctrine:
        ``"auto"`` (default; intensity-driven), ``"direct"`` or ``"indirect"``.
    protect_assets:
        when True and a ``value_raster`` is supplied, engines (and otherwise
        idle ground crews) spend budget on point protection of grid assets.
    """

    resources: List[object] = field(default_factory=list)
    doctrine: str = "auto"
    protect_assets: bool = False

    # -- helpers -----------------------------------------------------------
    def _front_fuel_mean(self, state: np.ndarray,
                         fuel: Optional[np.ndarray]) -> float:
        if fuel is None:
            return 3.0                    # chaparral default if no fuel sensor
        S = np.asarray(state)
        F = np.asarray(fuel, dtype=float)
        burning = (S == spaces.BURNING)
        if not burning.any():
            return float(F.mean())
        return float(F[burning].mean())

    # Tanken muss belohnt werden, 0 bleibend muss bestraft werden 

    @staticmethod
    def _merge(groups: Sequence[Sequence[Mutation]]) -> List[Mutation]:
        """Merge tactic outputs, deduping cells by highest state priority.

        Iterates ``groups`` in order; the first occurrence of a cell fixes its
        position, later occurrences only upgrade the state if their priority is
        higher. With a single group this is an identity (preserves v0.3 order).
        """
        chosen: "OrderedDict[Tuple[int, int], Mutation]" = OrderedDict()
        for group in groups:
            for (r, c, st, lyr) in group:
                key = (int(r), int(c))
                cur = chosen.get(key)
                if cur is None:
                    chosen[key] = (int(r), int(c), int(st), int(lyr))
                elif spaces.STATE_PRIORITY[int(st)] > spaces.STATE_PRIORITY[cur[2]]:
                    chosen[key] = (int(r), int(c), int(st), int(lyr))
        return list(chosen.values())

    # -- the per-step allocation ------------------------------------------
    def propose(
        self,
        state: np.ndarray,
        fuel: Optional[np.ndarray],
        wind_speed: float,
        wind_dir_deg: float,
        slope_deg: float = 0.0,
        roads: Optional[np.ndarray] = None,
        value_raster: Optional[np.ndarray] = None,
        step_min: float = 60.0,
        cell_m: float = 50.0,
    ) -> List[Mutation]:
        """Return this step's ``(row, col, state, layer)`` edits."""
        S = np.asarray(state)
        attack = _doc.choose_attack(
            self._front_fuel_mean(S, fuel), wind_speed, slope_deg,
            requested=(self.doctrine if self.doctrine in ("direct", "indirect")
                       else None),
        )
        groups: List[List[Mutation]] = []

        for res in self.resources:
            budget = res.capacity(wind_speed, step_min, cell_m, slope_deg)
            #Dead
            if budget <= 0:
                continue

            if isinstance(res, TankerFleet):
                # tankers always lay indirect retardant line (the v0.3 tactic).
                groups.append(_tac.indirect_line(
                    S, fuel, wind_dir_deg, budget,
                    line_state=res.state, layer=res.layer))

            elif isinstance(res, HeloFleet):
                if attack == "direct":
                    groups.append(_tac.direct_attack(
                        S, budget, wind_dir_deg, layer=res.layer))
                else:
                    groups.append(_tac.indirect_line(
                        S, fuel, wind_dir_deg, budget,
                        line_state=res.state, layer=res.layer))

            elif isinstance(res, (HandCrews, Dozers)):
                groups.append(_tac.containment_line(
                    S, fuel, wind_dir_deg, budget, layer=res.layer))

            elif isinstance(res, Engines):
                if self.protect_assets and value_raster is not None:
                    groups.append(_tac.point_protect(
                        S, value_raster, budget, layer=res.layer))
                # else: no protectable target -> engines idle (no-op).

        return self._merge(groups)
