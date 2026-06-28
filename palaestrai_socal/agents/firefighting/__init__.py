"""Full-blown firefighting extension of the v0.3 aero-tanker responder (v0.4).

This package grows the single-resource v0.3 firefighter (a fleet of ``n_planes``
Large Air Tankers laying one retardant line) into a *multi-resource,
multi-doctrine* incident, **without** changing the architectural contract that
keeps v0.3 clean (see ``docs/DESIGN_firefighting_actions.md``):

* every responder still writes the SAME ``gis.cell_mutations`` actuator;
* the GIS env stays a dumb applier arbitrating edits by fixed state priority;
* all decision logic is numpy-only and unit-testable without palaestrAI.

Modules
-------
* :mod:`.resources` -- one dataclass per resource (tankers, helos, hand crews,
  dozers, engines), each with a ``.capacity(...)`` cell budget function.
* :mod:`.tactics`   -- tactic primitives (indirect line, direct attack, ground
  containment line, burnout, point protection) reusing the v0.3 selectors.
* :mod:`.doctrine`  -- direct-vs-indirect, anchor-and-flank, triage-by-value.
* :mod:`.planner`   -- :class:`IncidentCommand`, the deterministic allocator the
  :class:`~palaestrai_socal.agents.firefighter_agent.FirefighterMuscle` delegates
  the per-step decision to. With tankers-only + indirect doctrine it reproduces
  v0.3's retardant line exactly (regression-tested).
"""
from __future__ import annotations

from .resources import (  # noqa: F401
    Dozers,
    Engines,
    HandCrews,
    HeloFleet,
    TankerFleet,
    build_resources,
)
from .planner import IncidentCommand  # noqa: F401

__all__ = [
    "TankerFleet",
    "HeloFleet",
    "HandCrews",
    "Dozers",
    "Engines",
    "build_resources",
    "IncidentCommand",
]
