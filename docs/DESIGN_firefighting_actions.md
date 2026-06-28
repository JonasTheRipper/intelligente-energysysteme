# Design & context — full-blown firefighting actions

**Status:** design / context (not yet implemented)
**Scope:** an extension of the v0.3 aero-tanker responder (`FirefighterMuscle`)
into a *multi-resource, multi-doctrine* firefighting agent.
**Audience:** anyone extending `palaestrai_socal/agents/firefighter_*`.
**Author:** Eric MSP Veith · License: GPL-3.0-or-later

---

## 0. TL;DR

v0.3 ships exactly **one** firefighting action: a fleet of `n_planes` Large Air
Tankers laying a long-term retardant line (`SUPPRESSED`) ahead of the fire head.
That is a deliberately narrow slice of real wildfire suppression. This document
lays out how to grow it into "full-blown firefighting" — ground crews, dozer
lines, direct attack, water/foam vs. long-term retardant, point protection of the
grid, staging/turnaround logistics, and an incident-command resource allocator —
**without** changing the architectural contract that makes v0.3 clean:

> Every hazard and responder is a palaestrAI agent that writes the **same**
> `gis.cell_mutations` actuator; the GIS environment is a *dumb applier* that
> arbitrates competing edits by a fixed state priority and ages temporary edits
> back to `UNBURNED`. Decision logic is numpy-only and lives in a `*_core.py`
> module so it is unit-testable without palaestrAI.

The extension is therefore mostly **new cell semantics, new constants, and a
richer action-selection core** — not new plumbing.

---

## 1. What v0.3 actually models (the baseline being extended)

| Aspect | v0.3 implementation | File |
|--------|---------------------|------|
| Resource | `n_planes` identical Type-1 LATs | `firefighter_agent.py` |
| Action | lay a contiguous `SUPPRESSED` line one cell downwind of the fire head | `firefighter_core.select_retardant_line` |
| Capacity | `n_planes → drops/step → line-km → cells`, scaled by `wind_efficiency` | `firefighter_core.retardant_budget` |
| Constraint | hard **grounding** at `wind ≥ GROUND_WIND_MS` (18 m/s); linear degrade from 13 m/s | `firefighter_core.wind_efficiency` |
| Persistence | retardant ages back to `UNBURNED` after `SUPPRESS_PERSIST_STEPS` (12) | `firefighter_core.age_suppressed` (env owns the timer) |
| Targeting | downwind fire head, greedy toward high `fuel_class`, contiguous | `firefighter_core.fire_head` / `select_retardant_line` |
| Output channel | `(row, col, SUPPRESSED, LAYER_SUPPRESSION)` on `gis.cell_mutations` | `spaces.encode_mutations` |
| Telemetry | dict on the muscle return channel (`planes_in_service`, `drops_this_step`, `retardant_cells`, `grounded`, `line_km_cumulative`) — **not** stored | `firefighter_agent.propose_actions` |
| Knob | one: `n_planes` (everything else is a documented constant) | `firefighter_core` |

**The two contracts that must be preserved by any extension:**

1. **Mutation arbitration** (`spaces.arbitrate_mutations`, env-side): every cell
   is resolved by `BURNED_OUT (terminal) > SUPPRESSED > FLOODED > BURNING >
   UNBURNED`, so the result is **independent of agent turn order**. Any new
   suppression state must be given an explicit, unambiguous priority here.
2. **No-op identity:** with zero suppression edits the fire CA is bit-for-bit
   identical to v0.2 (`tests/test_suppression_block.py`). Every new action must
   degrade to a true no-op when its budget is 0 (e.g. grounded, no crews,
   no water).

---

## 2. Action taxonomy — what "full-blown firefighting" means

Real wildland firefighting is a portfolio of *tactics* applied by *resources*
under *doctrine*. We map each onto the CA + grid model below.

### 2.1 Suppression tactics (cell-state effects)

| Tactic | Real meaning | CA effect | Proposed state / layer |
|--------|--------------|-----------|------------------------|
| **Long-term retardant line** (v0.3) | Phos-Chek dropped ahead of the front; works for hours | non-ignitable cell that ages out | `SUPPRESSED` / `LAYER_SUPPRESSION` (exists) |
| **Direct attack (water/foam)** | water/foam dropped *on* burning cells | extinguish: `BURNING → SUPPRESSED` (short hold) or `→ BURNED_OUT` if already consumed | reuse `SUPPRESSED` with a **short** persist timer, or a new `WETLINE` substate |
| **Dozer / handline (containment line)** | scraped mineral-soil break, no fuel | permanent-ish non-ignitable line | `SUPPRESSED` with a **long/no** age-out, or a new `CONTAINED` state |
| **Burnout / backfire** | intentionally ignite fuel between line and fire to remove it | `UNBURNED → BURNING` (controlled), converging on the line | a *responder-issued* `BURNING` edit (already expressible) gated by doctrine |
| **Structure/point protection** | wrap/foam an asset (substation, line tower) | protect a specific grid asset cell from de-energisation | couples to the **grid** env, not just GIS (see §5) |
| **Mop-up** | extinguish smouldering edges to prevent re-ignition | `BURNING → BURNED_OUT` at the cold trailing edge | low-priority action, late-phase |

> **Design choice — extend states vs. extend layers.** The clean path is to keep
> the small, arbitrated `cell_state` enum and encode *durability* via the
> per-cell **age/persist timer** rather than minting many new states. So
> "retardant" vs. "dozer line" differ by their `persist_steps`, not by a new
> code. Reserve a genuinely new state only when arbitration priority must differ
> (e.g. a `CONTAINED` line that should outrank an in-progress `BURNING` edit
> *and* survive longer than retardant). `spaces` already reserves `FLOODED=4`
> and `LAYER_FLOOD=2` as the template for adding a code cleanly.

### 2.2 Resources (who performs the tactic)

| Resource | Real | Knob(s) | Speed / reach | Wind/terrain constraint |
|----------|------|---------|---------------|-------------------------|
| Air tanker (LAT) | v0.3 | `n_planes` | fast, anywhere reachable | grounded ≥ 18 m/s |
| Helicopter (Type-1/2) | bucket water/foam | `n_helos` | slower, smaller drop, can dip nearby water | grounded later than LATs; night-capable subset |
| Hand crew (Type-1/IHC) | handline, burnout, mop-up | `n_crews` | slow (line-chains/hour), terrain-limited | **not** wind-grounded; slope-limited |
| Dozer | wide containment line | `n_dozers` | medium, fuel/slope-limited | slope cutoff, not buildable in rock/urban |
| Engine | structure protection, mobile water | `n_engines` | road-access cells only | road network mask |

Each resource is a **set of constants** + a capacity function analogous to
`retardant_budget`. The agent owns a *fleet mix*, not a single `n_planes`.

### 2.3 Doctrine (how tactics are chosen)

- **Direct vs. indirect attack:** direct = act on the fire edge (water on
  `BURNING`); indirect = build line ahead (v0.3). Choice depends on fireline
  intensity (a function of `fuel_class`, wind, slope).
- **Anchor-and-flank:** start a line at a defensible anchor (road, water, burned
  ground) and work along a flank, rather than the current "wherever the head is"
  greedy fill.
- **Triage / values-at-risk:** prioritise lines that protect high-value cells —
  in this testbed, **grid assets** (substations, transmission towers) and the
  load they feed. This is the bridge to the power-grid objective and the most
  novel part for an energy-informatics audience.

---

## 3. State & layer model changes

Keep `cell_state` minimal and arbitrated; push richness into **side layers and
timers**, which the env already owns.

### 3.1 Reuse what exists
- `SUPPRESSED` (3) / `LAYER_SUPPRESSION` (1) — all retardant/wetline/handline
  share this *visible* state.
- Per-cell `suppress_age` timer (env-owned, `age_suppressed`) — already the
  mechanism for durability. Make `persist_steps` **per-tactic** by storing, per
  suppressed cell, *which tactic* laid it.

### 3.2 Add (only where arbitration demands)
- `CONTAINED = 5` (optional new `cell_state`): a completed containment line that
  outranks `BURNING` and does **not** age out within the episode. Add to
  `VALID_STATES`, `STATE_PRIORITY` (between `SUPPRESSED` and `BURNED_OUT`, or
  above `SUPPRESSED` — decide and document), and to the CA's non-ignitable guard
  in `wildfire_cma/cma.py` (mirror the existing `SUPPRESSED` guard at lines
  ~307-313).
- A **tactic-id side array** (`LAYER_SUPPRESSION` payload or a parallel
  `suppress_kind` raster) so the env can choose the right `persist_steps` and so
  analysis can colour retardant vs. dozer line vs. wetline differently.
- A **resource-position channel** for honest plane/crew icons (see §6) — v0.3
  reconstructs plane positions from the SUPPRESSED diff because telemetry isn't
  stored; a richer agent should optionally emit positions as a stored sensor.

### 3.3 Invariants to keep
- `BURNED_OUT` stays terminal and monotonic.
- Arbitration stays total and tie-free (distinct priority integers).
- Zero-budget ⇒ zero edits ⇒ v0.2 identity.

---

## 4. Agent / core structure (the extension shape)

v0.3 already isolates plumbing (`firefighter_agent.py`, palaestrAI-facing) from
logic (`firefighter_core.py`, numpy-only). Preserve that split.

```
agents/
  firefighter_agent.py     # palaestrAI Muscle: sensors→core→cell_mutations (KEEP)
  firefighter_core.py      # tanker capacity + line selection (v0.3) (KEEP)
  firefighting/            # NEW package: one module per resource + a planner
    resources.py           # dataclasses: TankerFleet, HeloFleet, HandCrews,
                           #   Dozers, Engines — each .capacity(state, wind,
                           #   slope, roads, step_min, cell_m) -> budget
    tactics.py             # tactic primitives reusing v0.3 selectors:
                           #   indirect_line() (≈ select_retardant_line),
                           #   direct_attack(), handline(), burnout(),
                           #   point_protect()
    doctrine.py            # anchor-and-flank, triage-by-value, direct-vs-indirect
    planner.py             # IncidentCommand: allocate the fleet mix across
                           #   tactics this step under total budget + constraints
```

- **`FirefighterMuscle` stays the only palaestrAI touch-point.** It gains params
  beyond `n_planes` (`n_helos`, `n_crews`, `n_dozers`, `n_engines`, plus an
  optional `doctrine` string and `protect_assets` flag) and delegates the
  per-step decision to `planner.IncidentCommand.propose(...)`, which returns the
  same `List[(row, col, state, layer)]` it already encodes.
- **Backward compatibility:** with only `n_planes` set and `doctrine="indirect"`,
  the planner must reproduce v0.3 exactly (regression-test against the current
  `select_retardant_line` output).
- **New sensors the core may consume (all optional, with param fallbacks like
  v0.3's `_ensure_geo`):** `gis.slope` (or derive from `gis.dem`),
  `gis.road_mask` / `gis.water_mask`, and — for triage — a map of grid-asset
  cells (see §5). All must have graceful no-sensor fallbacks so the agent never
  hard-fails on a minimal env.

---

## 5. Coupling firefighting to the power grid (the testbed's point)

This is what makes "full-blown firefighting" interesting *here* rather than as a
generic fire model. Two coupling directions:

1. **Triage objective (fire → grid value):** the planner should prefer lines that
   keep the fire away from **grid-critical cells**. Build a `value_raster` where
   each cell's worth ∝ the served load it would cost if the asset there trips.
   This requires mapping grid elements (substations, transmission corridors) to
   GIS cells — the inverse of what `DamageMapperAgent` already does when it
   de-energises assets the fire engulfs (`agents/damage_core.py`). Reuse that
   asset→cell registration.
2. **Point protection (firefighting → grid):** a `point_protect` tactic spends
   engine/crew budget to make a specific grid-asset cell non-ignitable
   (`SUPPRESSED`/`CONTAINED`), directly preventing the load-shed trip the
   `DamageMapperAgent` would otherwise apply. The grid env needs no new
   actuator — protection is a GIS edit that stops the asset from ever being
   engulfed, so the existing fire→damage→load-shed chain simply never fires for
   that asset.

**KPI consequence:** with grid-aware triage, the headline metric shifts from
"acres saved" to **"customer-minutes (SAIDI) / MW preserved per resource-hour"** —
a far more decision-relevant number, and one the v0.3 grid-metrics renderer
already plots (SAIDI, served MW, intertie flow).

---

## 6. Telemetry, analysis & visualization

- **Store richer telemetry properly.** v0.3's telemetry dict rides the muscle
  return channel and is *not* persisted (the known `dict + int` dummy-brain
  warning is a symptom of returning a dict there). For multi-resource work,
  expose the per-step resource state (positions, drops, line-km by tactic) as a
  **stored GIS sensor** so analysis reads it from the store like every other
  metric, and return `None`/`np.nan` (not a dict) on the brain channel to silence
  the warning. This also lets the timelapse draw **honest** plane/helo/crew icons
  instead of reconstructing them from the SUPPRESSED diff (`analysis/plane_icons.py`).
- **Per-tactic colouring.** Extend the comparison renderers
  (`make_comparison_timelapse.py`, `grid_metrics_report.py`, already N-phase) to
  colour retardant vs. dozer line vs. wetline using the tactic-id side array, and
  to add a resource-utilisation panel (drops, line-km, idle aircraft).
- **New phases for sweeps.** The N-phase machinery means a "fleet-mix sweep"
  (e.g. all-air vs. air+ground vs. ground-only vs. grid-triage-on) is just more
  phases in one experiment, rendered as extra rows — exactly how the 0/3/7
  tanker sweep already works.

---

## 7. Experiment design

Build on `experiment_eaton_local_ab.yml` (already N-phase, seed 47, fine grid).

Suggested phase sets (each one extra row in the timelapse, one extra line in the
metrics chart):

1. **Doctrine sweep (fixed resources):** baseline / indirect-only /
   direct+indirect / anchor-and-flank / grid-triage — isolates *doctrine* value.
2. **Resource-mix sweep (fixed doctrine):** air-only(3) / air(3)+ground /
   ground-only / air(7)+ground+engines — isolates *capacity* value.
3. **Wind-sensitivity:** rerun the best mix at 8 / 13 / 18 / 25 m/s to show the
   grounding cliff and where ground crews (not wind-grounded) dominate.

Keep identical seed/bounds/raster across phases so differences are causal, as in
v0.3.

---

## 8. Constraints, realism & explicit non-goals

- **Constants over parameters.** As in v0.3, expose only operational knobs
  (resource counts, doctrine, triage on/off). Every productivity/wind/slope
  constant is a documented value in the relevant `firefighting/*` module, sourced
  to real data (LAT/helo/crew productivity, dozer line rates, engine GPM).
- **Determinism.** All selection stays deterministic given inputs (sorted
  tie-breaks), so phases are reproducible and diff-able.
- **Scripted, not learned (for now).** Keep the dummy-brain + scripted-muscle
  pattern. The natural learning upgrade is to make the **IncidentCommand
  allocator** a real palaestrAI brain (state = fire+grid+resources, action =
  budget split across tactics, reward = MW/SAIDI preserved per resource-hour) —
  this is the clean RL hook and mirrors the documented Overseer-Adversary
  extension point in `docs/AGENTS.md`.
- **Non-goals:** physically detailed plume/spotting models, individual
  aircraft trajectory simulation, crew fatigue scheduling, and real-time dispatch
  optimisation are out of scope; the CA-cell abstraction is intentionally coarse.

---

## 9. Implementation checklist (incremental, each shippable)

1. **Refactor, no behaviour change:** introduce `firefighting/` with
   `TankerFleet` + `indirect_line()` wrapping today's `firefighter_core`; route
   `FirefighterMuscle` through `IncidentCommand`; prove v0.3 identity by test.
2. **Fix telemetry channel:** store resource state as a GIS sensor; return
   non-dict on the brain channel (closes the `dict + int` warning).
3. **Add direct attack** (water on `BURNING`, short persist) + helo resource;
   add a wind regime where helos outlast tankers.
4. **Add ground line** (handline/dozer; not wind-grounded; slope/road masks) and,
   if needed, the `CONTAINED` state with arbitration + CA guard + tests.
5. **Grid triage + point protection:** build `value_raster` from the
   DamageMapper asset→cell map; add `point_protect`; switch headline KPI to
   MW/SAIDI-preserved-per-resource-hour.
6. **Renderers:** per-tactic colours, honest resource icons, utilisation panel.
7. **Experiments:** doctrine, resource-mix, and wind-sensitivity phase sweeps;
   write results into `CHANGELOG.md` / a `v0.4` release note.
8. **(Optional) Learning allocator:** promote `IncidentCommand` to a palaestrAI
   brain.

---

## 10. References in this repo

- `palaestrai_socal/agents/firefighter_agent.py` — the muscle to extend.
- `palaestrai_socal/agents/firefighter_core.py` — capacity + line-selection logic.
- `palaestrai_socal/spaces.py` — `cell_state` codes, layers, `STATE_PRIORITY`,
  `arbitrate_mutations`, `encode_mutations` (the `FLOODED`/`LAYER_FLOOD` reserve
  is the template for adding a new code).
- `wildfire_cma/cma.py` — the fire CA and its non-ignitable `SUPPRESSED` guard
  (where a new `CONTAINED` guard would go).
- `palaestrai_socal/agents/damage_core.py` — asset→cell mapping + load-shed trip
  (reuse for grid triage / point protection).
- `docs/AGENTS.md` — multi-agent contract, turn order, arbitration, and the
  Overseer-Adversary extension point (template for the learning allocator).
- `analysis/make_comparison_timelapse.py`, `analysis/grid_metrics_report.py`,
  `analysis/plane_icons.py` — N-phase renderers to extend per-tactic.
- `docs/v0.3_RELEASE.md`, `CHANGELOG.md` — v0.3 baseline this extends.
