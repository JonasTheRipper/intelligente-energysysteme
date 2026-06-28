# v0.2 Agent Architecture — SoCal Multi-Hazard Testbed

v0.2 decomposes the v0.1 monolithic `SoCalWildfireEnvironment` (which embedded
the fire CA, the damage mapper, and the power flow in one `Environment`) into a
**two-environment / multi-agent** design that runs entirely under palaestrAI,
with the power grid computed by the **real MIDAS scenario stepped by mosaik**.

```
            ┌────────────────────────────┐         ┌────────────────────────────┐
            │  gis_world                  │         │  socal_grid                │
            │  GisWorldEnvironment        │         │  SocalMidasGridEnvironment │
            │  (passive spatial substrate)│         │  (REAL palaestrai_mosaik + │
            │  holds authoritative S grid │         │   MIDAS powergrid/mosaik)  │
            └────────────┬───────────────┘         └─────────────┬──────────────┘
        gis.* sensors    │   ▲ gis.cell_mutations   load p_mw     │  ▲ ...load-*.p_mw
                         ▼   │                       sensors       ▼  │  (shed=0)
   ┌─────────────────────────────┐   ┌─────────────────────────────┐
   │ wildfire (WildfireCmaMuscle)│   │ damage_mapper                │
   │  GUARDIAN fire CA; injects  │   │ (DamageMapperMuscle)         │
   │  ignition (agent param) and │   │ reads gis.cell_state, sheds  │
   │  writes BURNING cell edits  │   │ load on fire-affected buses  │
   └─────────────────────────────┘   └─────────────────────────────┘
```

## Environments

### `gis_world` — `palaestrai_socal.gis_world_env:GisWorldEnvironment`
The **passive spatial substrate**. It owns the authoritative per-cell hazard
grid `S` and the static raster (fuel, elevation, bounds). It does **not** step
any hazard dynamics — it only *applies* the `gis.cell_mutations` edits agents
write and re-publishes spatial telemetry. This keeps it hazard-agnostic: fire,
suppression, and (future) flood all write through the same mutation actuator
using distinct state codes and a `layer` field.

### `socal_grid` — `palaestrai_socal.midas_grid_env:SocalMidasGridEnvironment`
A thin subclass of `palaestrai_mosaik.MosaikEnvironment` that wires the MIDAS
`midas_palaestrai.descriptor:Descriptor`, building the real SoCal mosaik World.
Every powergrid element is auto-exposed as a palaestrAI sensor/actuator
(`Powergrid-0.0-<element>.<attr>`). The grid is stepped by mosaik, not by any
hand-rolled `pp.runpp`.

## Agents

All current agents are **scripted** (palaestrAI `DummyBrain` + `DummyObjective`;
only the `Muscle` is custom). They do not learn — they are hazard / consequence
operators. The numpy-only core of each lives beside its muscle so it is
unit-testable without palaestrai or pandapower.

| Agent | Muscle | Reads | Writes | Core (numpy-only) |
|-------|--------|-------|--------|-------------------|
| `wildfire` | `agents.wildfire_agent:WildfireCmaMuscle` | `gis.*` (shape, bounds, fuel, dem, cell_state, wind) | `gis.cell_mutations` (BURNING) | `agents.wildfire_core:WildfireDriver` |
| `firefighter` (v0.3) | `agents.firefighter_agent:FirefighterMuscle` | `gis.cell_state`, `gis.fuel_class` (+ optional `gis.wind_field`) | `gis.cell_mutations` (SUPPRESSED / LAYER_SUPPRESSION) | `agents.firefighter_core` |
| `damage_mapper` | `agents.damage_agent:DamageMapperMuscle` | `gis.grid_shape`, `gis.bounds`, `gis.cell_state` | `...load-<bus>-<idx>.p_mw` (→ 0) | `agents.damage_core:DamageMapperDriver` |

Turn order is `wildfire → firefighter → damage_mapper`: the firefighter reads the
same-step fire field and lays a retardant line ahead of the head, and the GIS env
resolves the two `gis.cell_mutations` writers by fixed priority (below), so the
outcome is independent of turn order.

### Ignition is an agent parameter, not an environment action
The wildfire `ignition_points` (a list of `[lon, lat]`) and `ignition_step` are
**`WildfireCmaMuscle` params** in the experiment YAML. The muscle converts
lon/lat → raster `(row, col)` and injects `BURNING` idempotently at/after
`ignition_step`. There is no `ignition_lon/lat` environment actuator and no
overseer action for ignition — ignition is part of the hazard agent's policy.

### Load-shed trip (v0.2 damage mechanism)
v0.1 tripped buses/lines via `in_service=False` on the pandapower net. Under the
real MIDAS/mosaik co-simulation the only writable grid control surface is the
powergrid simulator's **load** (and `sgen`) `p_mw` actuators — mosaik exposes no
`bus.in_service` / `line.in_service` actuator. So `DamageMapperAgent` realises
de-energisation as a **load-shed trip**: it sets the `p_mw` actuator of every
load on a fire-affected bus to `0`. This reproduces the served-load shortfall
KPI within the native MIDAS actuator set. (See `agents/damage_core.py`.)

## Firefighter agent (v0.3 — IMPLEMENTED)

v0.3 ships the first responder, `FirefighterMuscle`
(`agents/firefighter_agent.py`), a fleet of `n_planes` Large Air Tankers that
lay long-term retardant ahead of the fire head. It binds to exactly the surface
the v0.2 substrate reserved:

* **Sensors:** `gis.cell_state`, `gis.fuel_class` (both length `N=nrows*ncols`,
  so they satisfy palaestrAI's flat-equal-length per-agent memory); wind comes
  from a `wind_speed`/`wind_dir_deg` param fallback matching the env config (the
  same pattern the wildfire muscle uses), or the live `gis.wind_field` sensor
  when subscribed.
* **Actuator:** `gis.cell_mutations` — it writes `(row, col, SUPPRESSED,
  LAYER_SUPPRESSION)` edits (`palaestrai_socal.spaces` reserves `SUPPRESSED=3`,
  `LAYER_SUPPRESSION=1`).

**One operational knob, `n_planes`.** Every other quantity is a documented
constant in `agents/firefighter_core.py` (real aero-tanker data): productivity
(`DROPS_PER_PLANE_PER_HOUR`, `LINE_KM_PER_DROP`), the wind grounding/degrade
curve (`DEGRADE_WIND_MS=13`, `GROUND_WIND_MS=18`), and retardant-line lifetime
(`SUPPRESS_PERSIST_STEPS=12`). The pipeline is `n_planes + wind → retardant
budget (cells) → downwind fire head → a contiguous SUPPRESSED line ahead of the
head, preferring high fuel`. In wind `≥ GROUND_WIND_MS` the fleet is grounded,
the budget is 0, and **no mutations are emitted** — which is why the Eaton 25 m/s
high-wind run is unchanged vs v0.2.

**Firebreak semantics.** The fire CA never re-ignites a `SUPPRESSED` cell and
never spreads *from* one — see
`tests/test_suppression_block.py`. With zero SUPPRESSED cells the spread step is
bit-for-bit identical to v0.2 (proven against a no-guard reference transition in
the same test).

**Mutation arbitration (env-side).** Because both wildfire and firefighter write
`gis.cell_mutations` in one step, `gis_world_env` resolves every cell by fixed
priority via `spaces.arbitrate_mutations`:
`BURNED_OUT (terminal) > SUPPRESSED > FLOODED > BURNING > UNBURNED`. This makes a
retardant line laid this step hold against same-step spread and the result
independent of agent turn order (`tests/test_arbitration.py`). The env also owns
a per-cell `_suppress_age` timer that reverts a line to `UNBURNED` after
`SUPPRESS_PERSIST_STEPS` env steps (retardant breakdown), exactly as it owns the
fire's burn timer.

**Scenarios.** On the statewide 600×760 grid one cell is ~947 m, so a realistic
retardant line is sub-cell and the budget rounds to ~0; the Eaton high-wind run
(`experiment_eaton.yml`) keeps a firefighter block but is grounded. The
operationally meaningful demo is `experiment_eaton_local.yml` — a ~50 m fine grid
around the Eaton Canyon ignition at moderate (8 m/s) wind, where sweeping
`n_planes ∈ {0,1,3,5,7}` measurably reduces burned acres
(`analysis/firefighter_report.py`).

## Overseer-Adversary (interface-only, NOT implemented)

The GUARDIAN framing has an Overseer-Adversary that chooses the hazard
parameter vector `Theta` (wind, fuel moisture, global ROS multiplier `kappa`,
and — in the original spec — ignition). In v0.2 this is an **interface contract
only**; it is intentionally not implemented. A future overseer would plug in as
another palaestrAI agent that writes the wildfire control surface:

* **`gis.wind_override`** `(speed, dir)` — already exposed by `gis_world`
  (negative entries mean "keep default"); a learning overseer can drive the
  meteorology here.
* **Hazard `Theta`** (`kappa`, `dead_fuel_moisture`, etc.) — would be added as
  additional `WildfireCmaMuscle` inputs / a dedicated overseer→wildfire channel.
  The wildfire core already accepts these as params (`agents/wildfire_core.py`),
  so exposing them as actuators is the only wiring required.

The Overseer-Adversary is documented here as the extension point; no learning
adversary ships in v0.2.
```
