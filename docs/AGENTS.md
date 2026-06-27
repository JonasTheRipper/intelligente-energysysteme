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
| `damage_mapper` | `agents.damage_agent:DamageMapperMuscle` | `gis.grid_shape`, `gis.bounds`, `gis.cell_state` | `...load-<bus>-<idx>.p_mw` (→ 0) | `agents.damage_core:DamageMapperDriver` |

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

## Firefighter interface (exposed, NOT implemented)

Per scope there is **no FirefighterAgent**. The substrate only guarantees the
read/write surface a future firefighter would bind to, verified by
`tests/test_gis_world_env.py::test_firefighter_interface_contract`:

* **Sensors:** `gis.cell_state`, `gis.front_cells`, `gis.wind_field`,
  `gis.fuel_class`, `gis.bounds`, `gis.grid_shape`, `gis.cell_size_m`.
* **Actuator:** `gis.cell_mutations` accepts a `SUPPRESSED` state written on the
  dedicated `LAYER_SUPPRESSION` layer, so a firefighter's edits are
  distinguishable from the fire's (`palaestrai_socal.spaces` reserves
  `SUPPRESSED=3` / `FLOODED=4` and `LAYER_SUPPRESSION=1` / `LAYER_FLOOD=2`).

A firefighter would subscribe to the fire front + wind, decide where to
suppress, and write `(row, col, SUPPRESSED, LAYER_SUPPRESSION)` mutations. The
fire CA never re-ignites a `SUPPRESSED` cell and never spreads *from* one (it is
not `BURNING`) — see `tests/test_wildfire_agent.py::test_suppressed_cells_do_not_spread`.

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
