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

## v0.3 A/B experiment + comparison analysis

`palaestrai_socal/experiment_eaton_local_ab.yml` runs the fine-grid Eaton
scenario **twice in one experiment / one store**, identical seed (47) and
identical bounds/raster/wind (8 m/s)/`max_steps` (60) — the only difference is
the firefighter knob:

| phase | uid | `n_planes` |
|-------|-----|-----------|
| A | `phase_0_no_ff` | 0 (pure v0.2 baseline, zero suppression) |
| B | `phase_1_with_ff` | 3 (lays a retardant line) |

palaestrAI tags each `world_states` row with its phase via
`environments.experiment_run_phase_id → experiment_run_phases.uid/.number`.
`analysis/store_readers.read_run` accepts `phase_uid=` / `phase_index=` to read a
**single** phase (omitting both keeps the v0.2 back-compat behaviour, i.e. all
rows for the env); `analysis/store_readers.list_phases` lists the phases present.

The A/B grid env keeps more sensors than the single-phase file (`*-bus-*.vm_pu`,
`*-line-*.p_from_mw`) so the reader can surface new per-step snap keys:
`vmin_pu` / `vmean_pu` (min & mean bus voltage p.u.), `intertie_mw`, and a
`load_mw` alias. `intertie_mw` is a **real** sum of `|p_from_mw|` over stored
line sensors when present; otherwise it falls back to a documented **proxy**
(equal to served-load import) and `meta["intertie_is_proxy"]` is `True`.

Two store-only renderers consume both phases:
- `analysis/make_comparison_timelapse.py` — a 2-row map timelapse (top = phase A
  no-FF, bottom = phase B with-FF, time-synced by frame index), with **fading
  aero-tanker plane icons** on the bottom map derived from the per-step
  SUPPRESSED cell-set diff (firefighter telemetry is on the muscle channel, not
  stored, so plane positions come from `gis.cell_state`; see
  `analysis/plane_icons.py`), plus a right-hand column of A-vs-B metric axes
  (cumulative SAIDI, min/mean voltage, served MW, intertie MW). GIF + MP4.
- `analysis/grid_metrics_report.py` — the static multi-panel PNG companion,
  annotating final deltas (acres saved, SAIDI reduction, MW preserved).

## Firefighter agent (v0.7 — LEARNING, SAC/CQL)

v0.7 adds the first **learning** agent in the testbed. The scripted v0.3–v0.5
firefighter stays exactly as documented above — it becomes the *teacher*. The
learner is a separate agent block that reuses the same `gis.cell_mutations`
control surface:

| Piece | Module |
|-------|--------|
| Brain (Learner process) | `agents.firefighter_drl_brain:FirefighterSacBrain` (subclasses hARL `SACBrain`) |
| Muscle (RolloutWorker process) | `agents.firefighter_drl_agent:LearningFirefighterMuscle` |
| Objective | `agents.saidi_objective:SaidiObjective` |
| Shared contract (numpy-only) | `agents.firefighter_drl` |

### The Box(17) / Discrete(4) contract

`agents/firefighter_drl.py` is the single source of truth both the online muscle
and the offline harvester import, so the two can never drift:

* `OBS_DIM = 17` — a compact vector (burning/burned/suppressed fractions, fire
  front geometry, mean slope, wind trig, served MW, SAIDI delta, step/max_steps),
  **not** a raw raster flatten.
* `N_TACTICS = 4` — the doctrine the policy selects: `ACT_NOOP=0`,
  `ACT_INDIRECT=1`, `ACT_DIRECT=2`, `ACT_TRIAGE=3`. The chosen doctrine is then
  executed by the *same* IncidentCommand machinery the scripted firefighter uses
  and gated by resource availability, so the learner cannot invent physically
  impossible suppression.

**The action is stored as a `(1,)` vector, never a 0-d scalar.** The muscle
returns `np.array([act_id], dtype=np.float64)`, and the offline loader reshapes
each harvested action to match. This is load-bearing rather than cosmetic:
`SACBrain.update()` batches offline and online transitions through a single
`np.array(actions)`, so a 0-d/1-d mix is ragged and raises
`ValueError: ... inhomogeneous shape ...`. hARL catches that and logs
`could not update`, so the failure mode is a firefighter that trains without ever
learning. Both halves of the contract are pinned in
`tests/test_firefighter_drl_agent.py`
(`test_muscle_online_action_is_1d`, `test_offline_bootstrap_action_shape_matches_online`).

### `SaidiObjective` — the reward

`reward = -delta_saidi / scale`, always `<= 0` and exactly `0` when load is fully
served, computed from the agent's `*-load-*.p_mw` sensors
(`CUSTOMERS_PER_MW = 200.0`, defaults `scale = 60.0`, `base_served_mw = 1.0`,
`dt_min = 60.0`). Only `-load-*.p_mw` uids are summed; a missing load
subscription yields `0.0` rather than an error.

Two shapes must both work. palaestrAI's `Memory.tail(1).sensor_readings` is a
**`pd.DataFrame`**, not a list, so the objective type-dispatches instead of doing
`list(readings or [])` — the latter both raises
`ValueError: The truth value of a DataFrame is ambiguous` and, without the `or`,
would iterate column *names*. It also handles the one-row object-cell frame the
Memory shim below produces. The objective additionally accepts its YAML
`params:` block either as a dict or unpacked as keyword arguments, because
palaestrAI's `load_with_params(module, params)` calls `Class(**params)`.

### Ragged-safe Memory shim — installed in **two** processes

`agents/_memory_compat.py` patches
`palaestrai.agent.memory._MuscleMemory._infos_to_df`, which tabulates one step's
sensor readings into a *rectangular* `pd.DataFrame` — one column per sensor uid,
each `np.reshape(np.array(value), -1)`. That assumes every sensor flattens to the
same length.

The DRL firefighter is the first agent to break the assumption: it mixes large
grid rasters (`gis.cell_state` ~23,660 elements, `gis.fuel_class`,
`gis.elevation_m`, `gis.wind_field`) with scalar `*-load-*.p_mw` power sensors,
so the constructor raises `ValueError: All arrays must be of the same length`.

The crash is **inside palaestrAI**, not in our code, and it cannot be dodged by
trimming our own subscription: `rollout_worker` stores `request.sensors` *before*
the per-agent `Filter` runs. Hence the patch:

* **Equal-length columns** take the upstream code path verbatim — the common case
  stays behaviourally identical to stock palaestrAI.
* **Ragged columns** fall back to a one-row frame whose cells hold the whole
  arrays, built as `pd.DataFrame({uid: pd.Series([value], dtype=object)})`.
  (`.at`-style assignment is wrong here: pandas unwraps a length-1 array into a
  0-d scalar, which silently corrupts the scalar power sensors.)

`install()` is idempotent and is called at import in **both**
`firefighter_drl_agent.py` (RolloutWorker) and `firefighter_drl_brain.py`
(Learner). Both are needed because palaestrAI runs them as **separate OS
processes** — the muscle's process hits the ragged frame in
`RolloutWorker._remember`'s debug-log `tail(1)`, and the brain's process hits it
again in the SAC brain's `memory.tail(1).objective.item()` read. Patching only
one leaves the other crashing. Regressions live in `tests/test_memory_compat.py`.

### Offline teacher harvest (CQL bootstrap)

`agents/harvest_teacher_transitions.py` reads the **scripted** firefighting phases
back out of a v0.5 PostgreSQL store and materialises one `.npz` per fire, labelling
each step with the doctrine the teacher effectively chose:

```bash
python -m palaestrai_socal.agents.harvest_teacher_transitions \
    --store postgresql://.../palaestrai_eaton_v05 \
    --out data/offline/eaton_teacher_all.npz \
    [--phases phase_1_air,phase_2_air_ground,phase_3_full_triage]
```

Schema: `obs` (N, 17) float32, `actions` (N,) int64, `rewards` (N,) float32,
`next_obs` (N, 17) float32, `dones` (N,) bool, plus a `meta` 0-d object array
recording the source store, phases and contract constants. Both harvested files
are committed: `data/offline/eaton_teacher_all.npz` and
`data/offline/palisades_teacher_all.npz`.

`FirefighterSacBrain(offline_npz=...)` loads these into the replay buffer during
setup and auto-enables the CQL(H) conservative regulariser, so the policy is
bootstrapped from teacher behaviour before a single online step — then fine-tuned
online.

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
