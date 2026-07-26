# Changelog

All notable changes to the **SoCal Wildfires — Grid Co-Simulation** testbed.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [v0.7] — 2026-07-26

**The learning firefighter** — the testbed gets its first *learning* agent: a SAC/CQL
firefighter that is bootstrapped from the scripted v0.4/v0.5 incident commander (now the
*teacher*) and then fine-tuned online. The scripted responder, the CA kernel and the
calibrated v0.5 baselines are **unchanged** — the learner is a separate agent block that
reuses the same `gis.cell_mutations` control surface. Architecture:
[`docs/AGENTS.md`](docs/AGENTS.md); operational how-to:
[`docs/RUNNING_THE_EXPERIMENT.md`](docs/RUNNING_THE_EXPERIMENT.md) §9.

### Added
- **DRL firefighter agent**: `agents/firefighter_drl_brain.py`
  (`FirefighterSacBrain`, a hARL `SACBrain` subclass) and
  `agents/firefighter_drl_agent.py` (`LearningFirefighterMuscle`). The policy selects a
  *doctrine*, which is then executed by the same `IncidentCommand` machinery the scripted
  firefighter uses and gated by resource availability, so the learner cannot invent
  physically impossible suppression.
- **`agents/firefighter_drl.py` — the shared `Box(17)` / `Discrete(4)` contract**
  (numpy-only, imported by both the online muscle and the offline harvester so the two
  cannot drift): a compact 17-feature observation (fire fractions, front geometry, mean
  slope, wind trig, served MW, SAIDI delta, step fraction) rather than a raster flatten,
  and 4 tactics (`ACT_NOOP`, `ACT_INDIRECT`, `ACT_DIRECT`, `ACT_TRIAGE`).
- **`agents/saidi_objective.py` (`SaidiObjective`)** — `reward = -delta_saidi / scale`,
  always ≤ 0 and exactly 0 when load is fully served, computed from the agent's
  `*-load-*.p_mw` sensors (`CUSTOMERS_PER_MW = 200.0`).
- **Offline teacher-transition harvest** (`agents/harvest_teacher_transitions.py`):
  reads the scripted firefighting phases back out of a v0.5 store and writes one `.npz`
  per fire — `obs` (N, 17) float32, `actions` (N,) int64, `rewards`, `next_obs`, `dones`,
  plus a `meta` record of the source store, phases and contract constants. Both harvested
  files are committed (`data/offline/{eaton,palisades}_teacher_all.npz`).
  `FirefighterSacBrain(offline_npz=…)` loads them into the replay buffer during setup and
  auto-enables the **CQL(H)** conservative regulariser, so the policy is behaviour-cloned
  before a single online step.
- **`agents/_memory_compat.py`** — a ragged-safe shim for palaestrAI's
  `_MuscleMemory._infos_to_df` (see *Fixed*).
- **Two new phases** in `experiment_eaton_firefighting.yml` — `phase_4_drl_train` and
  `phase_5_drl_test` — so the learner trains and is evaluated in the same store as the
  four scripted phases it is measured against.
- **Two production long-run experiments**:
  `experiment_eaton_firefighting_drl_long.yml` and
  `experiment_palisades_firefighting_drl_long.yml` (`episodes: 400`,
  `evaluate_every: 20`, `cql_alpha: 1.0`, `update_after: 1000`, `start_steps: 1000`,
  `use_real_dem: true`, each pointed at its own harvested `.npz`). Both stop early via
  `AgentObjectiveTerminationCondition` on the firefighter's `brain_avg30: -0.0002`.
- **`analysis/drl_firefighter_report.py`** — store-only report comparing the learned
  `phase_5_drl_test` against the scripted baseline phase (acres, SAIDI) plus the training
  reward curve; backed by the new `store_readers.read_agent_objectives()`.
- **+55 tests** across `tests/test_firefighter_drl{,_agent}.py`,
  `test_harvest_teacher_transitions.py`, `test_saidi_objective.py` and
  `test_memory_compat.py` (**235 passed, 4 skipped**; the skips are environmental —
  absent DEM cache, absent v0.5 store, missing MIDAS CSVs).

### Fixed
- **`SaidiObjective` loader contract** — palaestrAI's `load_with_params(module, params)`
  calls `Class(**params)`, so the constructor now accepts its YAML `params:` block either
  as a dict *or* unpacked as keyword arguments.
- **Ragged sensor mixes crashed palaestrAI's Memory.** `_infos_to_df` tabulates one step's
  readings into a *rectangular* `DataFrame`, assuming every sensor flattens to the same
  length. The DRL firefighter is the first agent to break that assumption — it mixes large
  grid rasters (`gis.cell_state` ≈ 23,660 elements) with scalar `*-load-*.p_mw` — raising
  `ValueError: All arrays must be of the same length`. The crash is *inside* palaestrAI and
  cannot be dodged by trimming our own subscription, because `rollout_worker` stores
  `request.sensors` before the per-agent `Filter` runs. `_memory_compat.install()` keeps the
  equal-length path byte-identical to upstream and falls back to a one-row object-cell frame
  only when columns are ragged. It is **idempotent and installed in two processes** —
  RolloutWorker *and* Learner — because palaestrAI runs them as separate OS processes that
  hit the ragged frame at different call sites.
- **`SaidiObjective.internal_reward` on real Memory rows.** `Memory.tail(1).sensor_readings`
  is a `pd.DataFrame`, not a list; the previous `list(readings or [])` both raised
  *truth value of a DataFrame is ambiguous* and, without the `or`, would have iterated column
  *names*. It now type-dispatches, and also handles the one-row object-cell frame the shim
  produces.
- **Offline/online action-shape mismatch in the CQL bootstrap.** The muscle emits
  `np.array([act_id])` — shape `(1,)` — while the offline loader stored a 0-d scalar.
  `SACBrain.update()` batches both sources through a single `np.array(actions)`, so once
  `update_after` was passed that array was ragged and every update raised
  *inhomogeneous shape*. hARL logs the failure and continues, so the symptom was a
  firefighter that trained without ever learning. Offline actions are now reshaped to `(1,)`,
  and **both halves of the contract are pinned** by
  `test_muscle_online_action_is_1d` and
  `test_offline_bootstrap_action_shape_matches_online` (the latter asserts the actual
  crashing operation — one batch spanning both sources — succeeds).

### Verification
- The six-phase Eaton experiment (`phase_0_no_ff` → `phase_5_drl_test`) ran end-to-end to a
  clean exit with zero error signatures. Evidence the learner actually trained (rather than
  merely not erroring): the firefighter dumps 3 brain states in each scripted phase but 16 in
  `phase_4_drl_train` and 12 in `phase_5_drl_test`, i.e. the SAC/CQL update fired and the
  trained brain transferred into the evaluation phase.
- **This was a short pipeline smoke run on the synthetic-terrain fallback, not a real-DEM
  result.** The DEM cache was absent, so `gis_world_env` degraded to synthetic elevation as
  designed. It validates the pipeline — it caught the action-shape defect above — but its
  SAIDI figures are **not** metric-comparable to production, and two episodes cannot train a
  policy regardless. Metric-comparable runs use the real DEM
  (`data/dem/fetch_dem_tiles.py`, OpenTopography key required) and the long-run YAMLs.

### Preserved
- The **scripted** v0.3–v0.5 firefighter is untouched and remains the teacher; all prior
  identity guarantees (v0.2 no-op, v0.3 retardant-line, v0.4 tactics) stay test-guarded.

## [v0.6] — 2026-07-02

**Timelapse presentation rework** — the firefighter-response comparison timelapse now
renders over a **real satellite basemap** and fills its layout with far less whitespace.
Simulation results, calibration, the CA kernel and all experiment configs are
**unchanged** — this release only touches the `analysis/` visualisation code.

### Added
- **Satellite basemap** (`analysis/satellite_basemap.py`): fetches and stitches public
  **Esri World Imagery** XYZ tiles (no API key), reprojects Web-Mercator → PlateCarree
  to align with the fire raster, and optionally overlays roads/place labels from Esri's
  reference layer. `satellite_rgb(extent, px_target=1600, with_labels=True)` returns an
  `(H, W, 3)` float RGB array. Tiles are disk-cached under `data/basemap_cache/`.
  Attribution ("Basemap imagery: Esri, Maxar, Earthstar Geographics") is printed on the
  figure per the imagery terms of use.

### Changed
- **`analysis/make_comparison_timelapse.py`**: the map rows now draw the satellite
  mosaic instead of the synthetic hillshade (`_basemap_rgb`, `_draw_map`). This fixes
  the v0.5 issues the reviewer flagged: the LA basin no longer looks like open ocean
  (the *real* Pacific coastline shows for Palisades), and urban street grids + city
  names are now visible so the terrain is readable by non-locals. If tiles can't be
  fetched the renderer transparently falls back to the v0.5 hillshade.
- **Figure geometry** re-tuned so each wide-and-short fire map fills its column
  (`aspect="auto"`, wider map column, per-row height capped) — the large empty bands
  above/below each v0.5 map are gone.
- **Fire/scar overlay opacity** (`_fire_cmap`) nudged up so the burn front, retardant
  lines and burned scar stay the clear focal content over the darker, busier satellite
  imagery.

### Dependencies
- Added `requests` and `Pillow` to `requirements.txt` (tile fetch + mosaic). Both were
  already present transitively; now pinned explicitly. No new heavy GIS deps
  (`contextily`/`rasterio` are **not** required).

## [v0.5] — 2026-07-02

**Real-fire calibration** — the no-firefighting baseline is now calibrated to the
**official CAL FIRE perimeters** of the January 2025 **Eaton** and **Palisades** fires,
and the v0.4 firefighting phases are re-run on top of each calibrated baseline as
counterfactuals. The CA kernel and the firefighting model are **unchanged**. Full
notes: [`docs/v0.5_CALIBRATION_VALIDATION.md`](docs/v0.5_CALIBRATION_VALIDATION.md);
operational how-to: [`docs/RUNNING_THE_EXPERIMENT.md`](docs/RUNNING_THE_EXPERIMENT.md) §8.

### Added
- **Perimeter-informed wind + containment** (`wildfire_cma/wind_field.py`):
  `perimeter_informed_wind_field(...)` (time-/space-varying Santa-Ana wind schedule
  derived from the real perimeter geometry), `reclassify_burned_footprint(...)`
  (homogenise fuel inside the observed footprint; Palisades `fuel_reclass=True`), and
  `contain_burnable_footprint(fuel, real_mask, margin_cells=2)` (dilate the real mask
  by `containment_margin` cells and set fuel outside → non-burnable, representing the
  fuel breaks / terrain / suppression that arrested each fire). Without containment the
  free-running CA overshoots by +145 % (Eaton) / +129 % (Palisades).
- **Validation harness** (`analysis/perimeter_validation.py`):
  `load_perimeter_polygons`, `rasterize_perimeter`, `score` (Dice / Jaccard / acres /
  area%), `meets_bar` (Dice ≥ 0.8 **and** \|area%\| ≤ 10).
- **Official CAL FIRE perimeters** (`data/perimeters/{eaton,palisades}_perimeter.geojson`).
- **Per-fire experiments**: `experiment_eaton_firefighting.yml` (130×182) and
  `experiment_palisades_firefighting.yml` (159×219), each with the calibrated
  environment (grid/bounds/ignition/wind/`containment_margin: 2`) across all 4 phases.
- **`analysis/verify_calibration.py`** — asserts peak **and** final(60) meet the bar for
  both fires via the production `WildfireDriver` path (no monkeypatch).
- **`analysis/make_sim_vs_real.py`** — static simulated-vs-official perimeter overlay
  with a metrics box, per fire.
- **+4 containment tests** (`tests/test_wind_field.py::TestContainBurnableFootprint`).

### Changed
- **Timelapse renderer** (`analysis/make_comparison_timelapse.py`): bigger GIS map with
  visible hillshaded topography; new `--perimeter <geojson>` (cyan-dashed official
  perimeter overlay) and `--cities "Name,lon,lat;…"` (labelled place markers) flags; an
  orientation legend and fixed axis ticks. Two separate timelapses (Eaton, Palisades).
- **`WildfireDriver` / `WildfireCmaMuscle`**: accept `perimeter_path`, `base_speed`,
  `boundary_gain`, `fuel_reclass`, `containment_margin` kwargs (back-compatible — the
  scalar-wind path is unchanged when no `perimeter_path` is supplied).

### Results (no-firefighting baseline vs. official perimeter, final env step 60)
- **Eaton** (130×182): **Dice = 0.906**, area error **−2.2 %** (13,786 vs 14,102 ac) — **PASS**.
- **Palisades** (159×219): **Dice = 0.952**, area error **+3.7 %** (24,677 vs 23,799 ac) — **PASS**.
- Firefighting counterfactuals: full-triage holds Eaton **SAIDI 1.61 → 0.24** (~5,588 ac
  saved) and Palisades ~1,811 ac saved — grid benefit again dominates raw acreage.

### Preserved
- **CA kernel unchanged**; **v0.2 / v0.3 / v0.4 identities** all still hold and are
  test-guarded (**181 passed, 1 skipped**).

## [v0.4] — 2026-06-28

**Full-blown firefighting** — the v0.3 aero-tanker responder becomes a deterministic,
scripted multi-resource incident-command model (design steps 1–7; learning brain
deliberately skipped). Full notes: [`docs/v0.4_RELEASE.md`](docs/v0.4_RELEASE.md);
operational how-to: [`docs/RUNNING_THE_EXPERIMENT.md`](docs/RUNNING_THE_EXPERIMENT.md).

### Added
- **`firefighting/` package** (`palaestrai_socal/agents/firefighting/`): pure-numpy,
  no palaestrAI dependency. `resources.py` (TankerFleet / HeloFleet / HandCrews /
  Dozers / Engines, each with a documented-constant `capacity(...)`), `tactics.py`
  (indirect/direct/containment/burnout/point_protect), `doctrine.py` (intensity-driven
  direct-vs-indirect + anchor-and-flank), `planner.py` (`IncidentCommand.propose` +
  `value_raster_from_buses`). The only operational knobs are resource **counts**.
- **`CONTAINED` cell state (= 5)**: durable ground line / protected point, arbitrated
  strictly between `SUPPRESSED` and `BURNED_OUT`; `STATE_PRIORITY` is total/tie-free;
  CA non-ignitable guard mirrors `SUPPRESSED`; does not age out.
- **Grid coupling / point protection**: engines protect grid-critical cells via a
  value raster built from the DamageMapper bus→cell map (`protect_assets: true`);
  graceful no-op without pandapower / grid JSON.
- **4-phase experiment** `palaestrai_socal/experiment_eaton_firefighting.yml`
  (`phase_0_no_ff`, `phase_1_air`, `phase_2_air_ground`, `phase_3_full_triage`).
- **Comprehensive docs**: `docs/v0.4_RELEASE.md`, `docs/RUNNING_THE_EXPERIMENT.md`.
- **+31 tests** (134 passing): `tests/test_firefighting_{resources,tactics,planner}.py`,
  `tests/test_contained_state.py`.

### Changed
- **Telemetry fix**: the firefighter muscle returns `None` on the brain channel
  (kills the v0.3 `dict + int` warning); telemetry lives on `self._last_telemetry`.
- **Renderers**: per-tactic colours (retardant pink vs ground-line brown) and a
  ground-line HUD in the timelapse; MW/SAIDI-preserved-per-resource-hour KPI in
  `grid_metrics_report.build_figure_n`; both renderers now accept an optional
  descriptive phase label via the extended, back-compatible `--phases`
  `uid:n_planes:Label` grammar.
- `palaestrai_socal/spaces.py`, `wildfire_cma/cma.py`, `analysis/store_readers.py`
  updated for the `CONTAINED` state.

### Results
- Escalating the response saves 137 → 365 → **654 acres** vs baseline, but the
  decisive effect is on the grid: the full-triage phase (point protection) holds
  **SAIDI at 0.27** (vs ~1.97 baseline) and keeps ~20 MW more load energised — a far
  larger grid benefit than acreage saved alone.

### Preserved
- **v0.3 identity** (tankers-only + indirect == `select_retardant_line`) and **v0.2
  no-op identity** (zero budget == baseline, bit-for-bit) both still hold and are
  test-guarded.

## [v0.3] — 2026-06-28

First **responder agent** and multi-fleet comparison tooling.
Full notes: [`docs/v0.3_RELEASE.md`](docs/v0.3_RELEASE.md).

### Added
- **Firefighter Agent** (`palaestrai_socal/agents/firefighter_agent.py`,
  `firefighter_core.py`): scripted fleet of `n_planes` Large Air Tankers laying a
  retardant firebreak via `SUPPRESSED` mutations through the `gis.cell_mutations`
  actuator. Single knob `n_planes`; grounded in wind ≥ 18 m/s; true no-op when no
  retardant is laid.
- **3-phase comparison experiment**
  (`palaestrai_socal/experiment_eaton_local_ab.yml`): one store, identical seed,
  three phases — `phase_0_no_ff` (0 planes), `phase_1_with_ff` (3), and
  `phase_2_with_ff7` (7).
- **N-phase comparison timelapse** (`analysis/make_comparison_timelapse.py`):
  one map row per phase + four shared metric panels (SAIDI, voltage, served
  power, intertie); fading plane icons and per-phase HUDs; `--phases` CLI.
- **N-way grid-metrics PNG** (`analysis/grid_metrics_report.py`,
  `build_figure_n()`): per-phase SAIDI / power / intertie lines, min/mean voltage,
  acres-saved banner; `--phases` CLI.
- Phase-aware store reader (`analysis/store_readers.py`): `read_run(phase_uid=…)`,
  `list_phases()`, new grid metrics `vmin_pu`/`vmean_pu`/`intertie_mw`/`load_mw`.

### Results
- Fleet size scales suppression: 0 → 3 → 7 tankers saves 0 → 137 → **294 acres**
  versus baseline (≈1.6 %), with the largest SAIDI reduction at 7 planes
  (1.98 → 1.84). Real line-flow sensors (`intertie_is_proxy = False`).

### Changed
- Comparison-timelapse layout fix: hide redundant longitude tick labels on upper
  map rows and increase row spacing so map titles no longer overlap axes.

### Known issues
- Firefighter muscle returns a telemetry dict; the dummy-brain learner logs a
  non-fatal `TypeError (dict + int)` once per firefighter turn. Output unaffected.

## [v0.2] — Agentic refactor

Real MIDAS/mosaik power-grid env + GIS world env + Wildfire CMA agent +
DamageMapper coupling agent (all under palaestrAI); PostgreSQL + TimescaleDB
store with `state_dump` trimming; Eaton-Fire no-suppression validation scenario;
dtype-safe actuator writes; `CappedDummyBrain` memory fix; extended test suite.

## [v0.1] — Baseline

SoCal wildfire / power-grid co-simulation baseline: NOAA-enriched MIDAS scenario,
palaestrAI environment, GUARDIAN cellular-automaton wildfire agent, and the
five-day Santa-Ana analysis.
