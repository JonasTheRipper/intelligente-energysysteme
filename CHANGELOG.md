# Changelog

All notable changes to the **SoCal Wildfires — Grid Co-Simulation** testbed.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
