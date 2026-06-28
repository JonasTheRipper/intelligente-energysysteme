# Changelog

All notable changes to the **SoCal Wildfires — Grid Co-Simulation** testbed.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
