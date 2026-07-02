# Running & Modifying the Firefighting Experiment (v0.4 + v0.5)

A complete, copy-paste operational guide for the multi-resource firefighting
experiment: how to **run** it, **change** the resource mix / doctrine / phases, and
**read** the results into figures. For the *design* see
[`DESIGN_firefighting_actions.md`](DESIGN_firefighting_actions.md); for the release
summary see [`v0.4_RELEASE.md`](v0.4_RELEASE.md).

> **v0.5 (calibrated real fires).** In v0.5 the same firefighting experiment is run on
> top of **no-firefighting baselines calibrated to the real Jan-2025 Eaton and
> Palisades perimeters**. If you want the calibrated two-fire workflow (perimeter
> validation, both experiment YAMLs, sim-vs-real figures), jump to **§8**; the
> calibration methodology and results live in
> [`v0.5_CALIBRATION_VALIDATION.md`](v0.5_CALIBRATION_VALIDATION.md).

> **Convention.** All commands assume you are at the repository root with the venv
> active and `PYTHONPATH` set to the repo root:
> ```bash
> cd /path/to/socal-wildfires
> export PYTHONPATH=$PWD
> ```

---

## 0. TL;DR — reproduce the v0.4 result end-to-end

```bash
cd socal-wildfires && export PYTHONPATH=$PWD

# 1. point this at YOUR PostgreSQL store; create a fresh DB + schema
cp runtime_pg_eaton.conf.yaml runtime_pg_myrun.conf.yaml
$EDITOR runtime_pg_myrun.conf.yaml         # set store_uri to a NEW, empty database
palaestrai -c runtime_pg_myrun.conf.yaml database-create

# 2. run the 4-phase experiment (writes to the store; ~30–45 min, ~5 GB RAM peak)
_outputs/run_full_pg.sh runtime_pg_myrun.conf.yaml \
    palaestrai_socal/experiment_eaton_firefighting.yml v04_ff

# 3. render the timelapse + grid-metrics figures
STORE="postgresql://USER:PW@HOST:PORT/DBNAME"   # same as store_uri
python analysis/make_comparison_timelapse.py --store "$STORE" --stride 1 --fps 10 \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only (3 tankers),phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
python analysis/grid_metrics_report.py --store "$STORE" --out analysis/grid_metrics_v04.png \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only (3 tankers),phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
```

---

## 1. Prerequisites

| Component | Pinned / expected |
|---|---|
| Python | 3.12 |
| numpy / pandas | **1.26.4 / 2.1.4** (hard-pinned; the CA relies on numpy semantics) |
| palaestrai | 3.5.9 |
| mosaik | 3.5.0 |
| pandapower | 3.4.0 (needed for the grid env + point protection; degrades gracefully if absent) |
| store | PostgreSQL 14+ (TimescaleDB recommended; `database-create` adds the extension + hypertables) |

A standalone (no-broker) driver also exists for the wildfire CA alone — see the main
[`README.md`](../README.md) §A. The firefighting comparison, however, needs the full
palaestrAI run because it depends on the multi-phase store.

---

## 2. The runtime config (store connection)

The experiment file says *what* to simulate; the **runtime config** says *where to
store it* and which broker port to use. Runtime configs are **git-ignored**
(`runtime_pg*.conf.yaml`) because they contain a local DB password.

Copy an existing one and edit the `store_uri`:

```yaml
# runtime_pg_myrun.conf.yaml
store_uri: "postgresql://palaestrai:YOUR_PW@127.0.0.1:5433/palaestrai_myrun"
executor_bus_port: 4242
logger_port: 0
```

**One store = one experiment run.** Phase uids must be unique within a store, so use
a **fresh, empty database** per run (or drop and recreate). Create the DB and schema:

```bash
# create the empty database (psql), then the palaestrAI schema
psql -h 127.0.0.1 -p 5433 -U palaestrai -d postgres \
     -c "CREATE DATABASE palaestrai_myrun;"
palaestrai -c runtime_pg_myrun.conf.yaml database-create
```

`database-create` is idempotent for the schema but expects the database itself to
exist. It installs the TimescaleDB extension and converts `world_states` /
`muscle_actions` to hypertables.

---

## 3. Running the experiment

### 3.1 Recommended: the background run script

`_outputs/run_full_pg.sh <runtime_conf> <experiment_yml> <tag>` starts a memory
sampler and launches palaestrAI, writing logs to `_outputs/<tag>_run.log` (with a
trailing `<tag>_EXIT_RC=<n>` line) and `_outputs/<tag>_sampler.log`.

```bash
# foreground (blocks the shell)
_outputs/run_full_pg.sh runtime_pg_myrun.conf.yaml \
    palaestrai_socal/experiment_eaton_firefighting.yml v04_ff

# detached (survives shell exit) — recommended for the ~30–45 min run
setsid bash _outputs/run_full_pg.sh runtime_pg_myrun.conf.yaml \
    palaestrai_socal/experiment_eaton_firefighting.yml v04_ff \
    < /dev/null > /dev/null 2>&1 &
```

### 3.2 Plain palaestrAI

```bash
palaestrai -c runtime_pg_myrun.conf.yaml start \
    palaestrai_socal/experiment_eaton_firefighting.yml
```

### 3.3 Monitoring a detached run

```bash
# is it still running?
pgrep -fc "palaestrAI\["

# did it finish, and how?
grep EXIT_RC _outputs/v04_ff_run.log        # want: v04_ff_EXIT_RC=0

# memory / disk over time
tail -f _outputs/v04_ff_sampler.log

# phase + snapshot progress (psql)
psql "$STORE" -tAc "SELECT id, uid FROM experiment_run_phases ORDER BY id;"
psql "$STORE" -tAc "SELECT environment_id, count(*) FROM world_states GROUP BY 1 ORDER BY 1;"
```

Each phase has **two** environments (`gis_world`, `socal_grid`); a complete phase has
**60 snapshots per environment**. Phases run **sequentially**, so peak RAM ≈ a single
phase (~5 GB). `phase_3_full_triage` is the slowest (engines + point protection +
value-raster build per step).

### 3.4 Cleanup after an interrupted run

Killing palaestrAI can leave mosaik/zmq orphans:

```bash
pkill -9 -f "palaestrAI\["
pgrep -fc "palaestrAI\["    # verify 0 (run in a fresh shell; it can match its own)
```

---

## 4. Modifying the experiment

The experiment is a multi-phase palaestrAI YAML. **Only the firefighter agent's
`muscle.params` block changes between phases** — everything else (envs, seed, turn
order, other agents) is identical, which is what makes the phases a clean A/B/C/D.

### 4.1 The firefighter knobs

In `palaestrai_socal/experiment_eaton_firefighting.yml`, each phase has a
`firefighter` agent whose muscle is `FirefighterMuscle`. Per the design's
"constants over parameters" rule, the knobs are **resource counts + two switches**:

```yaml
- name: firefighter
  brain:  { name: palaestrai_socal.agents.capped_dummy_brain:CappedDummyBrain }
  muscle:
    name: palaestrai_socal.agents.firefighter_agent:FirefighterMuscle
    params:
      n_planes: 3            # Large Air Tankers (the v0.3 knob)
      n_helos: 0             # rotor-wing water/foam
      n_crews: 0             # hand crews (handline → CONTAINED)
      n_dozers: 0            # dozers (faster line → CONTAINED)
      n_engines: 0           # engines (point protection → CONTAINED)
      doctrine: auto         # "auto" | "direct" | "indirect"
      protect_assets: false  # true → engines protect grid-critical cells
      grid_json: midas_socal/socal_grid_midas_rescaled.json   # needed by protect_assets
  objective: { name: palaestrai.agent.dummy_objective:DummyObjective }
```

| Knob | Meaning | Default | Effect |
|---|---|---|---|
| `n_planes` | aero tankers | 1 | retardant line; grounds ≥18 m/s; state `SUPPRESSED` |
| `n_helos` | helicopters | 0 | water/foam line; degrades >16, grounds ≥22 m/s; `SUPPRESSED` |
| `n_crews` | hand crews | 0 | slow handline; slope-derated; never grounded; `CONTAINED` |
| `n_dozers` | dozers | 0 | fast line; slope-derated; `CONTAINED` |
| `n_engines` | engines | 0 | point protection of high-value cells; `CONTAINED` |
| `doctrine` | attack mode | `auto` | `auto` chooses direct vs indirect by fireline intensity; `indirect` reproduces v0.3 |
| `protect_assets` | grid triage | `false` | when `true`, engines protect grid-critical cells via the value raster |
| `grid_json` | grid model path | — | required when `protect_assets: true` |

**Identity guarantees (don't break these):**
- `n_planes>0` with **all** other counts `0` and `doctrine: indirect` (or `auto` when
  intensity stays low) == **v0.3** retardant line, bit-for-bit.
- **All** counts `0` (or no firefighter agent) == **v0.2** baseline, bit-for-bit.

### 4.2 Change the resource mix of a phase
Edit the `params` of that phase's `firefighter` block (e.g. add `n_helos: 2`), then
re-run into a **fresh** store. To keep the v0.3/v0.2 regression intact, leave
`phase_0_no_ff` (no firefighter) and `phase_1_air` (`n_planes:3, doctrine:indirect`)
as they are.

### 4.3 Add a phase
Phases are top-level list entries under `schedule:`. The cleanest way to add one is
to **copy an existing phase block**, rename its uid (must be unique), and change only
the firefighter `params`. Keep the environment, seed, and turn-order blocks identical
so the new phase stays comparable. Then pass the new uid to the renderers (§5).

### 4.4 Change the scenario (seed / wind / steps / fire size)
These live in the environment params (shared by all phases — change them in **every**
phase, or they won't be comparable):
- `seed` — RNG seed (47 is the canonical fine-grid Eaton run).
- wind speed/direction — drive both fire spread and resource grounding.
- `max_steps` — episode length (60 = ~2.5 sim-days at the current step size).
- ignition + grid — the fine-grid Eaton footprint.

### 4.5 Tune resource productivity (rare)
Productivity / slope / wind constants are **module-level constants** in
`palaestrai_socal/agents/firefighting/resources.py` and `doctrine.py`
(e.g. `DOZER_LINE_M_PER_HOUR`, `CREW_SLOPE_CUTOFF_DEG`,
`DIRECT_ATTACK_MAX_INTENSITY`). Changing them affects **all** phases and is a model
change, not an experiment knob — update the relevant tests in
`tests/test_firefighting_*` if you do.

---

## 5. Reading the results → figures

Both renderers read a phase-aware store via `analysis/store_readers.py`. Set:

```bash
STORE="postgresql://USER:PW@HOST:PORT/DBNAME"
```

### 5.1 `--phases` grammar (back-compatible)

```
uid                       # plane count inferred from uid
uid:n_planes              # v0.3 syntax (e.g. phase_1_air:3)
uid:n_planes:Label text   # v0.4 — descriptive label; ':' separates, NO commas in labels
```

Commas separate phases, so a label must not contain a comma (use `+` or `/`).

### 5.2 Comparison timelapse (MP4 + GIF)

```bash
python analysis/make_comparison_timelapse.py --store "$STORE" \
  --stride 1 --fps 10 --outdir analysis \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only (3 tankers),phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
# -> analysis/comparison_timelapse.mp4 and .gif
```

One **map row per phase** (with per-tactic colours and a HUD showing retardant /
ground-line cell counts) plus four shared metric panels (SAIDI, voltage, served
power, intertie). Useful flags: `--stride N` (sub-sample frames), `--fps N`,
`--title "…"`.

### 5.3 Grid-metrics PNG (4 panels + cost-effectiveness KPI)

```bash
python analysis/grid_metrics_report.py --store "$STORE" \
  --out analysis/grid_metrics_v04.png \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only (3 tankers),phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
```

Panels: cumulative SAIDI, bus voltage (min dashed / mean solid), total served power,
intertie flow. It prints a `deltas` dict per phase with acres-saved, MW preserved,
SAIDI avoided, and the **MW/SAIDI preserved per tanker-hour** KPI.

### 5.4 Inspect frames before sharing

```bash
ffmpeg -y -i analysis/comparison_timelapse.mp4 \
  -vf "select='eq(n\,0)+eq(n\,30)+eq(n\,58)'" -vsync 0 analysis/_frame%02d.png
# view analysis/_frame*.png, then:
rm -f analysis/_frame*.png
```

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `phase uid already exists` / duplicate phases | The store isn't empty. Use a fresh database or drop+recreate, then `database-create`. |
| Renderer: `no world_states for environment … phase_uid='…'` | A label contained a comma (it got parsed as a phase) **or** the phase uid is wrong. Remove commas from labels; check `list_phases`. |
| All FF phases labelled "3 planes" | You used the old `uid:n` syntax. Add `:Label` to each token (§5.1). |
| `EXIT_RC` non-zero / run hangs | Check `_outputs/<tag>_run.log`; kill orphans (`pkill -9 -f "palaestrAI\["`) before re-running. |
| Point protection does nothing | `protect_assets:true` requires a valid `grid_json` **and** pandapower; otherwise it's a graceful no-op. |
| OOM during a run | Phases are sequential (~5 GB peak). Reduce `max_steps` or grid size, or give the box more RAM. |
| `dict + int` warning in logs | Should be gone in v0.4 (telemetry returns `None` on the brain channel). If you see it, you're on pre-v0.4 code. |

---

## 7. Quick reference

```bash
# run
setsid bash _outputs/run_full_pg.sh runtime_pg_myrun.conf.yaml \
  palaestrai_socal/experiment_eaton_firefighting.yml v04_ff < /dev/null > /dev/null 2>&1 &
grep EXIT_RC _outputs/v04_ff_run.log

# tests
python -m pytest tests/ -q -m "not slow"

# figures (set STORE first)
python analysis/make_comparison_timelapse.py --store "$STORE" --stride 1 --fps 10 \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only,phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
python analysis/grid_metrics_report.py --store "$STORE" --out analysis/grid_metrics_v04.png \
  --phases "phase_0_no_ff:0:No firefighting,phase_1_air:3:Air only,phase_2_air_ground:3:Air+ground,phase_3_full_triage:3:Full triage+protect"
```

---

## 8. v0.5 — the calibrated two-fire workflow

v0.5 runs the **same** firefighting experiment on **no-firefighting baselines
calibrated to the real Jan-2025 Eaton and Palisades perimeters**. Methodology, exact
calibrated parameters, and the pass/fail table are in
[`v0.5_CALIBRATION_VALIDATION.md`](v0.5_CALIBRATION_VALIDATION.md); this section is the
operational recipe.

### 8.1 Verify the calibration (no store needed)

This rebuilds each baseline via the production `WildfireDriver` path and asserts the
hard bar (**Dice ≥ 0.80 AND \|area%\| ≤ 10**) at **peak and final(60)** for both fires:

```bash
python analysis/verify_calibration.py
# -> Eaton final(60) Dice=0.906 area%=-2.2% PASS ; Palisades Dice=0.952 area%=+3.7% PASS
```

The calibrated knobs (grid, bounds, ignition, base_speed, boundary_gain, moisture,
kappa, fuel_reclass, containment_margin) are baked into `verify_calibration.py`,
`analysis/make_sim_vs_real.py`, and both experiment YAMLs. **Do not change them** unless
you re-calibrate against the perimeter (they are what makes the fires match reality).

### 8.2 Static sim-vs-real perimeter overlay

One-glance "did we match the real fire?" figure — simulated final burn vs. official
CAL FIRE perimeter, with a metrics box:

```bash
python analysis/make_sim_vs_real.py --fire eaton \
    --out analysis/_v05_eaton/sim_vs_real_eaton.png
python analysis/make_sim_vs_real.py --fire palisades \
    --out analysis/_v05_palisades/sim_vs_real_palisades.png
```

### 8.3 Run the firefighting counterfactuals for both fires

Each fire has its own experiment YAML and its own store (phase uids must be unique per
store). Set up TimescaleDB, create the schema, then run:

```bash
# --- Eaton ---
cp runtime_pg_eaton.conf.yaml runtime_pg_myeaton.conf.yaml   # set store_uri -> fresh DB
palaestrai -c runtime_pg_myeaton.conf.yaml database-create
setsid bash _outputs/run_full_pg.sh runtime_pg_myeaton.conf.yaml \
    palaestrai_socal/experiment_eaton_firefighting.yml eaton_v05 < /dev/null >/dev/null 2>&1 &

# --- Palisades (use a DIFFERENT DB and a different executor_bus_port) ---
cp runtime_pg_palisades.conf.yaml runtime_pg_mypal.conf.yaml  # set store_uri -> fresh DB
palaestrai -c runtime_pg_mypal.conf.yaml database-create
setsid bash _outputs/run_full_pg.sh runtime_pg_mypal.conf.yaml \
    palaestrai_socal/experiment_palisades_firefighting.yml palisades_v05 < /dev/null >/dev/null 2>&1 &

grep EXIT_RC _outputs/eaton_v05_run.log _outputs/palisades_v05_run.log   # want =0
```

> **Store-schema gotcha.** `database-create` needs the database to exist **and** the
> TimescaleDB (and PostGIS) extensions installed first, otherwise the store receiver
> fails with `relation "experiments" does not exist`. On a fresh DB:
> ```sql
> CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
> CREATE EXTENSION IF NOT EXISTS postgis CASCADE;
> ```
> then run `palaestrai -c <conf> database-create`.

> **Two experiments at once.** Give each fire a **separate database** and a **distinct
> `executor_bus_port`** in its runtime conf (e.g. 4242 for Eaton, 4243 for Palisades)
> so the two runs don't collide on the broker port.

### 8.4 Render the per-fire figures (timelapse + grid metrics)

Same renderers as §5, pointed at each fire's store, with the fire's perimeter overlaid
and city labels for orientation:

```bash
# Eaton
STORE_EATON="postgresql://USER:PW@HOST:PORT/palaestrai_eaton_v05"
python analysis/make_comparison_timelapse.py --store "$STORE_EATON" \
  --stride 1 --fps 10 --outdir analysis/_v05_eaton \
  --phases "phase_0_no_ff:0:no firefighters,phase_1_air:3:3 aero tankers,phase_2_air_ground:3:air+ground,phase_3_full_triage:3:full triage" \
  --perimeter data/perimeters/eaton_perimeter.geojson \
  --cities "Altadena,-118.131,34.190;Pasadena,-118.145,34.156;Sierra Madre,-118.053,34.162;La Canada,-118.201,34.199"
python analysis/grid_metrics_report.py --store "$STORE_EATON" \
  --out analysis/_v05_eaton/grid_metrics_eaton.png \
  --phases "phase_0_no_ff:0:no firefighters,phase_1_air:3:3 aero tankers,phase_2_air_ground:3:air+ground,phase_3_full_triage:3:full triage"

# Palisades
STORE_PAL="postgresql://USER:PW@HOST:PORT/palaestrai_palisades_v05"
python analysis/make_comparison_timelapse.py --store "$STORE_PAL" \
  --stride 1 --fps 10 --outdir analysis/_v05_palisades \
  --phases "phase_0_no_ff:0:no firefighters,phase_1_air:3:3 aero tankers,phase_2_air_ground:3:air+ground,phase_3_full_triage:3:full triage" \
  --perimeter data/perimeters/palisades_perimeter.geojson \
  --cities "Pacific Palisades,-118.526,34.048;Malibu,-118.667,34.032;Santa Monica,-118.491,34.020;Topanga,-118.601,34.094"
python analysis/grid_metrics_report.py --store "$STORE_PAL" \
  --out analysis/_v05_palisades/grid_metrics_palisades.png \
  --phases "phase_0_no_ff:0:no firefighters,phase_1_air:3:3 aero tankers,phase_2_air_ground:3:air+ground,phase_3_full_triage:3:full triage"
```

The timelapse renderer's v0.5 flags: `--perimeter <geojson>` overlays the official
perimeter (cyan dashed); `--cities "Name,lon,lat;…"` drops labelled place markers on the
GIS map for orientation.
