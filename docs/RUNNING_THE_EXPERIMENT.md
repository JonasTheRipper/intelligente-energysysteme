# Running & Modifying the Firefighting Experiment (v0.4)

A complete, copy-paste operational guide for the v0.4 multi-resource firefighting
experiment: how to **run** it, **change** the resource mix / doctrine / phases, and
**read** the results into figures. For the *design* see
[`DESIGN_firefighting_actions.md`](DESIGN_firefighting_actions.md); for the release
summary see [`v0.4_RELEASE.md`](v0.4_RELEASE.md).

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
