# SoCal Wildfires — Grid Co-Simulation

A reproducible co-simulation of **Southern California wildfires impacting the
power grid**, built on the OFFIS ARL stack ([MIDAS](https://pypi.org/project/midas-mosaik/)
/ mosaik and [palaestrAI](https://pypi.org/project/palaestrai/)). A wildfire,
modelled as a **GUARDIAN Constrained-Mutation operator**, spreads across a
California cellular-automaton landscape, removes the transmission/sub-transmission
assets it engulfs, and the resulting de-energised grid is solved with pandapower.

This repository delivers four things:

1. **NOAA-enriched MIDAS scenario** — the SoCal MIDAS co-simulation, with its
   weather provider switched from DWD Bremen to **NOAA** data (implemented
   *inside* the MIDAS scenario, not as a separate simulator).
2. **palaestrAI environment** — the SoCal MIDAS environment wrapped as a
   first-class `palaestrai.environment.Environment`, plus an experiment run file.
3. **Wildfire Agent (GUARDIAN CMA)** — a cellular-automaton wildfire driven by an
   Overseer-Adversary parameter vector Θ, registered against the **full SoCal
   GIS footprint**, with an optional **PostGIS** persistence layer.
4. **5-day analysis** — a 120-hour Santa-Ana wildfire / grid co-simulation with
   KPIs, plots, and a written report demonstrating fire spread → grid impact.

> Author: Eric MSP Veith · License: GPL-3.0-or-later (see `LICENSE`).

---

## Repository layout

```
socal_grid/          Geo-referenced SoCal pandapower model + convergence recipe
  dispatch_and_run.py    "config D": strengthen + co-locate dispatch + Iwamoto-NR
  socal_grid.json        2,294-bus / 2,595-line model (EPSG:4326)
midas_socal/         MIDAS scenario (NOAA-enriched)
  socal_midas.yml        scenario definition (weather = NOAA, not DWD Bremen)
  weather/noaa_provider.py   writes NOAA weather in the MIDAS CSV schema
  prepare_midas.py, run_sim.sh
palaestrai_socal/    palaestrAI environment + experiment run file
  environment.py         SoCalWildfireEnvironment (sensors/actuators = Θ)
  experiment.yml         arsenAI/palaestrAI experiment run (phase_0_santa_ana_5day)
wildfire_cma/        GUARDIAN wildfire cellular automaton + damage mapper + PostGIS
  cma.py                 WildfireCMA (S, τ, D, Θ); ROS eq-6, spread eq-7
  gis.py                 SoCal raster: real SRTM DEM (OpenTopography) or synthetic fallback, bounds, fuel map
  damage.py              DamageMapper: bus/line → cell co-registration, asset removal
  postgis.py, postgis_load.py   PostGIS staging (raster, grid, fire perimeter)
data/                CAISO actuals, GeoJSON layers, PostGIS init SQL
  dem/                   real SRTM GL3 terrain: fetch_dem_tiles.py + socal_srtm_gl3.json
                         (the 71 MB socal_srtm_gl3.npz mosaic is git-ignored — regenerate it)
analysis/            5-day simulation driver + outputs (run_5day.py, *.png, report)
tests/               unit (cma, postgis, smoke) + slow system tests
docs/                MIDAS_INTEGRATION.md
docker-compose.yml   PostGIS 16 + GIS loader
.gitlab-ci.yml       CI/CD: lint → unit → system (manual) → simulate (manual)
```

---

## Quick start

```bash
# 1) install (core runtime)
pip install -r requirements.txt          # numpy/pandas/pandapower/palaestrai/...
# or for development + CI:
pip install -r requirements-dev.txt

# 2) fast tests (no grid load, < 5 s)
pytest -m unit

# 3) heavy system tests (loads the 5.9 MB grid, runs power flow)
pytest -m slow

# 4) the headline result: 5-day wildfire / grid co-simulation
python analysis/run_5day.py --max-steps 120 --outdir analysis
#   -> analysis/FIVE_DAY_ANALYSIS.md + four PNG figures + five_day_kpis.csv

# 5) (optional) animated timelapse: GIS fire spread + failed lines + SAIDI curve
python analysis/make_timelapse.py    # -> analysis/wildfire_timelapse.gif + .mp4
```

---

## Real terrain (SRTM GL3 via OpenTopography)

The environment runs on the **real Southern California elevation surface** so the
fire spread can be judged against actual topography (Transverse Ranges, Channel
Islands, Salton Trough, Mojave). Terrain is SRTM GL3 (~90 m) pulled from the
[OpenTopography Global DEM API](https://portal.opentopography.org/apidocs/).

The 71 MB mosaic (`data/dem/socal_srtm_gl3.npz`) is **not committed** — it is
regenerated from the API. The full SoCal footprint is too large for a single
proxied request, so `fetch_dem_tiles.py` tiles the bounding box into a 4×3 grid,
requests each tile as an Arc/Info ASCII grid, parses it with pure numpy (no
rasterio/GDAL), and stitches the tiles into one north-at-top mosaic:

```bash
# requires an OpenTopography API key (free): https://portal.opentopography.org/myopentopo
export OPENTOPOGRAPHY_API_KEY=...
python data/dem/fetch_dem_tiles.py        # -> data/dem/socal_srtm_gl3.npz (~71 MB) + .json
```

`wildfire_cma.gis.socal_from_srtm()` loads that cache, bilinearly resamples it
onto the 600×760 model grid, and derives coarse fuel classes from elevation
(`_fuel_from_dem`: ocean & alpine non-burnable; the SoCal grass → chaparral →
montane-timber gradient in between). **If the cache is absent the environment
falls back to `synthetic_socal()` automatically** (`use_real_dem` defaults to
`True` in `SoCalWildfireEnvironment` but degrades gracefully), so the simulation
and the fast CI tests run with or without the DEM binary present.

---

## Reproduce the 5-day result with palaestrAI

There are **two equivalent ways** to run the 5-day Santa-Ana episode. Both drive
the identical `SoCalWildfireEnvironment` (same grid, same GUARDIAN CMA, same Θ
schedule) and reach the same KPIs.

### A. Standalone driver (no broker needed — recommended first run)

The quickest way to see the headline result. `analysis/run_5day.py` instantiates
the palaestrAI environment directly and steps it through 120 hourly steps with a
deterministic Santa-Ana Θ schedule, then writes the report + figures:

```bash
pip install -r requirements.txt
python analysis/run_5day.py --max-steps 120 --outdir analysis
```

Expected result (seed 47, synthetic 600×760 raster):

| KPI | Value |
|-----|-------|
| Baseline served load | ~35,000 MW |
| Final served load | ~31,328 MW |
| Buses / lines de-energised | ~213 / ~327 |
| Peak customers disconnected | ~734,000 (≈ day 2.5) |
| Final burn footprint | ~166,000 cells (all burned out) |
| Power flow convergence | 120 / 120 steps |

### B. Full palaestrAI run (experiment run file + store/broker)

To run it as a *real* palaestrAI experiment — with the Overseer-Adversary agent,
the run-governor, and results written to the palaestrAI store — use the bundled
experiment run file `palaestrai_socal/experiment.yml`.

```bash
# 1) install the palaestrAI / MIDAS stack
pip install -r requirements.txt        # includes palaestrai, palaestrai-mosaik, midas-palaestrai

# 2) generate the NOAA weather CSV the scenario reads (one-off)
python midas_socal/prepare_midas.py

# 3) (first time only) initialise the palaestrAI database / config
palaestrai database-create            # creates the results store
#   palaestrai database-migrate        # if upgrading an existing store

# 4) validate the experiment run file loads against the schema
python -c "from palaestrai.experiment import ExperimentRun; \
           ExperimentRun.load('palaestrai_socal/experiment.yml'); \
           print('experiment OK')"

# 5) launch the experiment (one 5-day episode, 120 hourly steps)
palaestrai start palaestrai_socal/experiment.yml
```

What the experiment does (see `palaestrai_socal/experiment.yml`):

- **environment** `SoCalWildfireEnvironment` (uid `socal_wildfire`) — the SoCal
  grid + GUARDIAN CMA, `env_step_min=60`, `max_steps=120`, default ignition
  `[-118.13, 34.19]` (Eaton-fire-like origin).
- **agent** `overseer_adversary` — observes the 13 grid+fire sensors and acts
  through the 6 Θ actuators. Per the paper-faithful design it uses
  `DummyBrain`/`DummyMuscle`/`DummyObjective` (it **acts but does not learn**);
  the modelled mechanism is the constrained mutation, not a trained policy.
- **reward / objective** — `customers_disconnected` (the adversary maximises
  grid harm).
- **simulation** — `TakingTurnsSimulationController`, one episode (`episodes: 1`),
  raw actuators stored.

> **Note on equivalence:** the standalone driver (A) supplies the Santa-Ana Θ on
> a fixed meteorological schedule so the run is fully reproducible offline. The
> palaestrAI run (B) lets the adversary agent emit Θ each turn within the same
> actuator bounds. For the `DummyMuscle` adversary the dynamics match; swap in a
> learning brain to *optimise* the ignition/wind strategy against the grid.

The **Wildfire Agent / GUARDIAN CMA is documented in full in
[`docs/CMA_AGENT.md`](docs/CMA_AGENT.md)** — the spread physics, fuel model,
damage mapper, the Θ vector, and the Python API.

---

## The four deliverables

### 1. NOAA weather in the MIDAS scenario

`midas_socal/weather/noaa_provider.py` fetches/serialises NOAA weather into the
exact CSV schema the MIDAS `weather` simulator consumes (`socal_noaa_weather.csv`).
The scenario `midas_socal/socal_midas.yml` points its `weather` module at this
CSV instead of the bundled DWD Bremen data — so the replacement lives **in the
scenario**, with no new simulator. See `docs/MIDAS_INTEGRATION.md` for the wiring
and the data-path fix.

```bash
midas_socal/run_sim.sh            # full scenario
midas_socal/run_sim.sh --smoke    # short run (CI / quick check)
```

### 2. palaestrAI environment

`palaestrai_socal/environment.py` turns the SoCal MIDAS environment into a
`palaestrai.environment.Environment`. It exposes:

- **13 sensors** — grid + fire telemetry (`min/mean_bus_vm_pu`,
  `customers_connected/disconnected`, `saidi_minutes`, `fire_front_cells`,
  `fire_affected_cells`, `failed_buses`, `failed_lines`, `wind_*`,
  `grid_served_mw`, `pf_converged`).
- **6 actuators = Θ** — `ignition_lon`, `ignition_lat`, `kappa`,
  `dead_fuel_moisture`, `wind_speed`, `wind_dir_deg`.
- **reward** — `customers_disconnected` (the Overseer-Adversary maximises grid harm).

The environment builds a *converging, balanced* baseline via the proven
`socal_grid/dispatch_and_run.py` recipe before the wildfire mutates it.
`palaestrai_socal/experiment.yml` is a valid palaestrAI experiment run
(`phase_0_santa_ana_5day`, 120 hourly steps); validate it with
`ExperimentRun.load("palaestrai_socal/experiment.yml")`.

### 3. Wildfire Agent — GUARDIAN cellular automaton

`wildfire_cma/cma.py` implements the GUARDIAN four-tuple `(S, τ, D, Θ)`:

- **S** — cell states `UNBURNED / BURNING / BURNED_OUT`.
- **τ** — the cellular automaton: rate-of-spread (eq-6) and per-neighbour
  ignition probability (eq-7) over a Moore neighbourhood, with wind, slope, and
  fuel-moisture factors; burning cells burn out after `t_burn_steps`.
- **D** — `damage.DamageMapper` co-registers every grid bus and line to raster
  cells and removes assets the fire engulfs (with a radiant-heat clearance buffer).
- **Θ** — the Overseer-Adversary control vector (ignition, wind, fuel moisture,
  global ROS multiplier `κ`).

The landscape spans the **full SoCal footprint**
(`SOCAL_BOUNDS = (-121.3, 32.4, -113.7, 37.7)`). `gis.synthetic_socal()` builds a
self-contained synthetic raster; `gis.from_rasters()` loads real LANDFIRE/3DEP
rasters when `rasterio` is available.

> **Full documentation of the Wildfire Agent — the physics (eq. 6 / eq. 7), the
> fuel model, the damage mapper, the Θ vector, and the public API — is in
> [`docs/CMA_AGENT.md`](docs/CMA_AGENT.md).**

**PostGIS** (`wildfire_cma/postgis.py`) stages the raster, the grid, and fire
perimeters. Bring up a server with `docker-compose up` (PostGIS 16-3.4); the
`gis-loader` service loads the synthetic raster + grid via `postgis_load.py`.

### 4. Five-day analysis

`analysis/run_5day.py` drives the palaestrAI environment through a 120-hour
Santa-Ana episode (strong dry offshore wind days 1–3, marine-layer recovery
days 4–5) and produces:

- `five_day_kpis.csv` — full per-step KPI table.
- `fire_growth.png`, `grid_impact.png`, `saidi_voltage.png`,
  `fire_perimeter_day5.png`.
- `FIVE_DAY_ANALYSIS.md` — narrative report.

A representative run ignites in the LA basin and grows to ~166k affected cells,
de-energising ~213 buses / ~327 lines and dropping served load from 35,000 MW to
~31,300 MW (~734k customers disconnected at peak) — the explosive growth occurs
during the day-1–3 Santa-Ana window and plateaus as the front burns out.

---

## CI/CD

`.gitlab-ci.yml` runs **light CI on every push** and keeps the expensive work
behind manual triggers:

| Stage | Trigger | What it does |
|-------|---------|--------------|
| `lint` | push | flake8 (hard-fail on syntax/undefined; style advisory) |
| `unit` | push | `pytest -m unit` (cma, postgis, gis, smoke — numpy only, no grid, < 5 s) |
| `system` | **manual** | `pytest -m slow` (grid dispatch + power flow + palaestrAI env) |
| `simulate` | **manual** | NOAA MIDAS smoke run + the full 5-day co-simulation, published as artifacts |

---

## Provenance & notes

- Grid convergence depends on the `dispatch_and_run` "config D" recipe — raw
  `pp.runpp` on `socal_grid.json` diverges. Do not bump `numpy`/`pandas` without
  re-validating; palaestrAI constrains `numpy<2`, `pandas==2.1.4`.
- The GUARDIAN framing follows the attached paper: the wildfire is a *constrained
  mutation* of the grid driven by Θ, **not** a learning DRL agent.
- Modelled after the January 2025 Eaton/Palisades LA-basin fires (front regime
  ~60–80 m/min under Santa-Ana forcing).
