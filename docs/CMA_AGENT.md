# The Wildfire Agent — GUARDIAN Constrained-Mutation Automaton (CMA)

This document describes the "Wildfire Agent" implemented in
[`wildfire_cma/`](../wildfire_cma). It is the GUARDIAN-paper wildfire, realised
as a **Constrained-Mutation Automaton (CMA)** that mutates the power grid as a
fire spreads across a California landscape raster.

> **Design choice (paper-faithful):** the wildfire is *not* a learning DRL
> agent. It is a **constrained mutation operator** on the grid topology, driven
> by an Overseer-Adversary parameter vector Θ. In the palaestrAI experiment the
> adversary uses a `DummyBrain`/`DummyMuscle` (it acts but does not learn); all
> the "intelligence" lives in the constrained mutation dynamics described here.

---

## 1. The four-tuple

The CMA is the GUARDIAN four-tuple

```
CMA = (S, τ, D, Θ)
```

| Symbol | Name | Implementation |
|--------|------|----------------|
| **S** | Cellular state grid | `WildfireCMA.state` — an `int8` array co-registered with the raster `R`. Values: `UNBURNED=0`, `BURNING=1`, `BURNED_OUT=2`. |
| **τ** | Transition function `S × R × Θ → S` | `WildfireCMA.step()` — the cellular-automaton update using a Rothermel-style rate-of-spread (eq. 6) and ignition probability (eq. 7) over a Moore neighbourhood. |
| **D** | Damage mapper `S × G → ΔG` | `wildfire_cma/damage.py::DamageMapper` — fails grid **buses** whose cell is burning/burned and overhead **lines** whose footprint falls within the radiant-heat clearance buffer of the fire. |
| **Θ** | Overseer-Adversary parameters | `wildfire_cma/cma.py::Theta` — ignition point(s), wind vector, dead-fuel moisture, and the global ROS multiplier `κ`. |

The raster stack `R` (fuel class, DEM, optional canopy) is co-registered with
the grid `G` in EPSG:4326 and built by `wildfire_cma/gis.py`.

---

## 2. The state grid `S`

`state` is an `(nrows, ncols)` array. Row 0 is the **north** edge (max latitude).
Each cell carries one of three states; a burning cell holds a `burn_timer` and
becomes `BURNED_OUT` after `t_burn_steps` CMA sub-steps (a proxy for fuel heat
content / residence time).

```python
from wildfire_cma.cma import UNBURNED, BURNING, BURNED_OUT
```

---

## 3. The transition function `τ` (the cellular automaton)

Each CMA sub-step (`dt_cma_min`, default 5 min), every `BURNING` cell attempts
to ignite its eight Moore neighbours.

### Rate of spread — eq. 6

```
R(i,j → i',j')  =  κ · R⁰(i,j) · φ_w(u, θ) · φ_s(z, θ)
```

* **`R⁰(i,j)`** — no-wind / no-slope base ROS [m/min] for the cell's fuel class
  (`BASE_ROS_BY_FUEL`), damped by dead-fuel moisture via a Rothermel moisture
  coefficient `η_m`. Above the fuel's *moisture of extinction*
  (`FUEL_MX_EXTINCTION`) the ROS is **0** — the fire cannot spread.
* **`φ_w`** — wind factor. The wind vector (blowing *toward* `wind_dir_deg+180°`)
  is aligned with the spread bearing; an exponential mid-flame coefficient
  (`exp(c·u·cos_align)`, `c=0.25`) is tuned so an extreme Santa-Ana (~20 m/s)
  produces the 60–80 m/min fronts reported for the January 2025 LA fires.
* **`φ_s`** — slope factor (`1 + 5.275·tan²(slope)·upslope_align`); fire runs
  faster uphill, using the DEM gradient and aspect.
* **`κ`** — the Overseer-Adversary's global ROS multiplier (1 ≤ κ ≤ 8).

### Spread probability — eq. 7

```
p  =  1 − exp( −R · dt_CMA / δ )
```

where `δ` is the cell size in metres (scaled by √2 for diagonal neighbours). A
neighbour ignites if `rng.random() < p`. This is the only stochastic element;
it is seeded for reproducibility.

### Fuel classes

`BASE_ROS_BY_FUEL` maps a coarse fuel-class id to a baseline ROS:

| id | fuel | base ROS [m/min] |
|---:|------|-----------------:|
| 0 | non-burnable (water, urban, barren, ag) | 0.0 |
| 1 | grass / GR | 4.0 |
| 2 | grass-shrub / GS | 2.5 |
| 3 | shrub / chaparral SH (dominant SoCal fuel) | 3.2 |
| 4 | timber-understory TU | 1.2 |
| 5 | timber-litter TL | 0.8 |
| 6 | slash-blowdown SB | 1.5 |

`gis.fbfm40_to_class()` maps the LANDFIRE 40-Scott-Burgan fuel models to these
classes when real rasters are loaded.

---

## 4. The damage mapper `D`

`DamageMapper(net, raster, clearance_m=120)` co-registers the grid to the raster
**once** (cached):

* each **bus** → its raster cell (from the bus `geo` Point);
* each **line** → the set of cells its `geo` LineString crosses.

On `evaluate(cma)` it returns a `DamageState`: the buses whose cell is on fire,
and the lines any of whose cells are within the radiant-heat clearance buffer.
`apply(net)` then sets `in_service=False` on the failed buses (plus their
attached loads/sgens/gens) and the failed lines — that is the **mutation** of
the grid `G` that the subsequent power flow sees.

In the SoCal model this co-registration covers **100% of the 2,294 buses and
2,595 lines**.

---

## 5. The Overseer-Adversary vector `Θ`

```python
from wildfire_cma.cma import Theta

theta = Theta(
    ignition_points=[(-118.13, 34.19)],  # GEOGRAPHIC (lon, lat), EPSG:4326
    # ignition_rc=[(row, col)],          # OR pin an exact raster cell (tests)
    wind_speed=20.0,        # m/s   (NOAA-sourced when driven by the env)
    wind_dir_deg=45.0,      # meteorological direction the wind blows FROM
    dead_fuel_moisture=0.03,# fraction (Santa-Ana ≈ 0.03–0.08)
    kappa=3.0,              # global ROS multiplier (1–8)
)
theta.clamp()               # geophysical-plausibility filter
```

`clamp()` keeps every parameter inside physically valid bounds — this is the
"constrained" in *constrained mutation*. The six fields map **one-to-one** onto
the six palaestrAI actuators (`ignition_lon`, `ignition_lat`, `kappa`,
`dead_fuel_moisture`, `wind_speed`, `wind_dir_deg`).

---

## 6. Using the CMA directly

```python
from wildfire_cma.cma import Theta, WildfireCMA
from wildfire_cma.gis import synthetic_socal

# 1. build a co-registered SoCal raster (synthetic, fully offline)
raster = synthetic_socal(nrows=600, ncols=760, seed=47)   # δ ≈ 947 m/cell

# 2. ignite under a Santa-Ana Θ
theta = Theta(ignition_points=[(-118.13, 34.19)],
              wind_speed=20.0, wind_dir_deg=45.0,
              dead_fuel_moisture=0.03, kappa=3.0)
cma = WildfireCMA(raster, theta, dt_cma_min=5.0, t_burn_steps=6, seed=47)

# 3. advance one wall-clock hour and inspect
cma.advance(minutes=60)
print(cma.stats())
# -> {'step':12, 'burning_cells':..., 'burned_cells':..., 'affected_cells':...,
#     'fraction_burned':..., 'front_size':...}

mask = cma.fire_mask()           # bool array of cells ever on fire
cma.ignite_lonlat(-117.9, 34.1)  # add another ignition mid-run
```

### API summary

| Member | Purpose |
|--------|---------|
| `WildfireCMA(raster, theta, dt_cma_min, t_burn_steps, seed)` | construct + ignite |
| `.step()` | one CMA sub-step (`τ`) |
| `.advance(minutes)` | run `≥1` sub-steps to cover wall-clock time |
| `.ros(r, c, dr, dc)` | eq. 6 rate of spread [m/min] |
| `.spread_prob(r, c, dr, dc, diagonal)` | eq. 7 ignition probability |
| `.ignite_lonlat(lon, lat)` | add a geographic ignition |
| `.stats()` | front/affected/burned cell counts |
| `.fire_mask()` | boolean burned-or-burning mask |

---

## 7. GIS landscape

`wildfire_cma/gis.py` defines the **full SoCal footprint**

```python
SOCAL_BOUNDS = (-121.3, 32.4, -113.7, 37.7)   # (minlon, minlat, maxlon, maxlat)
```

covering the SCE, LADWP and SDG&E service areas.

* `synthetic_socal(nrows, ncols, bounds, seed)` — a self-contained, deterministic
  fuel + DEM raster (no external data, ideal for CI and reproducible runs).
* `from_rasters(...)` — load real **LANDFIRE** fuel + **3DEP** DEM rasters
  (requires `rasterio`; install via the `gis` extra).
* `fbfm40_to_class()` — LANDFIRE FBFM40 → coarse fuel class.

### Optional PostGIS persistence

`wildfire_cma/postgis.py` stages the raster, the grid, and fire perimeters in a
PostGIS database (tables `raster_meta`, `fuel_cells`, `grid_bus`, `grid_line`,
`fire_perimeter`). Bring one up with the bundled compose file:

```bash
docker-compose up -d            # PostGIS 16-3.4 + a one-shot gis-loader
```

The CMA itself never requires PostGIS — it is an optional staging/inspection
layer for working with real shapefiles.

---

## 8. How the environment drives the CMA

Inside `SoCalWildfireEnvironment` (see the [README](../README.md)) each
environment macro-step (`env_step_min`, default 60 min):

1. reads the adversary's actuators and rebuilds `Θ` (then `Θ.clamp()`);
2. applies any ignition update;
3. calls `cma.advance(env_step_min)` to spread the fire (`τ`);
4. runs the damage mapper `D` and mutates the grid;
5. solves the power flow and emits the 13 grid+fire sensors and the
   `customers_disconnected` reward.

This is the loop exercised by the 5-day analysis and by the palaestrAI
experiment run.

---

## 9. Tests

| Test | Marker | What it checks |
|------|--------|----------------|
| `tests/test_cma.py` | unit | Θ clamp; lon/lat ignition; spread + burnout; firebreak blocks; wind directionality (SW > NE); high-moisture extinguishes |
| `tests/test_smoke.py` | unit | import graph + small-raster spread (CI smoke) |
| `tests/test_damage_mapper.py` | slow | 100% grid co-registration; ignition fails buses + lines; mutated grid still solves |

```bash
pytest -m unit            # fast (cma, smoke, postgis)
pytest tests/test_cma.py  # just the CMA unit tests
```
