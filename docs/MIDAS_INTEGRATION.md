# MIDAS Co-Simulation of the Southern California Grid

This document explains how the geo-referenced SoCal pandapower model (Phase 1)
was wrapped into a complete [MIDAS](https://pypi.org/project/midas-mosaik/)
co-simulation scenario, driven by **real CAISO actuals**, and run for a full
24-hour day at 15-minute resolution.

MIDAS (the *MultI-Domain test scenario for Agent-based Smart grids*, built on
the mosaik co-simulation framework by OFFIS) couples a set of domain
simulators — a power grid, time-series data providers, a weather model, a time
simulator and a database store — and steps them together. Our job was to make
the SoCal grid one of those coupled simulators and feed it realistic demand and
generation profiles.

---

## 1. What the scenario contains

The scenario (`socal_midas.yml`) wires together five MIDAS modules:

| Module | Scope | Role |
|--------|-------|------|
| `store` | — | Writes every element's results to `_outputs/socal_midas.csv` each step. |
| `timesim` | `socal` | Advances simulated wall-clock time from `2024-07-16 00:00 -0700`. |
| `powergrid` | `socal` | Runs `pp.runpp()` on the SoCal pandapower net every step. |
| `weather` | `bremen` | DWD Bremen weather (included for realism — see note below). |
| `powerseries` | `loads`, `sgens` | Two instances that drive every load and sgen element from CSV time series. |

- **Horizon:** `end: 1*24*60*60` seconds with `step_size: 15*60` → **96 steps**.
- **Grid:** 2,294 buses, 2,595 lines, 345 transformers, 6 interties
  (`ext_grid`), 35 GW population-weighted peak demand.

### Weather note

MIDAS ships only the German DWD **Bremen** weather dataset. It is included in
the scenario for completeness/realism of the co-simulation, but it does **not**
drive the SoCal load or generation. All demand and renewable dynamics come
directly from the real CAISO time series for 2024-07-16. (Swapping in a
California TMY/NSRDB weather file is a straightforward future extension.)

---

## 2. Data flow — real CAISO actuals into the model

`prepare_midas.py` is the bridge. It:

1. **Loads the CAISO actuals** (`data/caiso_2024-07-16.csv`, 5-min) and
   resamples them to 96 × 15-min steps, producing normalized day *shapes* for
   system demand, solar, and wind.
2. **Loads & strengthens the Phase-1 grid** (`model/socal_grid.json`), then
   converts the PV `gen` rows into zero-injection `sgen`s and drops the `gen`
   table — this removes voltage-control conflicts between PV plants and the
   interties.
3. **Builds two wide CSV time-series tables** (one column per power value):
   - `socal_load_ts.csv` — for every load bus a `load_p_bus_<b>` and
     `load_q_bus_<b>` column. Active power follows the CAISO demand shape scaled
     to that bus's peak MW; reactive power is `p × 0.31` (cos φ ≈ 0.95).
   - `socal_sgen_ts.csv` — for every renewable sgen a `p`/`q` column pair
     (solar→CAISO solar shape, wind→CAISO wind shape, battery→evening discharge),
     **plus** one local/conventional generator per load bus (`sgen_LOCAL_*`).
4. **Writes combined mapping files** (`load_mapping.json`, `sgen_mapping.json`)
   of the form `{bus: [[[p_col, q_col], scale], ...]}`. MIDAS resolves each bus
   to the grid's load/sgen element and drives both its `p_mw` and `q_mvar`
   straight from those CSV columns every step.

So the **dynamics are real CAISO data**; the **magnitudes and topology** come
from the 35 GW SoCal model.

---

## 3. The convergence recipe (why it solves on all 96 steps)

MIDAS calls a bare `pp.runpp(net, numba=...)` with a flat start and a 10-iteration
limit every step — no robust/continuation solver. A 2,294-bus grid only
converges under those conditions if it is dispatched carefully. The baked-in
recipe, verified to converge on **all 96 steps (0 failures)**:

- **Loads:** `p = demand_shape[t] × bus_peak`; `q = p × 0.31`.
- **Renewables:** `p = min(shape[t] × nameplate, CAP)` with a per-bus hosting
  cap `CAP = 2.0 × bus_peak_load + 5 MW`; `q = 0`.
  *The cap is essential:* 14 renewable sites sit on radial degree-1 stubs;
  injecting GW there with no local load causes a voltage-rise divergence.
- **Local generators** (one per load bus):
  `p = max(0.95 × bus_load − renewables_at_bus, 0)`;
  `q = 0.85 × bus_reactive_load`.
  *The local reactive support is the critical ingredient* — without it the
  minimum bus voltage collapses to ≈0.73 pu; with it the grid holds ≈0.99 pu.

The 6 interties (`ext_grid`) then carry only network losses plus the small
balancing residual.

### Key technical detail: supplying reactive power through MIDAS

The MIDAS `powerseries` module's `CustomTimeSeries` path only reliably reads
active power. To inject the local reactive support, we use the module's
**combined mapping** instead: a two-element column entry `[p_col, q_col]`
creates a `CombinedTimeSeries` that reads **both** `p_mw` and `q_mvar` directly
from the CSV with no cos-φ calculation. That is why the YAML sets
`use_custom_time_series: false`, `calculate_missing_power: false`, and uses
`combined_mapping_filename` / `prefer_combined_mapping_from_file: true`.

---

## 4. How to run it

```bash
# 1. (re)generate the grid JSON, time-series CSVs and mapping files for a date
cd midas_socal
python prepare_midas.py 2024-07-16

# 2. run the full day (96 steps); outputs land in ../_outputs/
cd ..
midasctl run socal_midas -c midas_socal/socal_midas.yml --skip-download

# 3. analyze the result database
midasctl analyze _outputs/socal_midas.csv      # add -f for the full report
```

Outputs:
- `_outputs/socal_midas.csv` — the full results database (every element, every step).
- `_outputs/socal_midas/socal_midas-Powergrid-0_report.{md,odt}` — the analysis report.
- `_outputs/socal_midas/Powergrid-0/*.png` — voltage, intertie, load/sgen plots.

---

## 5. Result highlights (CAISO 2024-07-16)

- **Power flow converged on all 96 steps** — no divergence anywhere in the day.
- **Bus health: 99.41 %** of all bus-voltage samples inside the ±0.04 pu band;
  the network-mean voltage stayed ≈1.005–1.013 pu the whole day.
- **Active energy demand 667,650 MWh** vs **supply 696,672 MWh**
  (104.35 % sufficiency — generation slightly exceeds demand, the surplus
  absorbed by the interties).
- **Generation mix:** local/conventional ≈622,838 MWh; renewables ≈29,477 MWh
  served after the per-bus hosting cap (solar 19,374 / wind 7,218 / battery
  2,885 MWh). Renewable peak ≈2.4 GW — the synthetic backbone's realistic
  hosting capacity at this topology.
- **Interties** net-exported the small local surplus, ranging roughly
  −490 to −1,800 MW across the day, dipping during the midday solar peak.

---

## 6. Files

```
midas_socal/
  prepare_midas.py        # CAISO -> MIDAS inputs (grid JSON, CSVs, mappings)
  socal_midas.yml         # the MIDAS scenario definition
  socal_grid_midas.json   # generated MIDAS-ready grid (reference copy)
  load_mapping.json       # generated bus->[p,q] combined mapping (reference copy)
  sgen_mapping.json       # generated bus->[p,q] combined mapping (reference copy)
  MIDAS_INTEGRATION.md    # this document
  run_sim.sh              # convenience launcher used for the long run

midas_results/
  socal_midas_powergrid_report.md   # the analyze() powergrid report
  socal_midas_weather_report.md     # the analyze() weather report
  plots/                            # key result plots (PNG)
```

The generated grid JSON, CSV time series and mapping files are also written into
the MIDAS data path (`~/.config/midas/midas_data/`) where the `powerseries`
module looks for them at run time.
