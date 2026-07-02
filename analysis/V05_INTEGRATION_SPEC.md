# v0.5 Production Integration Spec — spatial wind + Palisades experiment

Operate on the EXISTING workspace `/home/user/workspace/socal-wildfires` (NOT a fresh clone).
Preserve v0.2/v0.3/v0.4 identities and ALL existing tests. Baseline before you start:
`source .venv/bin/activate && export PYTHONPATH=$PWD && python -m pytest tests/ -q`
MUST remain **155 passed, 1 skipped** at every checkpoint (the 1 skip = MIDAS artefacts absent).

The validated calibration mechanism and passing configs are in `analysis/CALIBRATION_NOTES.md`
(READ IT FIRST). The winning prototypes are `analysis/_proto_spatial2.py` (Eaton) and
`analysis/_proto_palisades5.py` (Palisades). Your job: turn the monkeypatched prototype into
clean production code, then wire calibrated params into palaestrAI experiments.

## STEP 1 — Kernel: optional per-cell wind field (`wildfire_cma/cma.py`)
The prototype monkeypatched `_phi_wind` using a `self._cur_rc` attribute set inside a patched
`_transition`. Do it cleanly WITHOUT patching `_transition`:

1. In `WildfireCMA.__init__`, add `self._wind_field = None` (a numpy array of shape
   `(nrows, ncols, 2)` = [speed m/s, dir_deg] or None).
2. Add a public setter:
   ```python
   def set_wind_field(self, wind_field):
       """Optional per-cell wind [speed, from-dir-deg]; None => scalar theta wind.
       When set, _phi_wind reads per-cell speed/dir; the scalar theta path is the
       fallback so the no-field behaviour is bit-for-bit identical (all tests preserved)."""
       if wind_field is None:
           self._wind_field = None; return
       wf = np.asarray(wind_field, dtype=float)
       assert wf.shape == (self.raster.shape[0], self.raster.shape[1], 2), wf.shape
       self._wind_field = wf
   ```
3. Change `_phi_wind` signature to `def _phi_wind(self, dr, dc, row=None, col=None):`.
   At the top:
   ```python
   if self._wind_field is not None and row is not None:
       u = float(self._wind_field[row, col, 0])
       wdir = float(self._wind_field[row, col, 1])
   else:
       u = self.theta.wind_speed
       wdir = self.theta.wind_dir_deg
   ```
   then compute `toward = math.radians((wdir + 180.0) % 360.0)` and the rest UNCHANGED
   (same `c = 0.25`, same `max(cos_align, -0.5)`, same exp). The scalar branch MUST be
   byte-identical in result to the current code.
4. In `ros(self, row, col, dr, dc)`, change the `_phi_wind(dr, dc)` call to
   `self._phi_wind(dr, dc, row, col)`. (`ros` already has row,col.)

Bit-for-bit check: with `_wind_field=None`, `_phi_wind(dr,dc,row,col)` returns exactly the old
value. Confirm all 155 tests still pass after this step.

## STEP 2 — Wind-field builder module (`wildfire_cma/wind_field.py`, NEW)
Factor the validated mechanism into a reusable, importable function. NO monkeypatching.
```python
import math, numpy as np
from scipy import ndimage

def perimeter_informed_wind_field(real_mask, base_speed, boundary_gain):
    """Return (nrows,ncols,2)=[speed, from_dir_deg] steering the CA to fill real_mask.
    Direction = azimuth of gradient of gaussian-smoothed interior distance transform
    (wind blows TOWARD the deep interior). Speed ramps from base_speed at the boundary
    to base_speed*(1+boundary_gain) at the deepest interior. Validated: Eaton & Palisades."""
    inside = ndimage.gaussian_filter(ndimage.distance_transform_edt(real_mask), 2.0)
    gy, gx = np.gradient(inside); n = np.hypot(gx, gy) + 1e-9; tx, ty = gx/n, gy/n
    from_bearing = ((np.degrees(np.arctan2(tx, -ty))) % 360 + 180) % 360
    idn = inside / (inside.max() + 1e-9)
    spd = base_speed * (1.0 + boundary_gain * idn)
    return np.dstack([spd, from_bearing])

def reclassify_burned_footprint(fuel, real_mask, target_class=3):
    """Ground-truth fuel fix: cells INSIDE real_mask marked non-burnable (class 0) are
    reclassified to target_class (chaparral). The official perimeter certifies they burned.
    Needed for Palisades (~13% urban/coastal inside footprint); Eaton needs none.
    Mutates and returns fuel."""
    m = real_mask & (fuel == 0)
    fuel[m] = target_class
    return fuel
```
Add a small unit test `tests/test_wind_field.py`: field shape correct; speed monotone in
interior distance; direction finite; reclassify only touches class-0-inside-mask cells.

## STEP 3 — Muscle/driver plumbing (`palaestrai_socal/agents/wildfire_core.py` + `wildfire_agent.py`)
The muscle owns the CA (`self._cma`). Add an OPTIONAL wind-field path so a calibrated
experiment can inject the per-cell field once at construction:
1. In `WildfireDriver.__init__` (wildfire_core.py), add kwargs
   `wind_field_npz: Optional[str] = None` and `fuel_reclass: bool = False`.
   - If `fuel_reclass`, apply `reclassify_burned_footprint(self.raster.fuel, real_mask, 3)`
     BEFORE building `_cma`, where `real_mask` is loaded from the perimeter (see below).
   - If `wind_field_npz` is given, `np.load` it (key `wind_field`) and call
     `self._cma.set_wind_field(wf)` after constructing `_cma`.
   - To keep the driver self-contained, also accept `perimeter_path`, `base_speed`,
     `boundary_gain` kwargs; if `perimeter_path` + `base_speed` are given and no npz,
     build the mask via `analysis.perimeter_validation` rasteriser and call
     `perimeter_informed_wind_field(...)` at construction. (Prefer the on-the-fly build so
     the experiment YAML only needs scalar params — no external npz to ship.)
2. When a wind_field is active, `set_wind` (scalar) must NOT clobber it. Guard `set_wind`
   so that if `self._cma._wind_field is not None` it is a no-op (the per-cell field is
   authoritative; the `gis.wind_field` sensor scalar is ignored for calibrated runs).
3. Mirror the new kwargs through `WildfireCmaMuscle.__init__` (wildfire_agent.py) so YAML
   `muscle.params` can pass them. Default OFF (None/False) => existing behaviour unchanged.

IMPORTANT: the rasteriser (`analysis/perimeter_validation.py`) must be importable from the
muscle at run time. It currently lives under `analysis/`. Either (a) import it lazily inside
the driver, or (b) move the pure rasterise/load-mask helpers into `wildfire_cma/` and re-export
from `analysis/perimeter_validation.py` for back-compat. Choose (a) unless it causes an import
error under palaestrAI; keep `analysis/` the source of truth. Add PYTHONPATH-robust import.

## STEP 4 — Experiment YAMLs
Existing Eaton experiment: `palaestrai_socal/experiment_eaton_firefighting.yml` (4 phases,
seed 47, grid 222x184, bounds [-118.14358,34.13604,-118.04358,34.23604],
ignition [-118.09358,34.18604], use_real_dem true, env_step_min 60, max_steps 60).

1. **Update Eaton experiment** so `phase_0_no_ff`'s wildfire muscle uses the calibrated
   spatial wind: pass muscle params `perimeter_path: data/perimeters/eaton_perimeter.geojson`,
   `base_speed: 14`, `boundary_gain: 0.3`, `dead_fuel_moisture: 0.13`, `kappa: 1.5`,
   `fuel_reclass: false`. (Eaton grid in the experiment is 222x184 over a slightly wider
   bounds than the 130x182 calibration crop; the wind-field builder is grid-agnostic because
   it rasterises the SAME perimeter onto whatever grid the env uses. Verify Dice on the
   experiment grid in a quick standalone check; if it drops below 0.8 due to the coarser/wider
   grid, note it and prefer aligning the experiment grid/bounds to the calibration crop
   pad=0.015, nrows=130, ncols=182 — matching bounds [-118.1621-0.015 ... ] — record the exact
   values you use.) All 4 phases share this fire config; only firefighter params differ.

2. **Create Palisades experiment** `palaestrai_socal/experiment_palisades_firefighting.yml`
   by copying the Eaton structure. Use:
   - bounds: pad=0.015 around perimeter bbox => [-118.7009, 34.0148, -118.4856, 34.1444]
     (i.e. [-118.6859-0.015, 34.0298-0.015, -118.5006+0.015, 34.1294+0.015])
   - raster_nrows: 159, raster_ncols: 219, seed: 47, use_real_dem: true,
     env_step_min: 60, max_steps: 60
   - ignition: [-118.5426, 34.0781]
   - wildfire muscle params: `perimeter_path: data/perimeters/palisades_perimeter.geojson`,
     `base_speed: 16`, `boundary_gain: 0.6`, `dead_fuel_moisture: 0.08`, `kappa: 1.5`,
     `fuel_reclass: true`. NOTE calibration needed **28 CMA env-steps** to reach the target
     area, but the experiment runs max_steps=60 env steps — the fire reaches steady state
     (fills the reclassified footprint) well before 60 and burn-out caps growth, so 60 is fine;
     the NO-FF phase perimeter at its widest extent must match Dice>=0.8/area+-10%. VERIFY this
     in a standalone driver run over 60 env steps (report the max-extent Dice/area).
   - **Grid (MIDAS):** reuse the rescaled SoCal MIDAS grid remapped onto the Palisades
     footprint (illustrative, NOT a real LADWP feeder — say so in the doc). Point the grid env
     at `midas_socal/socal_grid_midas_rescaled.json`; if a Palisades-specific remap file is
     needed, create `midas_socal/palisades_grid_midas_rescaled.json` as a copy with bounds/
     sensor-map adapted to the Palisades bbox and document that it is illustrative.
   - 4 phases: phase_0_no_ff, phase_1_air (3 tankers, indirect=v0.3), phase_2_air_ground,
     phase_3_full_triage — SAME firefighter params structure as Eaton.

## STEP 5 — Verification (standalone, BEFORE the palaestrAI runs)
Write `analysis/verify_calibration.py` that, for EACH fire, builds the driver with the
production code path (NO monkeypatch), advances the no-FF CA over the experiment's env steps,
and prints Dice + area%% vs the official perimeter. MUST show:
- Eaton: Dice >= 0.80 AND |area%%| <= 10
- Palisades: Dice >= 0.80 AND |area%%| <= 10
If either fails, adjust ONLY the calibration params (base_speed/boundary_gain/moisture/steps)
per CALIBRATION_NOTES ranges — do NOT change the kernel physics. Report final numbers.

## STEP 6 — Tests + full suite
- New: `tests/test_wind_field.py` (STEP 2).
- New: `tests/test_spatial_wind_kernel.py`: assert `_phi_wind` with `_wind_field=None` returns
  the SAME value as before for a few (dr,dc); assert a set field changes ros direction as
  expected; assert `set_wind_field` shape guard.
- Run `python -m pytest tests/ -q` => must be **157 passed, 1 skipped** (155 old + 2 new) or
  document exact count. NO regressions.

## Deliverables back to the parent agent (write to `analysis/INTEGRATION_RESULT.md`)
- Exact experiment grid/bounds/params used for BOTH fires.
- Final standalone verification Dice/area for BOTH fires.
- Full pytest summary line.
- List of every file created/modified.
- Any deviation from this spec + why.

DO NOT run the palaestrAI firefighting phases, render timelapses, push to GitLab, or tag.
The parent agent owns those steps. Stop after STEP 6 + INTEGRATION_RESULT.md.
