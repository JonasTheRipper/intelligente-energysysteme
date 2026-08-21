# Finding: the scripted firefighting doctrines collapse to one policy

*Observed 2026-08-21 in the whitelist validation run (`wlval_off`, 4 scripted
phases x 60 steps, `raster_seed: 47`). Not investigated yet — recorded so it is
not rediscovered from scratch.*

## What was seen

The four scripted phases configure materially different fleets:

| phase | fleet | doctrine |
|---|---|---|
| `phase_0_no_ff` | none | — |
| `phase_1_air` | 3 planes | indirect |
| `phase_2_air_ground` | 3 planes + 4 crews + 2 dozers | auto |
| `phase_3_full_triage` | 3 planes + 4 crews + 2 dozers + 3 engines | auto, `protect_assets: true` |

They produce:

| phase | burned cells | houses lost |
|---|---|---|
| `phase_0_no_ff` | 6980 | 30 / 101 |
| `phase_1_air` | 3888 | 10 / 101 |
| `phase_2_air_ground` | **3888** | **10 / 101** |
| `phase_3_full_triage` | **3888** | **10 / 101** |

`phase_1_air` and `phase_2_air_ground` are **bit-identical at every one of the
60 steps** (same SHA-256 of `gis.cell_state` throughout). Adding four hand crews
and two dozers, and switching the doctrine from `indirect` to `auto`, produced
not one differing cell mutation.

`phase_3_full_triage` differs only by 9 cells in the engines' point-protection
state; its burned extent and house losses are identical to the other two.

So the effective action space is **"3 planes, or nothing"** (6980 -> 3888
burned). Every resource beyond the three aero tankers is inert.

## Why it matters

This is not a whitelist issue — it reproduces identically with the whitelist on
and off, and does not affect that result.

It matters for the **MOO vs SAIDI** experiment. The DRL firefighter selects
among four doctrines via `_DOCTRINE_MAP` (`firefighter_drl_agent.py`). If three
of those four are indistinguishable in outcome, the learner's only real lever is
*act vs. do nothing*, and **no objective function can differentiate policies that
produce identical trajectories**. A null result in the A/B would then say nothing
about MOO vs SAIDI — it would be a property of the action space.

Check this before concluding anything from the A/B.

## Where to look

* `palaestrai_socal/agents/firefighting/resources.py` — the `capacity()` methods
  for `HandCrews` / `Dozers`. At 90 m cells and a 60 min step, do they return a
  non-zero cell count? `_slope_derate` and the wind/grounding gates are the
  likely suspects.
* `palaestrai_socal/agents/firefighting/planner.py` — `IncidentCommand.propose`
  and `_merge`. If the tanker group already claims every candidate cell, the
  ground tactics' mutations may be deduped away by `STATE_PRIORITY`.
* `firefighter_core.py` `select_retardant_line` — whether the containment
  margin (`containment_margin: 2`) already arrests the fire at the real
  perimeter, leaving the marginal resources nothing to do. If so the collapse is
  *physical* rather than a bug, and the A/B needs a scenario where suppression
  is not already saturated.

## Reproduce

```bash
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, "analysis")
import store_readers as sr
PG = "postgresql://palaestrai:socal_local@127.0.0.1:5433/wlval_off"
con, ph = sr._connect(PG)
for phase in ("phase_0_no_ff", "phase_1_air", "phase_2_air_ground", "phase_3_full_triage"):
    rows = sr._fetch_env_rows(con, "gis_world", ph, phase, None, None)
    st = np.asarray(sr._sensors_by_suffix(rows[-1][1])["gis.cell_state"])
    print(phase, int(np.count_nonzero(np.isin(st, (1, 2)))))
con.close()
PY
```
