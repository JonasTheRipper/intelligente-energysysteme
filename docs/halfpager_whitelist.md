# Pruning the Sensor Interface: a 6.7x per-step speedup with bit-identical trajectories in co-simulated grid RL

## Problem

Reinforcement learning on coupled infrastructure requires co-simulation. Our
testbed joins a MIDAS/mosaik/pandapower model of a 2,294-bus Southern California
distribution grid to a wildfire cellular automaton, exchanging state through
palaestrAI's ARL sensor/actuator interface.

That interface advertises **66,223 sensors and 11,250 actuators** — every bus
voltage, line flow and load attribute, republished every simulation step. The
agents subscribe to **31** of them. The remaining 99.95% are constructed,
serialised, shipped across the message bus and discarded, once per step, for the
entire length of training.

## Method

A whitelist (`keep_sensors` / `keep_actuators`, fnmatch patterns) prunes the
sensor and actuator lists *before* mosaik wires the entities, so unsubscribed
sensors are never created at all. In the configuration studied here it retains
6,822 sensors (89.7% removed) and 1,932 actuators.

## Why pruning could be unsafe, and why it is not

Sensors are read-only reports, so removing them cannot change the simulation *by
construction* — but that is an argument, not a measurement. Two properties make
the argument sound rather than merely plausible:

1. palaestrAI validates every agent's subscription against the sensors the
   environments actually advertise, at setup, and raises before the first step
   on any mismatch. A run that starts therefore has full agent observability by
   construction: the whitelist cannot silently starve a learner, it can only
   abort the run.
2. Pruning preserves the relative order of surviving sensors, so the summation
   order in reward computation — and hence its floating-point result — is stable.

## Validation

Four arms, generated from a single source run document by exactly one documented
mutation each, over 4 scripted phases x 60 steps:

| comparison | purpose | cells differing |
|---|---|---|
| OFF vs OFF2 | reproducibility control | **0** |
| OFF vs ON | the treatment | **0** |
| NEGB vs NEG (+3.6% wind speed) | sensitivity control | 360 |

The sensitivity control is what gives the two zeros meaning. Without it,
"identical" reported twice is equally consistent with a comparison that cannot
detect anything at all.

Compared per step: the SHA-256 of the fire raster, the structural telemetry, and
all 1,932 stored load sensors. Agreement is exact, not within a tolerance.

Equivalence of the arms was verified from the **stored** run documents
(`experiment_runs.document_json`), not the configuration files: OFF and ON
differ in exactly ten leaves — the run uid, the derived experiment uid, and
`keep_sensors`/`keep_actuators` in each of the four phases. No simulation code
changed during the measurement window.

## Results

Running each arm at two lengths (8 and 60 steps) determines
`total = 4 x (startup + steps x per_step)` exactly, separating fixed cost from
marginal cost:

| | startup per phase | per step |
|---|---|---|
| whitelist OFF | 53.8 s | 9.77 s |
| whitelist ON | 53.4 s | 1.45 s |
| speedup | 1.01x | **6.73x** |

The whitelist does not improve startup at all. The entire gain is per-step, which
is the term that matters: the end-to-end 4.6x observed at 60 steps is a blend of
the two and is *not* a property of the system — it approaches 6.73x as episodes
lengthen. Wall-clock noise, measured between two bit-identical runs, was 0.6%.

## Limitations

- **Scripted policies only.** The DRL phases include an agent that samples an
  OS-entropy-seeded action space, so those runs are not reproducible against
  themselves and bit-comparison does not apply. Equivalence for learners rests on
  the structural argument above, not on measurement.
- **Projection.** The store retains only load sensors (identical transformer in
  both arms), so equivalence is established on the outcome-relevant projection —
  which is the one the grid-outage metric is computed from.
- **One scenario, one seed.**

## Takeaway

Co-simulation interfaces built by enumeration rather than by subscription impose
a large, silent, per-step cost. Pruning the interface to what agents actually
read is a configuration change, is verifiable to bit-identity against a
sensitivity control, and here returned 6.7x of simulation throughput — the
difference between a 28-hour and a 4-hour training campaign.

---

### Reproduction

```bash
python analysis/make_whitelist_configs.py   # generate the arms
./run_whitelist_validation.sh               # ~2 h, 6 arms, 3 comparisons
python analysis/whitelist_figure.py         # both figures
```

Controls are enforced in the runner: it aborts before reporting the treatment if
the reproducibility control diverges or the sensitivity control does not.
