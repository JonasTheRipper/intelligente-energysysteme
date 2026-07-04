"""Unit tests for the shared v0.7 DRL firefighter contract (firefighter_drl).

These pin the observation/action/reward *contract* that both the offline
teacher harvester and the online learning muscle depend on. They are numpy-only
(no palaestrai, no torch, no store), so they run in the lightweight CI ``unit``
stage: they exercise ``extract_obs`` dimensionality/bounds, the individual
feature encoders (fractions, front detection, slope, wind trig), the doctrine
inference from teacher mutations, and the resource-availability wind gate.
"""

import math

import numpy as np

from palaestrai_socal import spaces
from palaestrai_socal.agents import firefighter_drl as drl


def _grid(nr=12, nc=12):
    return np.full((nr, nc), spaces.UNBURNED, dtype=np.int8)


def _obs_kwargs(**over):
    kw = dict(
        state=_grid(),
        fuel=None,
        dem=None,
        cell_size_m=50.0,
        wind_speed=8.0,
        wind_dir_deg=45.0,
        served_mw=1.0,
        base_served_mw=1.0,
        saidi=0.0,
        prev_saidi=0.0,
        tankers_avail=True,
        ground_crews_avail=False,
        engines_avail=False,
        step=1,
        max_steps=60,
    )
    kw.update(over)
    return kw


# -- contract constants ----------------------------------------------------
def test_contract_constants():
    assert drl.OBS_DIM == 17
    assert drl.N_TACTICS == 4
    assert (drl.ACT_NOOP, drl.ACT_INDIRECT, drl.ACT_DIRECT, drl.ACT_TRIAGE) == (
        0, 1, 2, 3,
    )


# -- extract_obs shape / dtype / bounds ------------------------------------
def test_extract_obs_shape_and_dtype():
    obs = drl.extract_obs(**_obs_kwargs())
    assert obs.shape == (drl.OBS_DIM,)
    assert obs.dtype == np.float32


def test_extract_obs_softly_bounded():
    S = _grid()
    S[0:6, 0:6] = spaces.BURNING          # lots of fire
    obs = drl.extract_obs(**_obs_kwargs(state=S, wind_speed=40.0, saidi=999.0))
    # every feature stays within the documented [-1, 1] soft bound
    assert np.all(obs >= -1.0 - 1e-6) and np.all(obs <= 1.0 + 1e-6)


def test_extract_obs_fractions_match_counts():
    S = _grid(10, 10)
    S[0, 0:5] = spaces.BURNING            # 5 burning
    S[1, 0:2] = spaces.BURNED_OUT         # 2 burned
    S[2, 0:1] = spaces.SUPPRESSED         # 1 suppressed
    S[3, 0:4] = spaces.CONTAINED          # 4 contained
    obs = drl.extract_obs(**_obs_kwargs(state=S))
    assert obs[0] == np.float32(5 / 100)
    assert obs[1] == np.float32(2 / 100)
    assert obs[2] == np.float32(1 / 100)
    assert obs[3] == np.float32(4 / 100)


def test_extract_obs_wind_trig_and_norm():
    obs = drl.extract_obs(**_obs_kwargs(wind_speed=10.0, wind_dir_deg=90.0))
    assert obs[6] == np.float32(10.0 / drl.WIND_SCALE)
    assert obs[7] == np.float32(math.sin(math.radians(90.0)))   # ~1
    assert abs(float(obs[8])) < 1e-6                            # cos(90)=0


def test_extract_obs_step_progress():
    o1 = drl.extract_obs(**_obs_kwargs(step=1, max_steps=60))
    o60 = drl.extract_obs(**_obs_kwargs(step=60, max_steps=60))
    assert o1[16] == np.float32(1 / 60)
    assert o60[16] == np.float32(1.0)


def test_extract_obs_dsaidi_nonnegative():
    # SAIDI can only accrue -> the dSAIDI feature is clipped at >= 0.
    obs = drl.extract_obs(**_obs_kwargs(saidi=1.0, prev_saidi=5.0))
    assert obs[12] == np.float32(0.0)
    obs2 = drl.extract_obs(**_obs_kwargs(saidi=6.0, prev_saidi=5.0))
    assert obs2[12] > 0.0


def test_extract_obs_availability_flags():
    obs = drl.extract_obs(
        **_obs_kwargs(
            tankers_avail=True, ground_crews_avail=False, engines_avail=True
        )
    )
    assert (obs[13], obs[14], obs[15]) == (
        np.float32(1.0), np.float32(0.0), np.float32(1.0),
    )


# -- slope encoder ---------------------------------------------------------
def test_mean_slope_flat_is_zero():
    flat = np.full((8, 8), 100.0)
    assert drl.mean_slope_deg(flat, 50.0) == 0.0


def test_mean_slope_positive_on_ramp():
    ramp = np.tile(np.arange(8.0) * 50.0, (8, 1))   # 100% grade E-W
    slope = drl.mean_slope_deg(ramp, 50.0)
    assert slope > 0.0
    assert drl.mean_slope_deg(None, 50.0) == 0.0    # missing DEM -> 0


# -- teacher doctrine inference -------------------------------------------
def test_teacher_action_from_mutations():
    assert drl.teacher_action_from_mutations([]) == drl.ACT_NOOP
    contained = [(1, 1, spaces.CONTAINED, 0)]
    assert drl.teacher_action_from_mutations(contained) == drl.ACT_TRIAGE
    suppressed = [(2, 2, spaces.SUPPRESSED, 0), (2, 3, spaces.SUPPRESSED, 0)]
    assert drl.teacher_action_from_mutations(suppressed) == drl.ACT_INDIRECT


# -- resource availability wind gate --------------------------------------
def test_resource_availability_wind_gate():
    # tankers fly in moderate wind, grounded in high wind; ground crews never
    # wind-grounded; engines gated purely on count.
    lo = drl.resource_availability(
        n_planes=3, n_helos=0, n_crews=4, n_dozers=0, n_engines=2,
        wind_speed=8.0,
    )
    assert lo == {
        "tankers_avail": True, "ground_crews_avail": True, "engines_avail": True,
    }
    hi = drl.resource_availability(
        n_planes=3, n_helos=0, n_crews=4, n_dozers=0, n_engines=0,
        wind_speed=25.0,
    )
    assert hi["tankers_avail"] is False        # grounded
    assert hi["ground_crews_avail"] is True    # crews still work
    assert hi["engines_avail"] is False        # none configured
