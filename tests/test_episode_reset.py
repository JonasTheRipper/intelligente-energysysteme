"""Guards for per-episode muscle state (the "fire only ignites once" bug).

palaestrAI resets the *environment* between episodes of a phase, but keeps the
*muscles* alive and merely notifies them via
:meth:`palaestrai.agent.muscle.Muscle.reset` -- a hook whose default
implementation does nothing. Every muscle in this project that carries state
across steps therefore has to implement it, or that state silently leaks from
one episode into the next.

The bug this file exists to prevent, verbatim from the A/B run that found it::

    train ep  0   burned=1132   houses_lost=1
    train ep  1   burned=0      houses_lost=0
    ...
    train ep 19   burned=0      houses_lost=0

The wildfire driver latches ``_ignited`` on first injection, so with a live
muscle and a freshly-reset raster the fire was never re-injected: 19 of 20
training episodes ran their full 60 steps on unburnt terrain. Nothing raised --
an empty episode is indistinguishable in the store from a perfectly-suppressed
one -- which is exactly why it needs a test rather than an assertion.

The driver-level tests are numpy-only and run in the light ``unit`` stage; the
muscle-level ones need palaestrai/harl and are skipped without them.
"""

import numpy as np
import pytest

from palaestrai_socal.agents.wildfire_core import WildfireDriver
from wildfire_cma.cma import BURNED_OUT, BURNING, UNBURNED

BOUNDS = (-120.0, 33.0, -118.0, 35.0)
NR, NC = 30, 40

_HAS_STACK = True
try:  # pragma: no cover - availability probe
    import palaestrai.agent  # noqa: F401
    import harl  # noqa: F401
    import torch  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_STACK = False

_needs_stack = pytest.mark.skipif(
    not _HAS_STACK, reason="palaestrai/harl/torch not installed (light stage)"
)


def _driver(**kw):
    """A flat, all-burnable raster so ignition always takes."""
    params = dict(
        fuel=np.full((NR, NC), 3, dtype=np.int16),
        dem=np.zeros((NR, NC), dtype=float),
        delta_m=100.0, bounds=BOUNDS,
        ignition_points=[(-119.0, 34.0)], ignition_step=1,
        env_step_min=5.0, dt_cma_min=5.0, t_burn_steps=6, kappa=3.0,
        wind_speed=20.0, wind_dir_deg=45.0, seed=1,
    )
    params.update(kw)
    return WildfireDriver(**params)


def _run_episode(driver, n_steps=10):
    """Drive one episode from a fresh raster; return the cells that caught fire.

    Mirrors what the environment does between episodes: the grid handed to the
    muscle goes back to all-UNBURNED. The muscle/driver does NOT.
    """
    grid = np.full((NR, NC), UNBURNED, dtype=np.int8)
    for _ in range(n_steps):
        for (r, c, s, _layer) in driver.step(grid):
            grid[r, c] = s
    return int(np.count_nonzero((grid == BURNING) | (grid == BURNED_OUT)))


# -- the driver --------------------------------------------------------------
@pytest.mark.unit
def test_fire_ignites_in_every_episode_after_reset():
    """The regression guard: N episodes, N fires."""
    d = _driver()
    burned = []
    for _ in range(4):
        burned.append(_run_episode(d))
        d.reset()
    assert all(b > 0 for b in burned), (
        f"episodes without fire: {burned} -- the ignition latch was not "
        "cleared at the episode boundary"
    )


@pytest.mark.unit
def test_without_reset_only_the_first_episode_burns():
    """Pins the failure mode itself, so the guard above cannot pass vacuously.

    If a future refactor makes ignition re-trigger on its own this test fails
    loudly rather than leaving the guard testing nothing.
    """
    d = _driver()
    first = _run_episode(d)
    second = _run_episode(d)          # no reset() in between
    assert first > 0
    assert second == 0, (
        "expected the un-reset driver to stay latched; if ignition now "
        "re-arms itself, delete this test and simplify reset()"
    )


@pytest.mark.unit
def test_reset_clears_burn_timer_and_step_counter():
    d = _driver()
    _run_episode(d)
    assert d._step > 0 and d._ignited and d.burn_timer.any()
    d.reset()
    assert d._step == 0
    assert not d._ignited
    assert not d.burn_timer.any(), (
        "a stale burn timer burns out cells of the NEW fire early"
    )


@pytest.mark.unit
def test_reset_respects_ignition_step_again():
    """Reset restores the delay, it does not ignite immediately."""
    d = _driver(ignition_step=3)
    _run_episode(d)
    d.reset()
    grid = np.full((NR, NC), UNBURNED, dtype=np.int8)
    assert d.step(grid) == []
    assert d.step(grid) == []
    assert any(s == BURNING for (_r, _c, s, _l) in d.step(grid))


@pytest.mark.unit
def test_reset_keeps_the_expensive_build_products():
    """Re-arming must not rebuild the raster or drop the per-cell wind field."""
    d = _driver()
    raster, cma, fuel = d.raster, d._cma, d.raster.fuel.copy()
    d._cma.set_wind_field(np.dstack([
        np.full((NR, NC), 12.0), np.full((NR, NC), 90.0)
    ]))
    d.reset()
    assert d.raster is raster and d._cma is cma
    assert np.array_equal(d.raster.fuel, fuel)
    assert d._cma._wind_field is not None


@pytest.mark.unit
def test_reset_does_not_replay_an_identical_fire():
    """Episodes must still differ -- reset re-arms, it does not re-seed.

    Re-seeding the CMA rng here would hand the learner the same fire N times
    and remove the environment variation training depends on.
    """
    d = _driver()
    sizes = []
    for _ in range(6):
        sizes.append(_run_episode(d))
        d.reset()
    assert len(set(sizes)) > 1, (
        f"every episode burned the same {sizes[0]} cells -- the spread rng "
        "appears to be re-seeded per episode"
    )


# -- the damage mapper -------------------------------------------------------
@pytest.mark.unit
def test_damage_driver_unlatches_shed_buses_on_reset():
    from palaestrai_socal.agents.damage_core import DamageMapperDriver

    drv = DamageMapperDriver({1: (-119.0, 34.0)}, BOUNDS, (NR, NC))
    grid = np.full((NR, NC), UNBURNED, dtype=np.int8)
    r, c = drv.bus_cell[1]
    grid[r, c] = BURNING
    assert drv.evaluate(grid) == {1}
    drv.reset()
    assert drv.evaluate(np.full((NR, NC), UNBURNED, dtype=np.int8)) == set(), (
        "a bus shed in an earlier episode stayed shed on a healthy grid"
    )


# -- the muscles -------------------------------------------------------------
@_needs_stack
@pytest.mark.slow
@pytest.mark.parametrize(
    "module,cls",
    [
        ("wildfire_agent", "WildfireCmaMuscle"),
        ("firefighter_agent", "FirefighterMuscle"),
        ("damage_agent", "DamageMapperMuscle"),
        ("firefighter_drl_agent", "LearningFirefighterMuscle"),
    ],
)
def test_every_muscle_overrides_reset(module, cls):
    """Static guard against the whole class of bug, not just this instance.

    A muscle that keeps state across steps and inherits the no-op
    ``Muscle.reset`` leaks that state between episodes. Any new muscle added
    here must make a deliberate choice -- even "nothing to reset" has to be
    written down as an override.
    """
    import importlib

    from palaestrai.agent.muscle import Muscle

    muscle = getattr(
        importlib.import_module(f"palaestrai_socal.agents.{module}"), cls
    )
    assert muscle.reset is not Muscle.reset, (
        f"{cls} does not override Muscle.reset(); per-episode state will leak "
        "into the next episode"
    )


@_needs_stack
@pytest.mark.slow
def test_wildfire_muscle_reset_is_safe_before_the_driver_exists():
    """reset() may fire before the first step (empty episode, early abort)."""
    from palaestrai_socal.agents.wildfire_agent import WildfireCmaMuscle

    WildfireCmaMuscle(raster_nrows=NR, raster_ncols=NC).reset()


@_needs_stack
@pytest.mark.slow
def test_drl_muscle_reset_clears_episode_state_but_not_warmup():
    """The warm-up counter is per-PHASE; the observation features are per-episode."""
    from palaestrai_socal.agents.firefighter_drl_agent import (
        LearningFirefighterMuscle,
    )

    m = LearningFirefighterMuscle(
        n_planes=1, start_steps=10_000, max_steps=60,
        raster_nrows=NR, raster_ncols=NC, bounds=list(BOUNDS), cell_size_m=100.0,
    )
    m._step_i = 60
    m._cum_customer_min = 1234.5
    m._prev_saidi = 0.7
    m._actions_proposed = 42

    m.reset()

    assert m._step_i == 0
    assert m._cum_customer_min == 0.0
    assert m._prev_saidi == 0.0
    assert m._actions_proposed == 42, (
        "resetting the SAC warm-up counter per episode restarts exploration "
        "every episode and the policy never converges"
    )


@_needs_stack
@pytest.mark.slow
def test_scripted_firefighter_reset_clears_cumulative_telemetry():
    from palaestrai_socal.agents.firefighter_agent import FirefighterMuscle

    m = FirefighterMuscle(n_planes=2, raster_nrows=NR, raster_ncols=NC)
    m._line_km_cumulative = 12.5
    m._last_telemetry = {"line_km_cumulative": 12.5}
    m.reset()
    assert m._line_km_cumulative == 0.0
    assert m._last_telemetry == {}
