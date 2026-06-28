"""Adapter tests for the FirefighterMuscle palaestrai shim.

These touch the real palaestrAI ``SensorInformation`` / ``ActuatorInformation``
/ ``Box`` types, so (like the damage-agent coercion tests) they are skipped in
the lightweight CI ``unit`` stage that does not install palaestrai, and run in
the manual ``system`` stage / locally. The numpy-only decision logic is covered
exhaustively by ``test_firefighter_core.py``.

What they pin down (from ``_v0.3_IMPL_BRIEF.md``): the muscle emits dtype-correct
``gis.cell_mutations`` (a retardant line of SUPPRESSED cells on
LAYER_SUPPRESSION) in moderate wind, and an EMPTY mutation set when the fleet is
grounded by high wind.
"""

import numpy as np
import pytest

_HAS_PALAESTRAI = True
try:  # pragma: no cover - availability probe
    import palaestrai.types  # noqa: F401
    import palaestrai.agent  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_PALAESTRAI = False

_needs_palaestrai = pytest.mark.skipif(
    not _HAS_PALAESTRAI, reason="palaestrai not installed (lightweight stage)")

NR, NC = 30, 30
BOUNDS = [-118.14358, 34.13604, -118.04358, 34.23604]
CELL_M = 50.0


def _sensors(state):
    from palaestrai.agent import SensorInformation
    from palaestrai_socal import spaces
    fuel = np.full((NR, NC), 3, dtype=np.float64)
    return [
        SensorInformation(
            value=state.astype(np.float64).ravel(),
            space=spaces.vector_box(0.0, 16.0, NR * NC), uid="gis.cell_state"),
        SensorInformation(
            value=fuel.ravel(),
            space=spaces.vector_box(0.0, 64.0, NR * NC), uid="gis.fuel_class"),
    ]


def _actuator():
    from palaestrai.agent import ActuatorInformation
    from palaestrai_socal import spaces
    return ActuatorInformation(
        value=spaces.encode_mutations([], cap=spaces.CAP),
        space=spaces.mutation_space(spaces.CAP), uid="gis.cell_mutations")


def _muscle(n_planes, wind_speed):
    from palaestrai_socal.agents.firefighter_agent import FirefighterMuscle
    return FirefighterMuscle(
        n_planes=n_planes, env_step_min=60.0, wind_speed=wind_speed,
        wind_dir_deg=0.0, raster_nrows=NR, raster_ncols=NC,
        bounds=BOUNDS, cell_size_m=CELL_M)


def _burning_front():
    from palaestrai_socal import spaces
    S = np.full((NR, NC), spaces.UNBURNED, dtype=np.int8)
    S[14, 12:18] = spaces.BURNING
    return S


@_needs_palaestrai
def test_emits_suppressed_line_in_moderate_wind():
    from palaestrai_socal import spaces
    muscle = _muscle(n_planes=3, wind_speed=8.0)     # below DEGRADE -> full budget
    acts, brain = muscle.propose_actions(_sensors(_burning_front()), [_actuator()])
    # v0.4 telemetry fix: the brain/data channel is None (no more dict + int
    # warning); telemetry lives on the muscle instance.
    assert brain is None
    out = acts[0]
    # dtype-correct: the written value matches the actuator's own Box dtype.
    assert np.asarray(out.value).dtype == out.space.dtype
    muts = spaces.decode_mutations(np.asarray(out.value).ravel(), cap=spaces.CAP)
    assert len(muts) > 0
    # every mutation is a SUPPRESSED cell on the suppression layer, in bounds.
    for (r, c, st, layer) in muts:
        assert st == spaces.SUPPRESSED
        assert layer == spaces.LAYER_SUPPRESSION
        assert 0 <= r < NR and 0 <= c < NC
    telem = muscle._last_telemetry
    assert telem["grounded"] == 0.0
    assert telem["suppressed_cells"] == float(len(muts))


@_needs_palaestrai
def test_empty_mutations_when_grounded():
    from palaestrai_socal import spaces
    muscle = _muscle(n_planes=7, wind_speed=25.0)    # >= GROUND_WIND_MS -> grounded
    acts, brain = muscle.propose_actions(_sensors(_burning_front()), [_actuator()])
    assert brain is None                             # telemetry fix: non-dict channel
    muts = spaces.decode_mutations(np.asarray(acts[0].value).ravel(), cap=spaces.CAP)
    assert muts == []
    telem = muscle._last_telemetry
    assert telem["grounded"] == 1.0
    assert telem["mutation_cells"] == 0.0


@_needs_palaestrai
def test_budget_scales_with_more_planes():
    from palaestrai_socal import spaces

    def _n_cells(n):
        m = _muscle(n_planes=n, wind_speed=8.0)
        acts, _ = m.propose_actions(_sensors(_burning_front()), [_actuator()])
        return len(spaces.decode_mutations(
            np.asarray(acts[0].value).ravel(), cap=spaces.CAP))

    # more planes -> at least as much line (capped by available head candidates).
    assert _n_cells(1) <= _n_cells(3) <= _n_cells(7)
    assert _n_cells(3) > 0
