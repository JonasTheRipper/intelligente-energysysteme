"""Regression tests for the ragged-safe palaestrAI Memory shim.

:mod:`palaestrai_socal.agents._memory_compat` patches
``_MuscleMemory._infos_to_df`` so the DRL firefighter's sensor mix -- large
grid rasters (``gis.cell_state`` ~23660 elements) alongside scalar
``*-load-*.p_mw`` power readings -- no longer blows up the rectangular
``pd.DataFrame`` that palaestrAI builds for every remembered step.

These tests pin both halves of the contract:

* the **uniform** path must stay byte-equivalent to stock palaestrAI, so the
  shim is invisible to every other agent, and
* the **ragged** path must produce a one-row frame whose object cells preserve
  each sensor's array *as an array* -- including length-1 power sensors, which
  an earlier ``frame.at[0, key] = value`` implementation silently unwrapped
  into 0-d scalars, corrupting exactly the readings ``SaidiObjective`` sums.

They also pin the properties that make the patch safe to apply from several
processes: idempotence, preserved ``staticmethod`` semantics, and that nothing
is ever written to the installed palaestrAI package.

The module needs palaestrAI's real ``_MuscleMemory``; it skips cleanly when
palaestrAI is not installed (as in the lightweight CI unit stage).
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

memory_mod = pytest.importorskip(
    "palaestrai.agent.memory",
    reason="palaestrAI not installed; the shim has nothing to patch",
)
_MuscleMemory = memory_mod._MuscleMemory

# Captured before the shim is imported, so it is the genuine upstream
# implementation. Everything below restores this between tests.
_ORIGINAL = _MuscleMemory.__dict__["_infos_to_df"]
if getattr(_MuscleMemory._infos_to_df, "_socal_ragged_safe", False):
    pytest.skip(
        "Memory was already patched before this module imported; the pristine "
        "upstream implementation is unrecoverable",
        allow_module_level=True,
    )

from palaestrai_socal.agents import _memory_compat  # noqa: E402

upstream = _ORIGINAL.__func__


class _Reading:
    """Stand-in for SensorInformation: only uid/value are ever read."""

    def __init__(self, uid, value):
        self.uid = uid
        self.value = value


def _ragged():
    """The firefighter's actual subscription shape: rasters plus scalars."""
    return [
        _Reading("gis.cell_state", np.zeros(23660, dtype=np.int8)),
        _Reading("gis.fuel_class", np.zeros(23660, dtype=float)),
        _Reading("gis.wind_field", np.array([15.0, 45.0])),
        _Reading("Powergrid-0.0-load-0-0.p_mw", np.array([0.7])),
        _Reading("Powergrid-0.0-load-1-1.p_mw", np.array([0.3])),
    ]


def _uniform():
    return [
        _Reading("Powergrid-0.0-load-0-0.p_mw", np.array([0.7])),
        _Reading("Powergrid-0.0-load-1-1.p_mw", np.array([0.3])),
    ]


def _duplicate_uids():
    # Upstream does not guarantee unique uids. Equal collected lengths keep
    # this on the uniform path (unequal counts would be genuinely ragged).
    return [
        _Reading("dup", np.array([1.0])),
        _Reading("dup", np.array([2.0])),
        _Reading("other", np.array([3.0])),
        _Reading("other", np.array([4.0])),
    ]


@pytest.fixture(autouse=True)
def pristine_memory():
    """Run each test against un-patched palaestrAI, and restore it after."""
    _MuscleMemory._infos_to_df = _ORIGINAL
    yield
    _MuscleMemory._infos_to_df = _ORIGINAL


# -- the defect the shim exists for ----------------------------------------
def test_upstream_raises_on_ragged_sensor_mix():
    """Without the shim, palaestrAI cannot tabulate the firefighter's step."""
    with pytest.raises(ValueError, match="same length"):
        upstream(_ragged())


# -- uniform path: indistinguishable from stock palaestrAI -----------------
@pytest.mark.parametrize(
    "infos", [_uniform(), [], _duplicate_uids()],
    ids=["scalar-sensors", "empty", "duplicate-uids"],
)
def test_equal_length_output_matches_upstream(infos):
    _memory_compat.install()
    pd.testing.assert_frame_equal(
        _MuscleMemory._infos_to_df(infos), upstream(infos)
    )


# -- ragged path -----------------------------------------------------------
def test_ragged_returns_one_row_frame_of_object_cells():
    _memory_compat.install()
    frame = _MuscleMemory._infos_to_df(_ragged())

    assert len(frame) == 1
    assert list(frame.columns) == [
        "gis.cell_state", "gis.fuel_class", "gis.wind_field",
        "Powergrid-0.0-load-0-0.p_mw", "Powergrid-0.0-load-1-1.p_mw",
    ]
    # every cell keeps its own sensor's full array, indexable by uid.
    assert frame["gis.cell_state"].iloc[0].shape == (23660,)
    assert frame["gis.fuel_class"].iloc[0].shape == (23660,)
    assert frame["gis.wind_field"].iloc[0].tolist() == [15.0, 45.0]


def test_length_one_sensors_stay_arrays_not_scalars():
    """Regression: ``.at`` assignment unwraps length-1 arrays into 0-d scalars.

    The power sensors the objective reads are length-1, so an implementation
    that loses their shape breaks ``value[0]`` downstream while still looking
    like a plausible frame.
    """
    _memory_compat.install()
    cell = _MuscleMemory._infos_to_df(_ragged())["Powergrid-0.0-load-0-0.p_mw"]
    value = cell.iloc[0]

    assert isinstance(value, np.ndarray)
    assert value.shape == (1,)
    assert value.tolist() == [0.7]
    assert float(value[0]) == pytest.approx(0.7)


def test_memory_getitem_survives_ragged_readings(monkeypatch):
    """The real ``__getitem__`` -- what ``tail(1)`` walks -- must work."""
    # palaestrAI 3.5.9 still uses the np.NAN alias numpy 2 removed; production
    # pins numpy 1.26, so restore it rather than skip the code path.
    monkeypatch.setattr(np, "NAN", np.nan, raising=False)
    _memory_compat.install()

    mem = _MuscleMemory()
    mem.sensor_readings.append(_ragged())
    mem.actuator_setpoints.append([])
    mem.rewards.append([_Reading("reward", np.array([-1.0]))])
    mem.dones.append(False)
    mem.objective.append(np.array([-1.0]))

    shard = mem[0]

    assert shard.sensor_readings["gis.cell_state"].iloc[0].shape == (23660,)
    assert float(shard.objective.item()) == -1.0


# -- install() properties --------------------------------------------------
def test_install_is_idempotent():
    assert _memory_compat.installed() is False
    assert _memory_compat.install() is True
    assert _memory_compat.installed() is True

    patched = _MuscleMemory._infos_to_df
    assert _memory_compat.install() is False
    # a second install from another process/import must not re-wrap.
    assert _MuscleMemory._infos_to_df is patched


def test_install_preserves_staticmethod_semantics():
    _memory_compat.install()

    assert isinstance(_MuscleMemory.__dict__["_infos_to_df"], staticmethod)
    pd.testing.assert_frame_equal(
        _MuscleMemory._infos_to_df(_uniform()),
        _MuscleMemory()._infos_to_df(_uniform()),
    )


def test_install_does_not_modify_the_installed_package():
    """The patch is a runtime rebind; site-packages stays untouched."""
    with open(memory_mod.__file__, "rb") as fh:
        before = hashlib.sha256(fh.read()).hexdigest()

    _memory_compat.install()
    _MuscleMemory._infos_to_df(_ragged())

    with open(memory_mod.__file__, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == before
