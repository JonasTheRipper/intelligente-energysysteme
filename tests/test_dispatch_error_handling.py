#!/usr/bin/env python3
"""Guard the error handling in ``dispatch_and_run.run``.

``run`` retries a failed power flow with a different algorithm and, if that
also fails, reports the step unsolved and returns ``False``. Before this suite
existed the two ``except`` clauses were bare ``except Exception``, so *any*
error out of ``pp.runpp`` -- a missing column, a renamed API, an uninstalled
transitive dependency -- was downgraded to "did not converge". The caller then
wrote a complete set of zeroed KPIs that is indistinguishable from a genuine
system-wide blackout.

That is not hypothetical: it is how the pandapower ZIP-load column rename
(``const_z_percent`` -> ``const_z_p_percent``) produced plausible-looking
all-zero results instead of an error.

These tests pin the contract:

* a genuine ``LoadflowNotConverged`` still drives the fallback and still
  returns ``False`` -- the solver behaviour is unchanged;
* anything else propagates to the caller.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata

import pytest

# Same guard as test_pandapower_compat: skip only when pandapower is genuinely
# absent (the lightweight `unit` stage), never when it is installed but broken.
try:
    importlib_metadata.version("pandapower")
except importlib_metadata.PackageNotFoundError:
    pytest.skip(
        "pandapower not installed (lightweight unit stage)",
        allow_module_level=True,
    )

import pandapower as pp  # noqa: E402  -- must follow the skip above
import pandapower.networks as pn  # noqa: E402

from socal_grid import dispatch_and_run as dar  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture()
def net():
    """A small, fast, real pandapower net.

    Only ``net.load`` is exercised: every test stubs out ``pp.runpp``, and
    ``run`` is called with ``verbose=False`` so it never reads ``res_bus``.
    """
    return pn.create_cigre_network_mv()


@pytest.fixture(autouse=True)
def _neutralise_grid_prep(monkeypatch):
    """Stub the SoCal-specific network surgery.

    ``strengthen`` and ``dispatch`` assume the real SoCal model. They are not
    under test here -- the exception routing is -- so they are replaced with
    no-ops to keep these tests fast and independent of the 5.9 MB grid.
    """
    monkeypatch.setattr(dar, "strengthen", lambda net: None)
    monkeypatch.setattr(dar, "dispatch", lambda net, scale=1.0: None)


def _raise_always(exc):
    """A ``pp.runpp`` replacement that always raises ``exc``."""

    def _fake(net, *args, **kwargs):
        raise exc

    return _fake


# --------------------------------------------------------------------------
# 1. Non-convergence: behaviour must be unchanged.
# --------------------------------------------------------------------------

def test_persistent_divergence_returns_false(net, monkeypatch, capsys):
    """All three algorithms diverge -> report the step and return False."""
    monkeypatch.setattr(
        dar.pp, "runpp",
        _raise_always(pp.LoadflowNotConverged("Power Flow did not converge")),
    )

    assert dar.run(net, verbose=False) is False
    assert "FAILED" in capsys.readouterr().out


def test_fallback_recovers_after_divergence(net, monkeypatch):
    """First attempt diverges, the dc-init retry succeeds -> True.

    This is the reason the handler exists at all; narrowing it must not
    disable the recovery path.
    """
    calls = {"n": 0}

    def _fake(net_, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise pp.LoadflowNotConverged("Power Flow did not converge")
        return None

    monkeypatch.setattr(dar.pp, "runpp", _fake)

    assert dar.run(net, verbose=False) is True
    # 3 continuation steps, the first of which needed one extra attempt.
    assert calls["n"] == len(dar.CONT_STEPS) + 1


# --------------------------------------------------------------------------
# 2. Everything else must escape.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [
        # The actual pandapower 3.1.x ZIP-load regression.
        KeyError("const_z_percent"),
        # A renamed or removed pandapower API.
        AttributeError("module 'pandapower' has no attribute 'runpp'"),
        # pandapower installed --no-deps with a transitive dep missing.
        ModuleNotFoundError("No module named 'pandera'"),
        # A signature change in runpp.
        TypeError("runpp() got an unexpected keyword argument 'algorithm'"),
        # A malformed net.
        ValueError("bus 42 is not in the net"),
    ],
    ids=["keyerror", "attributeerror", "modulenotfound", "typeerror", "valueerror"],
)
def test_non_convergence_errors_propagate(net, monkeypatch, exc):
    """A broken dependency must fail loudly, not masquerade as divergence."""
    monkeypatch.setattr(dar.pp, "runpp", _raise_always(exc))

    with pytest.raises(type(exc)):
        dar.run(net, verbose=False)


def test_error_in_fallback_also_propagates(net, monkeypatch):
    """The inner handler must be just as narrow as the outer one.

    Divergence on the first attempt enters the fallback loop; a dependency
    error raised *there* must still escape rather than be recorded as
    ``last`` and reported as a failed continuation step.
    """
    calls = {"n": 0}

    def _fake(net_, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise pp.LoadflowNotConverged("Power Flow did not converge")
        raise KeyError("const_z_percent")

    monkeypatch.setattr(dar.pp, "runpp", _fake)

    with pytest.raises(KeyError):
        dar.run(net, verbose=False)


# --------------------------------------------------------------------------
# 3. Meta-guard: keep the tuple narrow.
# --------------------------------------------------------------------------

def test_convergence_errors_stays_narrow():
    """Fail if someone widens CONVERGENCE_ERRORS back towards Exception.

    ``ppException`` is pandapower's base class for *all* its errors, including
    configuration and validation ones, so catching it would reintroduce the
    silent-zero failure mode this module guards against.
    """
    from pandapower.auxiliary import ppException

    assert pp.LoadflowNotConverged in dar.CONVERGENCE_ERRORS

    for caught in dar.CONVERGENCE_ERRORS:
        # Every entry must be a strict subclass of pandapower's error base --
        # never ppException itself, and never a builtin.
        assert issubclass(caught, ppException), caught
        assert caught is not ppException

    # The failure modes that must never be swallowed.
    for builtin in (KeyError, AttributeError, ModuleNotFoundError,
                    TypeError, ValueError):
        assert not issubclass(builtin, dar.CONVERGENCE_ERRORS), builtin
