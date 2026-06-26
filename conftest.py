"""Pytest configuration: auto-mark tests as ``unit`` or ``slow``.

The light-CI / heavy-manual split is driven entirely by markers so a single
test suite serves both. We classify by module:

* ``test_cma``, ``test_postgis`` and ``test_gis`` are pure unit tests
  (numpy only -- no grid, no power flow, no pandapower/palaestrai imports).
* ``test_damage_mapper``, ``test_palaestrai_env`` and ``test_midas_steps``
  load the 5.9 MB SoCal grid and run a pandapower dispatch / power flow, so
  they are marked ``slow`` (run via the manual ``system`` CI stage or locally).

CI usage::

    pytest -m unit          # fast stage, every push
    pytest -m slow          # manual stage, full grid co-simulation
"""

import os
import sys

import pytest

# make the repo root importable for all tests (palaestrai_socal, wildfire_cma...)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_SLOW_MODULES = {"test_damage_mapper", "test_palaestrai_env", "test_midas_steps"}
_UNIT_MODULES = {"test_cma", "test_postgis", "test_gis"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        mod = item.module.__name__.split(".")[-1]
        if mod in _SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
        elif mod in _UNIT_MODULES:
            item.add_marker(pytest.mark.unit)
