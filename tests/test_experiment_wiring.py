"""Static wiring checks over every experiment run file.

palaestrAI's ``experiment-check-syntax`` validates the YAML against a schema,
but it cannot tell whether an agent's sensor/actuator uids actually resolve
against the environments -- that only happens once the SimulationController has
started every environment, which for the Eaton experiments is ~80 s of MIDAS
setup per phase. A typo therefore surfaces as a CRITICAL several minutes into a
multi-hour run:

    found sensor/actuator assignments ... which could not be matched with the
    sensors/actuators actually provided by the environments:
    {'firefighter': ({'gis_world.gis.cell_mutations'}, set())}

These tests catch the statically-detectable subset in milliseconds. They are
deliberately generic (they scan every experiment_*.yml) so a new run file is
covered the moment it is added.
"""

import glob
import os

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILES = sorted(glob.glob(os.path.join(_ROOT, "palaestrai_socal", "experiment_*.yml")))

# uids the environments publish as ACTUATORS. Listing one under an agent's
# `sensors:` is the failure mode above: the SimulationController looks for a
# sensor of that name, finds none, and aborts the phase.
_KNOWN_ACTUATOR_SUFFIXES = ("gis.cell_mutations", "gis.wind_override")


def _agents(path):
    """Yield (phase_uid, agent_name, sensors, actuators) for one run file."""
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    for phase in doc.get("schedule") or []:
        for phase_uid, cfg in phase.items():
            for agent in cfg.get("agents") or []:
                yield (
                    phase_uid,
                    agent.get("name", "?"),
                    list(agent.get("sensors") or []),
                    list(agent.get("actuators") or []),
                )


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(p) for p in _FILES])
def test_no_uid_is_both_sensor_and_actuator(path):
    """An agent must not list the same uid as sensor and actuator."""
    for phase_uid, name, sensors, actuators in _agents(path):
        overlap = set(sensors) & set(actuators)
        assert not overlap, (
            f"{os.path.basename(path)} :: {phase_uid} :: agent '{name}' lists "
            f"{sorted(overlap)} under BOTH sensors and actuators"
        )


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(p) for p in _FILES])
def test_actuator_uids_are_not_listed_as_sensors(path):
    """Environment actuators must never appear in an agent's sensor list."""
    for phase_uid, name, sensors, _actuators in _agents(path):
        bad = [
            s for s in sensors
            if any(str(s).endswith(suf) for suf in _KNOWN_ACTUATOR_SUFFIXES)
        ]
        assert not bad, (
            f"{os.path.basename(path)} :: {phase_uid} :: agent '{name}' lists "
            f"actuator(s) {bad} under sensors:"
        )


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(p) for p in _FILES])
def test_every_agent_has_at_least_one_actuator(path):
    """palaestrAI requires a non-empty actuator list per agent."""
    for phase_uid, name, _sensors, actuators in _agents(path):
        assert actuators, (
            f"{os.path.basename(path)} :: {phase_uid} :: agent '{name}' has no "
            "actuators; the AgentConductor cannot set it up"
        )


@pytest.mark.parametrize("path", _FILES, ids=[os.path.basename(p) for p in _FILES])
def test_no_duplicate_sensor_entries(path):
    """A repeated uid silently doubles that reading in the agent's Memory."""
    for phase_uid, name, sensors, _actuators in _agents(path):
        dupes = {s for s in sensors if sensors.count(s) > 1}
        assert not dupes, (
            f"{os.path.basename(path)} :: {phase_uid} :: agent '{name}' lists "
            f"{sorted(dupes)} more than once under sensors:"
        )
