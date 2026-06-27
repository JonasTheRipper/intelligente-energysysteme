"""Per-environment store filters (palaestrAI ``EnvironmentStateTransformer``).

palaestrAI stores one ``world_states`` row per environment per step, and the
stored ``state_dump`` is the environment's **full** ``sensor_information`` list
(``palaestrai/store/receiver.py`` ``_write_world_state`` -> ``state_dump =
message.sensors``). For the SoCal v0.2 envs that is huge and almost entirely
redundant:

* ``gis_world`` re-publishes ``gis.fuel_class`` and ``gis.elevation_m`` (two
  static ``nrows*ncols`` float64 grids, ~8 MB each as JSON at 600x760) on
  *every* step even though they never change, plus the padded ``gis.front_cells``
  set and two scalar counters the timelapse never reads. ~37 MB/step.
* ``socal_grid`` (real MIDAS) publishes every bus/line/load attribute of the
  2294-bus net each step (~16.5 MB), but the store-only timelapse + KPIs only
  need the load ``p_mw`` sensors (served-MW shortfall) plus the two ``grid_probe``
  sensors.

That volume filled the 20 GB root disk on the full 120-step run (see
``V02_OOM_FINDINGS.md`` §6). palaestrAI's first-class fix is the
``EnvironmentStateTransformer`` (``palaestrai/environment/environment_state_transformer.py``):
a per-environment callable wired via the YAML ``state_transformer:`` block,
loaded by ``environment_conductor.py`` (~line 119) and applied in
``environment.py`` (~line 292) to ``EnvironmentState.sensor_information`` before
the update response is built. ``StoreDumpTrimmer`` is that callable.

Why this does NOT starve the agents
------------------------------------
The transformer trims the *published* sensor list, which feeds both the store
and (via the SimulationController) the agents. It is safe here because:

* The ``SimulationController`` accumulates sensor readings across steps
  (``_sensors_available.update(...)`` / ``_current_sensor_readings`` are only
  cleared on episode reset), so a sensor dropped on later steps keeps its last
  published value for any agent that still subscribes to it.
* ``WildfireCmaMuscle._ensure_driver`` reads ``gis.fuel_class`` /
  ``gis.elevation_m`` exactly **once** (it builds the driver on the first
  inference and returns early thereafter); subsequent steps only read
  ``gis.cell_state``. So keeping the static grids on the first stored step and
  dropping them afterwards is transparent to the agent and still lets
  ``analysis/store_readers.read_run`` rebuild the basemap from row 0.
* The agent->sensor wiring is fixed at environment *setup* (from the
  ``EnvironmentBaseline.sensors_available`` advertised by ``start_environment``),
  which this transformer does not touch -- it only runs on per-step updates.
"""

from __future__ import annotations

from typing import List, Optional

from palaestrai.environment.environment_state_transformer import (
    EnvironmentStateTransformer,
)


class StoreDumpTrimmer(EnvironmentStateTransformer):
    """Keep only the sensors the store-backed timelapse + KPIs actually need.

    A sensor is kept on a given step iff its uid

    * ends with one of ``keep_suffixes``, or
    * contains one of ``keep_substrings``, or
    * ends with one of ``static_suffixes`` **and this is the first step**
      (static fields are written once, on the first stored row, then dropped).

    Everything else is dropped from the stored ``state_dump``. uids are matched
    on the environment-internal name (e.g. ``gis.cell_state``,
    ``Powergrid-0.0-load-7-9.p_mw``) -- palaestrAI prepends the ``<env_uid>.``
    prefix only when forwarding to agents, not in this transform.

    Parameters
    ----------
    keep_suffixes : list[str], optional
        uid suffixes to keep on every step.
    keep_substrings : list[str], optional
        Substrings; a uid containing any of them is kept on every step.
    static_suffixes : list[str], optional
        uid suffixes for large fields that never change -- kept on the first
        step only, dropped on all subsequent steps.
    """

    def __init__(
        self,
        keep_suffixes: Optional[List[str]] = None,
        keep_substrings: Optional[List[str]] = None,
        static_suffixes: Optional[List[str]] = None,
        **params,
    ):
        super().__init__(**params)
        self._keep_suffixes = tuple(keep_suffixes or ())
        self._keep_substrings = tuple(keep_substrings or ())
        self._static_suffixes = tuple(static_suffixes or ())
        self._calls = 0

    def _keep_always(self, uid: str) -> bool:
        if any(uid.endswith(s) for s in self._keep_suffixes):
            return True
        if any(sub in uid for sub in self._keep_substrings):
            return True
        return False

    def _is_static(self, uid: str) -> bool:
        return any(uid.endswith(s) for s in self._static_suffixes)

    def __call__(self, environment_state):
        self._calls += 1
        first_step = self._calls == 1
        sensors = environment_state.sensor_information or []
        kept = [
            si
            for si in sensors
            if self._keep_always(si.uid)
            or (first_step and self._is_static(si.uid))
        ]
        environment_state.sensor_information = kept
        return environment_state
