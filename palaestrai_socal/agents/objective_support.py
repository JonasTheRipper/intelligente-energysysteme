"""Shared Memory-reading helpers for the SoCal objectives.

An :class:`~palaestrai.agent.objective.Objective` is handed the agent's
:class:`~palaestrai.agent.memory.Memory` and nothing else, so any objective that
needs more than the environment reward has to dig its inputs out of
``memory.tail(1)``. That tail is not one shape but three:

1. **A list of information objects** -- what the unit tests' fakes hand over.
2. **An equal-length ``pd.DataFrame``** -- stock palaestrAI
   (``_MuscleMemory._infos_to_df``) builds one column per uid, each flattened
   with ``np.reshape(value, -1)``. Every column must be the same length, so a
   scalar sensor in an all-scalar frame is a single-row column.
3. **A one-row object-cell ``pd.DataFrame``** -- what the ragged-safe shim
   :mod:`palaestrai_socal.agents._memory_compat` produces once the agent mixes
   grid rasters with scalar sensors (the firefighter's case). Each cell then
   holds a whole array.

uids are matched by **suffix**, because palaestrAI prefixes them with the
environment uid when forwarding to agents (``gis_world.gis.houses_total``)
while the environment-internal name has no prefix.

Numpy/pandas only -- no palaestrai import -- so objectives built on it stay
testable in the light CI ``unit`` stage.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

LOG = logging.getLogger("palaestrai_socal.agents.objective_support")


def suffix_match(uid: Any, suffix: str) -> bool:
    """True if a (possibly environment-prefixed) uid ends with ``suffix``."""
    uid = str(uid)
    return uid == suffix or uid.endswith("." + suffix) or uid.endswith(suffix)


def _first_finite(value: Any) -> Optional[float]:
    """First usable float in a reading, or None if there is none.

    A uid missing from a concatenated frame reads back as NaN, which must be
    reported as *absent* rather than poisoning arithmetic downstream.
    """
    try:
        flat = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if not flat.size:
        return None
    scalar = float(flat[0])
    return scalar if np.isfinite(scalar) else None


def read_scalar(tail, suffix: str) -> Optional[float]:
    """Read one scalar sensor out of a Memory tail, or None if absent.

    Handles all three tail shapes described in the module docstring. Type-
    dispatches rather than testing truthiness: ``bool(DataFrame)`` raises
    ``ValueError: The truth value of a DataFrame is ambiguous``, and iterating
    one yields column *names* rather than readings.
    """
    readings = getattr(tail, "sensor_readings", None)
    if readings is None:
        return None

    if hasattr(readings, "columns"):  # pd.DataFrame, without importing pandas
        if readings.empty:
            return None
        for col in readings.columns:
            if suffix_match(col, suffix):
                return _first_finite(readings[col].iloc[-1])
        return None

    try:
        items: List[Any] = list(readings)
    except TypeError:
        return None
    # Memory.tail may nest readings one level deep depending on version.
    if items and isinstance(items[0], (list, tuple)):
        items = [i for group in items for i in group]
    for info in items:
        uid = getattr(info, "uid", None) or getattr(info, "sensor_id", None)
        if uid is not None and suffix_match(uid, suffix):
            return _first_finite(getattr(info, "value", None))
    return None
