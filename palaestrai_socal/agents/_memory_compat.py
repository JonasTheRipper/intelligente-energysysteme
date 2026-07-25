"""Ragged-safe compatibility shim for palaestrAI's agent Memory.

Why this exists
---------------
:meth:`palaestrai.agent.memory._MuscleMemory._infos_to_df` tabulates one step's
sensor readings into a *rectangular* ``pd.DataFrame`` -- one column per sensor
uid, each column ``np.reshape(np.array(value), -1)``. That silently assumes
every sensor flattens to the same number of elements.

The v0.7 DRL firefighter is the first learning agent whose subscription mixes
large grid rasters (``gis.cell_state`` ~23660 elements, ``gis.fuel_class``,
``gis.elevation_m``, ``gis.wind_field``) with scalar ``*-load-*.p_mw`` power
sensors. The resulting columns have wildly different lengths, so the
``pd.DataFrame`` constructor raises::

    ValueError: All arrays must be of the same length

The crash happens inside palaestrAI itself -- in ``RolloutWorker._remember``'s
debug-log ``tail(1)`` and again in the SAC brain's ``memory.tail(1)``
``.objective.item()`` read -- so it cannot be avoided by filtering our own
sensors: ``rollout_worker`` stores ``request.sensors`` *before* the per-agent
``Filter`` runs.

What the shim does
------------------
:func:`install` patches ``_MuscleMemory._infos_to_df`` in memory:

* **Equal-length columns** take the upstream code path verbatim, so the common
  case is behaviourally identical to stock palaestrAI.
* **Ragged columns** fall back to a rectangular one-row frame whose cells hold
  the per-sensor arrays as objects. The frame stays indexable by sensor uid,
  which is all the logging and bookkeeping paths need.

Nothing on the training path consumes the ragged frame numerically: SAC trains
on the compact 17-dim observation, and offline teacher data arrives as ``.npz``
via ``load_transitions_into_buffer`` rather than through ``pretrain()``.

``site-packages`` is never modified -- this rebinds an attribute on the imported
class object at runtime, and :func:`install` is idempotent, so importing it from
both the RolloutWorker and the Learner process is safe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd

from palaestrai.agent.memory import _MuscleMemory

# Marks an already-patched function so repeated install() calls are no-ops.
_PATCH_FLAG = "_socal_ragged_safe"


def _infos_to_df(infos) -> pd.DataFrame:
    """Ragged-tolerant replacement for ``_MuscleMemory._infos_to_df``.

    Mirrors upstream exactly when every column flattens to the same length.
    """
    data = defaultdict(list)
    for i in infos:
        data[i.uid].append(i.value)
    # Upstream wraps values in np.ndarrays because pandas does not infer plain
    # lists of zero-dim arrays as arrays. Uids are not guaranteed unique, so a
    # single key may collect more than one value.
    wrapped_data: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        wrapped_data[key] = np.reshape(np.array(value), -1)

    if len({len(v) for v in wrapped_data.values()}) <= 1:
        # Uniform (or empty) -> byte-identical to stock palaestrAI.
        return pd.DataFrame(wrapped_data)

    # Ragged -> one row, one object cell per sensor uid. Each column is built
    # as a one-element object Series: assigning through ``.at`` would let
    # pandas unwrap a length-1 array into a 0-d scalar, silently changing the
    # shape of exactly the scalar power sensors the objective reads back.
    return pd.DataFrame(
        {key: pd.Series([value], dtype=object)
         for key, value in wrapped_data.items()}
    )


_infos_to_df._socal_ragged_safe = True


def installed() -> bool:
    """True when :func:`install` has already patched the Memory class."""
    return getattr(_MuscleMemory._infos_to_df, _PATCH_FLAG, False)


def install() -> bool:
    """Patch ``_MuscleMemory._infos_to_df`` in place; idempotent.

    Returns True when this call performed the patch, False when it was already
    installed. Safe to call from every process that touches a Memory.
    """
    if installed():
        return False
    _MuscleMemory._infos_to_df = staticmethod(_infos_to_df)
    return True
