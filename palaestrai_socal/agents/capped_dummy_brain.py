"""CappedDummyBrain -- a DummyBrain that bounds its replay Memory.

palaestrAI's :class:`~palaestrai.agent.learner.Learner` appends every
``MuscleUpdateRequest``'s ``sensor_readings`` to ``Brain.memory`` on *every*
step, unconditionally -- it does not check whether the brain actually learns
(see ``palaestrai/agent/learner.py`` ~line 219). ``Brain.memory`` is created
with the default ``Memory(size_limit=1e6)`` entries, i.e. effectively unbounded
for any realistic episode.

For the SoCal v0.2 agents this is catastrophic: the wildfire agent subscribes to
three *full-grid* sensors (``gis.fuel_class``/``gis.elevation_m``/
``gis.cell_state``), each ``nrows*ncols`` float64. At 600x760 that is ~34 MB of
sensor history appended to this brain's Memory *per step*; over 120 steps the
wildfire brain process alone grows by ~4 GB (measured), and the damage_mapper
brain adds ~1.3 GB. Two brains that never train then hoard >5 GB and the kernel
OOM-kills the run on the 7.8 GB box.

Since a DummyBrain never reads its Memory (``thinking`` only echoes the muscle's
objective), retaining that history serves no purpose. This subclass caps the
Memory to the last ``memory_size_limit`` entries, which keeps each brain
process flat (a few hundred MB) instead of multi-GB. ``setup()`` is the
palaestrAI-sanctioned hook for adjusting ``Memory.size_limit`` and is guaranteed
to run in the brain's own process before the main loop.
"""

from __future__ import annotations

from palaestrai.agent.dummy_brain import DummyBrain

from palaestrai_socal.agents import _memory_compat

# The Learner appends the muscle's sensor readings to the brain's Memory on
# every step and then reads ``memory.tail(1)`` back for a debug log -- eagerly,
# because logging evaluates its arguments. For the firefighter that mix is
# ragged (grid rasters + scalar power/house sensors), which raises in stock
# palaestrAI. This is a separate process from the RolloutWorker, so it needs
# its own install(); the call is idempotent.
_memory_compat.install()


class CappedDummyBrain(DummyBrain):
    """A :class:`DummyBrain` whose replay Memory is bounded.

    Parameters
    ----------
    memory_size_limit : int
        Maximum number of step-entries the brain's Memory retains. The default
        of 2 keeps only the most recent entries (a DummyBrain never uses them),
        which prevents the full-grid sensor history from accumulating.
    """

    def __init__(self, memory_size_limit: int = 2):
        super().__init__()
        self._memory_size_limit = int(memory_size_limit)
        # Set it eagerly too; ``Memory.append`` reads ``size_limit`` dynamically
        # and truncates after every append, so this alone already bounds growth.
        self._memory.size_limit = self._memory_size_limit

    def setup(self):
        # Re-assert in the brain's own process (the documented place to set the
        # Memory size limit), in case the framework rebinds memory before run().
        super().setup()
        self._memory.size_limit = self._memory_size_limit
