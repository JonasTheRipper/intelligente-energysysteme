"""BurnedHousesObjective -- charge the agent for structures the fire destroys.

The structural axis of the multi-objective firefighter (see
:mod:`palaestrai_socal.agents.moo_objective`). Where
:class:`~palaestrai_socal.agents.saidi_objective.SaidiObjective` prices the
*electrical* consequence of the fire, this prices the *structural* one::

    reward = -(houses destroyed this step) / total houses / scale     (<= 0)

palaestrAI maximises the objective, so the charge is negative and a step that
destroys nothing scores exactly zero -- the same "charged per step" convention
SaidiObjective uses, which is what makes the two composable in
:class:`MooObjective`.

Why an increment rather than a survival level
---------------------------------------------
The obvious formulation is the surviving fraction, ``1 - burned/total``, which
is bounded in ``[0, 1]`` and reads naturally. It is a poor *reward*, for two
reasons:

* **Credit assignment.** It pays the agent at every step for houses that were
  never threatened. Over a 60-step episode a do-nothing policy collects ~55 of
  a possible 60, and the difference between a good and a bad policy is a few
  percent of the return -- the critic has to resolve the interesting signal
  against a large constant baseline.
* **Composability.** ``SaidiObjective`` returns a charge (<= 0). Summing a
  charge of ~1e-4 with a level of ~0.9 means the level decides the weighted
  sum outright, whatever the weights say.

The increment sums over an episode to exactly ``-(total burned)/total``, i.e.
``survival_fraction - 1``: **the same quantity, the same optimum**, delivered as
dense credit at the step where the loss actually happens. The survival fraction
remains the right *reporting* KPI, and :meth:`survival_fraction` exposes it.

Where the numbers come from
---------------------------
:class:`~palaestrai_socal.gis_world_env.GisWorldEnvironment` publishes the house
telemetry directly, because it owns the fuel raster and the cell states:

``gis.houses_total``
    class-9 (``wildfire_cma.cma.HOUSE_FUEL_CLASS``) cells in the raster.
``gis.houses_burned_this_step``
    house cells that became terminal (``BURNED_OUT``) during the step.
``gis.houses_burned_total``
    running sum of the above, used only for the reporting KPI.

So this objective needs no raster decoding, no lon/lat work and no state of its
own -- the environment already did that work, once, and stored it. The agent
must subscribe to those uids for them to reach its Memory.
"""

from __future__ import annotations

import logging
from typing import Optional

from palaestrai.agent.objective import Objective

from palaestrai_socal.agents import objective_support as osup

LOG = logging.getLogger("palaestrai_socal.agents.buho_objective")

#: Fraction of the settlement lost in ONE step that maps to a reward of -1.
#: 0.02 (2% of all houses in a single hour) is a severe step: the calibrated
#: Eaton run's worst hour destroys roughly that share.
HOUSES_SCALE = 0.02


class BurnedHousesObjective(Objective):
    """Return ``-(houses destroyed this step / total houses) / scale``.

    Construction follows :class:`SaidiObjective`: palaestrAI's
    ``load_with_params`` unpacks a YAML ``params:`` block as **keyword
    arguments**, so both call styles work and an explicit ``params`` dict wins.

    Parameters
    ----------
    scale:
        Fraction of the settlement destroyed in one step that normalises to
        ``-1`` (default 0.02). Divides the per-step loss so the reward is O(1)
        for the SAC/CQL critics and comparable to the other objectives.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        *,
        scale: float = HOUSES_SCALE,
    ):
        settings = {"scale": scale}
        if params:
            settings.update(params)
        super().__init__(params=settings)

        self._scale = float(settings["scale"]) or 1.0
        self._warned_no_sensors = False
        #: last values read, for the reporting KPI and for debugging.
        self.houses_total: float = 0.0
        self.houses_burned_total: float = 0.0

    # -- reporting ---------------------------------------------------------
    def survival_fraction(self) -> float:
        """Share of the settlement still standing, in ``[0, 1]``.

        The interpretable headline number -- report this, optimise the
        increment. Returns 1.0 before any telemetry has been seen.
        """
        if self.houses_total <= 0:
            return 1.0
        lost = min(self.houses_burned_total, self.houses_total)
        return 1.0 - (lost / self.houses_total)

    # -- objective ---------------------------------------------------------
    def internal_reward(self, memory, **kwargs) -> float:
        tail = memory.tail(1)

        total = osup.read_scalar(tail, "gis.houses_total")
        burned_this_step = osup.read_scalar(tail, "gis.houses_burned_this_step")

        if total is None or burned_this_step is None:
            # The agent does not subscribe to the house telemetry (or it has
            # not been published yet). Charge nothing rather than raising:
            # palaestrAI substitutes 0.0 for a raising objective anyway, but
            # with a "your results are screwed" log line every step.
            if not self._warned_no_sensors:
                LOG.warning(
                    "BurnedHousesObjective: gis.houses_total / "
                    "gis.houses_burned_this_step not in this agent's Memory; "
                    "reporting 0.0. Add them to the agent's sensor list."
                )
                self._warned_no_sensors = True
            return 0.0

        self.houses_total = float(total)
        cumulative = osup.read_scalar(tail, "gis.houses_burned_total")
        if cumulative is not None:
            self.houses_burned_total = float(cumulative)

        if total <= 0:
            # A raster with no settlement in it (a fine grid may contain none).
            # Nothing to lose, so nothing to charge.
            return 0.0

        lost_fraction = max(0.0, float(burned_this_step)) / float(total)
        return -lost_fraction / self._scale
