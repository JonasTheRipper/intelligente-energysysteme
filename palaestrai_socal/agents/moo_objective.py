"""MooObjective -- the firefighter's multi-objective utility function.

Combines the single-axis objectives into one scalar palaestrAI can maximise::

    reward = alpha * (saidi / saidi_norm) + beta * (houses / houses_norm)

Both terms are ``<= 0`` (each is a *charge* levied per step), so the total is
``<= 0`` and a perfect step -- no outage, no structures lost -- scores exactly
zero.

Why the normalisers exist (read this before tuning alpha/beta)
--------------------------------------------------------------
The axes are **not** natively commensurable, and a plain weighted sum of them is
a trap. Measured on the calibrated Eaton scenario:

=========================  ==========================  ====================
term                       typical per-step magnitude  reaches -1 when
=========================  ==========================  ====================
``SaidiObjective``         1e-4 .. 1e-3                never, in practice
``BurnedHousesObjective``  0 .. ~1                     2% of houses burn
=========================  ==========================  ====================

``SaidiObjective`` returns ``-delta_saidi / 60``, and the whole 60-step
no-firefighting baseline accrues only ~1.97 SAIDI -- roughly ``-5.5e-4`` per
step. Left unnormalised beside a term of order 1 that is a ratio of about
1,600:1: with ``alpha == beta`` the grid axis is numerically invisible while
*appearing* to carry half the weight.

So the weights carry *intent* ("how much do I care?") and the normalisers do the
*unit conversion* ("what counts as a bad step on this axis?"). The defaults put
each term near ``[-1, 0]`` for a bad step, so equal weights really do mean equal
influence.

**Do not rescale ``SaidiObjective`` itself to fix this.** Its ``/60`` is baked
into two other places: the ``AgentObjectiveTerminationCondition`` threshold
(``brain_avg30: -0.0002``) in the long DRL run, and every reward in the
harvested offline replay buffer
(:mod:`palaestrai_socal.agents.harvest_teacher_transitions` computes
``-dsaidi / saidi_scale`` with the identical constant). Changing it silently
invalidates the CQL bootstrap.

Consequence for the CQL bootstrap
---------------------------------
Switching the firefighter from ``SaidiObjective`` to ``MooObjective`` changes
what the online reward *means*, but the offline ``.npz`` still holds pure-SAIDI
rewards. Either re-harvest with matching rewards, or start the multi-objective
agent without the offline bootstrap (``offline_npz`` unset). Mixing them trains
the critic on two different reward functions.

Verifying the balance empirically
---------------------------------
:attr:`last_terms` records the raw, normalised and weighted contribution of each
axis on the most recent call, and :meth:`running_shares` reports what fraction
of the total charge each axis has carried so far. Log those over a pilot episode
and adjust the normalisers to the scenario before committing to a long training
run -- the defaults are calibrated for Eaton, not for every grid.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from palaestrai.agent.objective import Objective

from palaestrai_socal.agents import _memory_compat
from palaestrai_socal.agents.buho_objective import BurnedHousesObjective
from palaestrai_socal.agents.saidi_objective import SaidiObjective

LOG = logging.getLogger("palaestrai_socal.agents.moo_objective")

# Reading both axes requires the agent to subscribe to grid rasters *and*
# scalar power/house sensors, which is exactly the ragged mix stock palaestrAI
# cannot tabulate. Installing here covers any process that loads the objective.
_memory_compat.install()

#: Per-step charge on each axis that counts as "a bad step" -> normalised -1.
#: SAIDI: ~2x the calibrated no-firefighting baseline step (-5.5e-4).
SAIDI_NORM = 1.0e-3
#: Houses: BurnedHousesObjective already normalises by its own ``scale``.
HOUSES_NORM = 1.0


class MooObjective(Objective):
    """Weighted, unit-normalised sum of the grid-outage and structural axes.

    Construction follows :class:`SaidiObjective`: palaestrAI's
    ``load_with_params`` unpacks a YAML ``params:`` block as **keyword
    arguments**, so both call styles work and an explicit ``params`` dict wins.

    The sub-objectives are constructed **once**, here, and reused every step.
    :class:`BurnedHousesObjective` carries reporting state across steps, so
    rebuilding it per call would reset it continuously.

    Parameters
    ----------
    alpha, beta:
        Weights for grid outage and burned structures. Any non-negative values;
        they need not sum to 1 (a warning is logged if they do not, since that
        rescales the whole objective -- and with it any
        ``AgentObjectiveTerminationCondition`` threshold).
    saidi_norm, houses_norm:
        Per-step charge on each axis that normalises to ``-1``. See the module
        docstring -- these, not the weights, make the terms comparable.
    saidi_params, houses_params:
        Optional dicts forwarded to the sub-objective constructors so the
        experiment YAML can configure them (e.g.
        ``saidi_params: {base_served_mw: 35000.0}``). Without this the
        sub-objectives silently take their defaults, which suit the normalised
        testbed grid only by coincidence.
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        saidi_norm: float = SAIDI_NORM,
        houses_norm: float = HOUSES_NORM,
        saidi_params: Optional[dict] = None,
        houses_params: Optional[dict] = None,
    ):
        settings = {
            "alpha": alpha,
            "beta": beta,
            "saidi_norm": saidi_norm,
            "houses_norm": houses_norm,
            "saidi_params": saidi_params or {},
            "houses_params": houses_params or {},
        }
        if params:
            settings.update(params)
        super().__init__(params=settings)

        self.alpha = float(settings["alpha"])
        self.beta = float(settings["beta"])
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if value < 0.0:
                raise ValueError(
                    f"MooObjective: {name} must be >= 0 (got {value}); a "
                    "negative weight turns that axis into a goal to maximise "
                    "harm on it."
                )
        weight_sum = self.alpha + self.beta
        if weight_sum <= 0.0:
            raise ValueError(
                "MooObjective: at least one of alpha/beta must be > 0."
            )
        if not np.isclose(weight_sum, 1.0):
            LOG.warning(
                "MooObjective weights sum to %.3f, not 1.0; the combined "
                "reward is scaled by that factor, which shifts any "
                "AgentObjectiveTerminationCondition threshold with it.",
                weight_sum,
            )

        self._norms = {
            "saidi": float(settings["saidi_norm"]) or 1.0,
            "houses": float(settings["houses_norm"]) or 1.0,
        }

        self.saidi_objective = SaidiObjective(
            params=dict(settings["saidi_params"])
        )
        self.burned_houses_objective = BurnedHousesObjective(
            params=dict(settings["houses_params"])
        )

        #: Raw / normalised / weighted contribution of each axis, last call.
        self.last_terms: Dict[str, Dict[str, float]] = {}
        self._cumulative: Dict[str, float] = {"saidi": 0.0, "houses": 0.0}

    # -- diagnostics -------------------------------------------------------
    def running_shares(self) -> Dict[str, float]:
        """Fraction of the total charge so far carried by each axis.

        A quick sanity check that the normalisers are doing their job: if one
        axis reads 0.99 the objective is effectively single-objective.
        """
        total = sum(abs(v) for v in self._cumulative.values())
        if total <= 0.0:
            return {k: 0.0 for k in self._cumulative}
        return {k: abs(v) / total for k, v in self._cumulative.items()}

    # -- objective ---------------------------------------------------------
    def internal_reward(self, memory, **kwargs) -> float:
        raw = {
            "saidi": float(
                self.saidi_objective.internal_reward(memory, **kwargs) or 0.0
            ),
            "houses": float(
                self.burned_houses_objective.internal_reward(memory, **kwargs)
                or 0.0
            ),
        }
        weights = {"saidi": self.alpha, "houses": self.beta}

        weighted = {
            axis: weights[axis] * (value / self._norms[axis])
            for axis, value in raw.items()
        }
        for axis, value in weighted.items():
            self._cumulative[axis] += value

        self.last_terms = {
            axis: {
                "raw": raw[axis],
                "normalised": raw[axis] / self._norms[axis],
                "weighted": weighted[axis],
            }
            for axis in raw
        }
        # Every term is a charge (<= 0), so the sum is too; clip float noise.
        return min(0.0, float(sum(weighted.values())))
