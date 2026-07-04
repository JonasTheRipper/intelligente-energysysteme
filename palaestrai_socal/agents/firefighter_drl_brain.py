"""FirefighterSacBrain -- the v0.7 DRL firefighter learner (CQL-bootstrapped).

The learning half of the v0.7 DRL firefighter. It is a thin subclass of the
hARL :class:`harl.SACBrain` that does two firefighter-specific things:

1.  **Sizes its networks to the compact 17-dim contract, not the raw grid.**
    The firefighter agent subscribes to large grid sensors (``gis.cell_state``
    etc.) so the deterministic tactics can act, but the *policy* consumes the
    17-feature summary the muscle builds
    (:func:`palaestrai_socal.agents.firefighter_drl.extract_obs`). Left alone,
    :meth:`SACBrain.setup` would build the actor/critic against the flattened
    grid sensors (thousands of dims) and the ``gis.cell_mutations`` Box actuator
    -- neither of which matches the ``Box(17)`` / ``Discrete(4)`` transitions the
    muscle and the offline harvester actually produce. So we temporarily swap in
    a synthetic ``Box(17)`` sensor + ``Discrete(4)`` actuator around the base
    ``setup()`` call, guaranteeing the nets are exactly the contract size.

2.  **Seeds the replay buffer with offline teacher transitions (CQL bootstrap).**
    When ``offline_npz`` is given, after the buffer exists we load the harvested
    ``(obs, action, next_obs, reward, done)`` transitions
    (:mod:`palaestrai_socal.agents.harvest_teacher_transitions`) via
    :meth:`SACBrain.load_transitions_into_buffer`. That flips the brain's
    ``_offline_data_present`` flag, so the CQL(H) conservative regulariser
    auto-enables (unless ``cql_enabled`` is overridden) and online SAC fine-tunes
    from a behaviour-cloned start instead of from scratch.

Params (experiment YAML ``brain.params``) are the base
:class:`harl.SACBrain` hyperparameters plus:

* ``offline_npz`` path to a harvested teacher ``.npz`` (optional; no bootstrap
  when omitted). Relative paths resolve against the process CWD (the repo root
  when launched with the testbed runtime conf).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Sequence

import numpy as np

from palaestrai.agent import ActuatorInformation, SensorInformation
from palaestrai.types import Box, Discrete

from harl import SACBrain

LOG = logging.getLogger("palaestrai_socal.agents.firefighter_drl_brain")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from palaestrai_socal.agents import firefighter_drl as drl  # noqa: E402


class FirefighterSacBrain(SACBrain):
    """SAC/CQL brain over the DRL firefighter's Box(17)/Discrete(4) contract."""

    def __init__(
        self,
        *args,
        offline_npz: Optional[str] = None,
        replay_size: int = int(1e6),
        fc_dims: Sequence[int] = (256, 256),
        activation: str = "torch.nn.ReLU",
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        gradient_steps: int = -1,
        batch_size: int = 100,
        update_after: int = 1000,
        update_every: int = 50,
        cql_alpha: float = 1.0,
        cql_enabled: Optional[bool] = None,
        cql_n_action_samples: int = 10,
        **kwargs,
    ):
        super().__init__(
            *args,
            replay_size=replay_size,
            fc_dims=fc_dims,
            activation=activation,
            gamma=gamma,
            polyak=polyak,
            lr=lr,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            update_after=update_after,
            update_every=update_every,
            cql_alpha=cql_alpha,
            cql_enabled=cql_enabled,
            cql_n_action_samples=cql_n_action_samples,
            **kwargs,
        )
        self._offline_npz = offline_npz

    # -- synthetic contract spaces ----------------------------------------
    @staticmethod
    def _contract_sensors() -> list:
        """A single Box(17) sensor matching the DRL observation contract."""
        space = Box(
            low=np.full(drl.OBS_DIM, -1.0, dtype=np.float32),
            high=np.full(drl.OBS_DIM, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        return [
            SensorInformation(
                value=np.zeros(drl.OBS_DIM, dtype=np.float32),
                space=space,
                uid="firefighter_drl.obs",
            )
        ]

    @staticmethod
    def _contract_actuators() -> list:
        """A single Discrete(N_TACTICS) actuator matching the action contract."""
        space = Discrete(drl.N_TACTICS)
        return [
            ActuatorInformation(
                value=0,
                space=space,
                uid="firefighter_drl.doctrine",
            )
        ]

    # -- setup: build nets at contract size, then seed offline buffer ------
    def setup(self):
        """Build actor/critic at Box(17)/Discrete(4), then CQL-bootstrap.

        The framework has populated ``self.sensors`` / ``self.actuators`` with
        the firefighter's *raw* grid sensors and cell-mutation actuator. We swap
        in the synthetic contract spaces so :meth:`SACBrain.setup` sizes the
        networks to ``OBS_DIM`` / ``N_TACTICS`` (the vectors the muscle actually
        emits), then restore the real lists so nothing else downstream breaks.
        """
        real_sensors = self.sensors
        real_actuators = self.actuators
        self.sensors = self._contract_sensors()
        self.actuators = self._contract_actuators()
        try:
            super().setup()
        finally:
            # keep the contract spaces on the brain: the SAC brain reads action
            # type / dims from them only inside setup(), and training transitions
            # come from data_from_muscle, so restoring is optional -- but we keep
            # the real lists to avoid surprising any other framework consumer.
            self.sensors = real_sensors
            self.actuators = real_actuators

        LOG.info(
            "FirefighterSacBrain nets built at obs_dim=%d act_dim=%d "
            "(replay=%d batch=%d cql_alpha=%.3g)",
            drl.OBS_DIM, drl.N_TACTICS, self.replay_size,
            self.batch_size, self.cql_alpha,
        )
        self._load_offline()

    # -- offline CQL bootstrap --------------------------------------------
    def _resolve_npz(self) -> Optional[str]:
        if not self._offline_npz:
            return None
        p = self._offline_npz
        if os.path.isabs(p) and os.path.exists(p):
            return p
        for cand in (p, os.path.join(_ROOT, p), os.path.join(os.getcwd(), p)):
            if os.path.exists(cand):
                return cand
        LOG.warning(
            "FirefighterSacBrain offline_npz %r not found; skipping bootstrap",
            self._offline_npz,
        )
        return None

    def _load_offline(self) -> int:
        """Load harvested teacher transitions into the replay buffer (if any)."""
        path = self._resolve_npz()
        if path is None:
            return 0
        data = np.load(path, allow_pickle=True)
        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"]).ravel()
        rewards = np.asarray(data["rewards"], dtype=np.float32).ravel()
        next_obs = np.asarray(data["next_obs"], dtype=np.float32)
        dones = np.asarray(data["dones"]).ravel()

        if obs.shape[1] != drl.OBS_DIM:
            raise ValueError(
                f"offline obs dim {obs.shape[1]} != contract {drl.OBS_DIM} "
                f"({path})"
            )
        n = obs.shape[0]
        # 5-tuples in load_transitions_into_buffer's layout:
        #   (obs, action, next_obs, reward, done).
        transitions = (
            (
                obs[i],
                np.asarray(actions[i]),
                next_obs[i],
                float(rewards[i]),
                bool(dones[i]),
            )
            for i in range(n)
        )
        added = self.load_transitions_into_buffer(transitions)
        LOG.info(
            "FirefighterSacBrain CQL bootstrap: loaded %d/%d offline "
            "transitions from %s (cql_enabled=%s)",
            added, n, os.path.basename(path), self.cql_enabled,
        )
        return added

    def __repr__(self) -> str:
        return (
            f"FirefighterSacBrain(name={self.name}, "
            f"obs_dim={drl.OBS_DIM}, act_dim={drl.N_TACTICS}, "
            f"offline_npz={self._offline_npz!r})"
        )
