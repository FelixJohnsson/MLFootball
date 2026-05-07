"""AttackerTeamRlEnv subclass: frozen opponent loaded from a checkpoint.

The opponent model is loaded *inside* the process that owns this env (one copy
per SubprocDualVecEnv worker).  Observations are built from the opponent's
native perspective — no mirroring — so a RIGHT-trained model drives the
RIGHT_TEAM directly, and vice versa.
"""

from __future__ import annotations

import numpy as np

from sim.actions import TickActions
from sim.attacker_team_env import AttackerTeamRlEnv


class TeamVsOpponentEnv(AttackerTeamRlEnv):
    """Two RL-controlled teammates vs a frozen PPO opponent (both players)."""

    def __init__(self, *, opponent_model_path: str, **kwargs) -> None:
        # No SharedPolicyRef — opponent is handled locally.
        super().__init__(opposing_team_policy_ref=None, **kwargs)
        self._opponent_path = opponent_model_path
        self._opponent = None  # lazy-loaded (avoids pickling a model)

    def _ensure_opponent_loaded(self) -> None:
        if self._opponent is None:
            from stable_baselines3 import PPO

            self._opponent = PPO.load(self._opponent_path, device="cpu")

    # -- overrides the hook added to AttackerTeamRlEnv -------------------------

    def _get_opposing_actions(self) -> tuple[TickActions, TickActions]:
        """Predict opponent actions using native ego observations (no mirroring)."""
        self._ensure_opponent_loaded()

        s = self.state
        if s.controlled_scores_into_right_net:
            # We control LEFT → opponent is RIGHT
            o1 = s.get_obs_right_team_primary()
            o2 = s.get_obs_right_team_partner()
        else:
            # We control RIGHT → opponent is LEFT
            o1 = s.get_obs_left_team_primary()
            o2 = s.get_obs_left_team_partner()

        a1, _ = self._opponent.predict(o1.astype(np.float32), deterministic=True)
        a2, _ = self._opponent.predict(o2.astype(np.float32), deterministic=True)
        return (
            TickActions(float(a1[0]), float(a1[1]), False, float(a1[2])),
            TickActions(float(a2[0]), float(a2[1]), False, float(a2[2])),
        )
