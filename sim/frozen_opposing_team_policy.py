"""Tick actions for **RIGHT_TEAM**'s pair using weights trained on **LEFT_TEAM** (+x net).

Observation/action mirroring lets a +x-trained checkpoint drive −x-facing roster geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.actions import TickActions
from sim.game import GameState
from sim.policy_mirror import mirror_action_for_right_trained_policy


def opposing_team_tick_actions_shared_policy(
    state: GameState,
    *,
    shared_policy_model: Any,
    deterministic: bool = True,
) -> tuple[TickActions, TickActions]:
    """``opposing_team_mirror_policy_observations()`` feeds pre-mirrored 25‑D tensors."""
    o1, o2 = state.opposing_team_mirror_policy_observations()
    ad1, _ = shared_policy_model.predict(o1.astype(np.float32), deterministic=deterministic)
    ad2, _ = shared_policy_model.predict(o2.astype(np.float32), deterministic=deterministic)
    m1 = mirror_action_for_right_trained_policy(np.asarray(ad1, dtype=np.float64))
    m2 = mirror_action_for_right_trained_policy(np.asarray(ad2, dtype=np.float64))
    return (
        TickActions(float(m1[0]), float(m1[1]), False, float(m1[2])),
        TickActions(float(m2[0]), float(m2[1]), False, float(m2[2])),
    )


class SharedPolicyRef:
    """Mutable handle to SB3 ``PPO`` weights read during opposing-team inference."""

    __slots__ = ("model",)

    def __init__(self) -> None:
        self.model: Any | None = None
