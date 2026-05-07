"""Tick actions for an opposing pair driven by a checkpoint trained on **LEFT_TEAM** (+x net).

When the **controlled** roster is LEFT: opposing RIGHT uses mirrored obs/actions.

When the **controlled** roster is RIGHT: opposing LEFT uses **native** LEFT obs (no mirror); the
checkpoint layout must still match **left-attacking** training (e.g. ``train_ppo_single --left_team``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.actions import TickActions
from sim.game import GameState
from sim.policy_mirror import mirror_action_for_right_trained_policy


def _actions_array_to_tick(actions: Any) -> TickActions:
    a = np.asarray(actions, dtype=np.float32).reshape(-1)
    return TickActions(float(a[0]), float(a[1]), False, float(a[2]))


def _opposing_left_native_tick_actions(
    state: GameState,
    shared_policy_model: Any,
    *,
    deterministic: bool,
) -> tuple[TickActions, TickActions]:
    """LEFT roster (blues): native primary/partner obs — for frozen +x-trained checkpoints."""
    o1 = np.clip(state.get_obs_left_team_primary(), -1.0, 1.0).astype(np.float32)
    o2 = np.clip(state.get_obs_left_team_partner(), -1.0, 1.0).astype(np.float32)
    ad1, _ = shared_policy_model.predict(o1, deterministic=deterministic)
    ad2, _ = shared_policy_model.predict(o2, deterministic=deterministic)
    return _actions_array_to_tick(ad1), _actions_array_to_tick(ad2)


def opposing_team_tick_actions_shared_policy(
    state: GameState,
    *,
    shared_policy_model: Any,
    deterministic: bool = True,
) -> tuple[TickActions, TickActions]:
    """Drive opposing players with a LEFT-trained checkpoint.

    • Controlled **LEFT** → mirror obs on greens (RIGHT_TEAM).
    • Controlled **RIGHT** → native LEFT obs on blues (same layout as ``watch --model_left``).
    """
    if not state.controlled_scores_into_right_net:
        return _opposing_left_native_tick_actions(
            state, shared_policy_model, deterministic=deterministic
        )
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
