"""Configurable reward coefficients for ``AttackerTeamRlEnv`` (dual or single trainers)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamRlRewardWeights:
    """Dense + sparse shaping for ``AttackerTeamRlEnv.step_team`` (edit per train script)."""

    step_penalty_per_env_step: float = 0.01
    ball_to_scoring_target_coef: float = 8.0
    ball_opposite_net_penalty_coef: float = 4.0
    ctrl_primary_to_ball_coef: float = 1.0
    ctrl_partner_to_ball_coef: float = 1.0
    goal_correct_net_bonus: float = 50.0
    goal_wrong_net_penalty: float = 50.0


DEFAULT_TEAM_RL_REWARD_WEIGHTS = TeamRlRewardWeights()
