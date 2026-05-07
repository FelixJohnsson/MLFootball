"""Configurable reward coefficients for ``AttackerTeamRlEnv`` (dual or single trainers)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TeamRlRewardWeights:
    """Dense + sparse shaping for ``AttackerTeamRlEnv.step_team`` (edit per train script)."""

    step_penalty_per_env_step: float = 0.01
    ball_to_scoring_target_coef: float = 0.01
    ball_opposite_net_penalty_coef: float = 0.01
    #: Potential on distance-to-ball Δ for whichever controlled player is **closest** each tick.
    chase_nearest_to_ball_coef: float = 0.05
    #: Potential encouraging the **farther** player toward ``partner_support_ideal_ball_dist_px``.
    partner_support_ball_potential_coef: float = 0.001
    partner_support_ideal_ball_dist_px: float = 30.0
    #: Penalty scales from 0 (at threshold) up to coef when teammates overlap (breaks stacking).
    teammate_spread_threshold_px: float = 50
    teammate_spread_penalty_coef: float = 0.01
    #: Both controlled players within this distance (px) to ball while ball is slow ⇒ crowding / wedge hint.
    stagnant_crowding_ball_touch_px: float = 0.01
    #: Extra shaping every env ``step_team`` while ball slow and both teammates crowding it.
    stagnant_crowding_tick_penalty: float = 0
    #: Multiplier on the episodic stagnant penalty slab when crowded (ball still slow past threshold ticks).
    stagnant_crowding_stall_multiplier: float = 0
    #: Controlled players may leave the pitch (soft boundary); penalize excess distance outside [0,W]×[0,H].
    off_pitch_penalty_coef: float = 0.001
    #: Per env ``step_team``: subtract coef × max(0, opp_goals − our_goals) for the controlled roster.
    losing_per_goal_behind_coef: float = 0.005
    goal_correct_net_bonus: float = 80.0
    goal_wrong_net_penalty: float = 80.0


DEFAULT_TEAM_RL_REWARD_WEIGHTS = TeamRlRewardWeights()
