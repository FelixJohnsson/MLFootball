"""Two RL-controlled teammates vs mirrored-policy or heuristic opposing team."""

from __future__ import annotations

import math
import random

import numpy as np

from field import FIELD_DIAG, attack_target_goal_xy, left_attack_target_goal_xy

from sim.actions import TickActions
from sim.constants import DEFAULT_MAX_TICKS, TEAM_ATTACKER_ACT_DIM
from sim.reward_weights import DEFAULT_TEAM_RL_REWARD_WEIGHTS, TeamRlRewardWeights
from sim.frozen_opposing_team_policy import (
    SharedPolicyRef,
    opposing_team_tick_actions_shared_policy,
)
from sim.game import GameState
from sim.types import StepInfo


class AttackerTeamRlEnv:
    metadata = {"render_modes": []}
    render_mode: str | None = None

    def __init__(
        self,
        *,
        max_ticks: int = DEFAULT_MAX_TICKS,
        dt: float = 1.0 / 60.0,
        frame_skip: int = 4,
        stagnant_ball_env_steps: int = 20,
        stagnant_ball_penalty: float = -0.01,
        stagnant_ball_speed_eps: float = 12.0,
        opposing_team_policy_ref: SharedPolicyRef | None = None,
        opposing_team_policy_deterministic: bool = True,
        controlled_scores_into_right_net: bool = True,
        reward_weights: TeamRlRewardWeights | None = None,
    ) -> None:
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if stagnant_ball_env_steps < 1:
            raise ValueError("stagnant_ball_env_steps must be >= 1")
        self.dt = dt
        self.frame_skip = frame_skip
        self.stagnant_ball_env_steps = stagnant_ball_env_steps
        self.stagnant_ball_penalty = float(stagnant_ball_penalty)
        self.stagnant_ball_speed_eps = float(stagnant_ball_speed_eps)
        self.opposing_team_policy_ref = opposing_team_policy_ref
        self.opposing_team_policy_deterministic = bool(opposing_team_policy_deterministic)
        self.reward_weights = reward_weights or DEFAULT_TEAM_RL_REWARD_WEIGHTS
        self.state = GameState.new(
            max_ticks=max_ticks,
            controlled_scores_into_right_net=controlled_scores_into_right_net,
        )
        self._ball_stagnant_counter = 0
        self._prev_ball_scoring_target_dist = 0.0
        self._prev_ball_opposite_net_dist = 0.0
        self._prev_ctrl_p1_ball_dist = 0.0
        self._prev_ctrl_p2_ball_dist = 0.0

    def _scoring_target_and_opposite_net_xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(Controlled roster's scoring mouth, other mouth on the −x/+x axis).

        ``opposite_net`` is *not* "the goalkeeper" — it's the geometric far net: the mouth
        the controlled team must not dribble toward for this shaping term.
        """
        if self.state.controlled_scores_into_right_net:
            return attack_target_goal_xy(), left_attack_target_goal_xy()
        return left_attack_target_goal_xy(), attack_target_goal_xy()

    def _sync_prev_distances(self) -> None:
        (sx, sy), (ox, oy) = self._scoring_target_and_opposite_net_xy()
        cp1, cp2 = self.state.controlled_team_players()
        b = self.state.ball
        self._prev_ball_scoring_target_dist = math.hypot(b.x - sx, b.y - sy)
        self._prev_ball_opposite_net_dist = math.hypot(b.x - ox, b.y - oy)
        self._prev_ctrl_p1_ball_dist = math.hypot(cp1.x - b.x, cp1.y - b.y)
        self._prev_ctrl_p2_ball_dist = math.hypot(cp2.x - b.x, cp2.y - b.y)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.state.reset()
        self._ball_stagnant_counter = 0
        self._sync_prev_distances()

    def observation(self, ego: int) -> np.ndarray:
        assert ego in (0, 1)
        raw = (
            self.state.get_obs_controlled_primary()
            if ego == 0
            else self.state.get_obs_controlled_partner()
        )
        return np.clip(raw, -1.0, 1.0).astype(np.float32)

    def step_team(self, action: np.ndarray) -> tuple[float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float32).reshape(TEAM_ATTACKER_ACT_DIM)
        clipped = np.clip(a, -1.0, 1.0).astype(np.float32)
        tick_p1 = TickActions(
            float(clipped[0]),
            float(clipped[1]),
            False,
            float(clipped[2]),
        )
        tick_p2 = TickActions(
            float(clipped[3]),
            float(clipped[4]),
            False,
            float(clipped[5]),
        )
        total_reward = 0.0
        last_si: StepInfo | None = None
        scoring_net: str | None = None
        (target_x, target_y), (opp_net_x, opp_net_y) = self._scoring_target_and_opposite_net_xy()

        for _ in range(self.frame_skip):
            ref = self.opposing_team_policy_ref
            m = ref.model if ref is not None else None
            if m is not None:
                o1_tick, o2_tick = opposing_team_tick_actions_shared_policy(
                    self.state,
                    shared_policy_model=m,
                    deterministic=self.opposing_team_policy_deterministic,
                )
                last_si = self.state.step(
                    tick_p1,
                    self.dt,
                    controlled_player2_actions=tick_p2,
                    opposing_team_player1_actions=o1_tick,
                    opposing_team_player2_actions=o2_tick,
                )
            else:
                last_si = self.state.step(
                    tick_p1, self.dt, controlled_player2_actions=tick_p2
                )
            cp1, cp2 = self.state.controlled_team_players()
            b = self.state.ball
            curr_ball_target = math.hypot(b.x - target_x, b.y - target_y)
            curr_ball_opposite_net = math.hypot(b.x - opp_net_x, b.y - opp_net_y)
            curr_c1_ball = math.hypot(cp1.x - b.x, cp1.y - b.y)
            curr_c2_ball = math.hypot(cp2.x - b.x, cp2.y - b.y)
            total_reward += (
                self.reward_weights.ball_to_scoring_target_coef
                * (self._prev_ball_scoring_target_dist - curr_ball_target)
                / FIELD_DIAG
            )
            total_reward -= (
                self.reward_weights.ball_opposite_net_penalty_coef
                * (self._prev_ball_opposite_net_dist - curr_ball_opposite_net)
                / FIELD_DIAG
            )
            total_reward += (
                self.reward_weights.ctrl_primary_to_ball_coef
                * (self._prev_ctrl_p1_ball_dist - curr_c1_ball)
                / FIELD_DIAG
            )
            total_reward += (
                self.reward_weights.ctrl_partner_to_ball_coef
                * (self._prev_ctrl_p2_ball_dist - curr_c2_ball)
                / FIELD_DIAG
            )
            if self.state.controlled_scores_into_right_net:
                if last_si.scoring_net == "right":
                    total_reward += self.reward_weights.goal_correct_net_bonus
                if last_si.scoring_net == "left":
                    total_reward -= self.reward_weights.goal_wrong_net_penalty
            else:
                if last_si.scoring_net == "left":
                    total_reward += self.reward_weights.goal_correct_net_bonus
                if last_si.scoring_net == "right":
                    total_reward -= self.reward_weights.goal_wrong_net_penalty

            self._prev_ball_scoring_target_dist = curr_ball_target
            self._prev_ball_opposite_net_dist = curr_ball_opposite_net
            self._prev_ctrl_p1_ball_dist = curr_c1_ball
            self._prev_ctrl_p2_ball_dist = curr_c2_ball
            if last_si.scoring_net is not None:
                scoring_net = last_si.scoring_net
            if last_si.terminated or last_si.truncated:
                break

        assert last_si is not None

        total_reward -= self.reward_weights.step_penalty_per_env_step

        bspd = self.state.ball.speed()
        if bspd < self.stagnant_ball_speed_eps:
            self._ball_stagnant_counter += 1
        else:
            self._ball_stagnant_counter = 0
        if self._ball_stagnant_counter >= self.stagnant_ball_env_steps:
            stagnant_penalty = self.stagnant_ball_penalty * self._ball_stagnant_counter
            stagnant_penalty = max(stagnant_penalty, -1)
            total_reward += stagnant_penalty

        terminated = bool(last_si.terminated)
        truncated = bool(last_si.truncated)
        info = {
            "scoring_net": scoring_net,
            "score": self.state.score,
            "tick_count": self.state.tick_count,
            "ball_stagnant_env_steps": self._ball_stagnant_counter,
            "TimeLimit.truncated": truncated and not terminated,
        }
        return total_reward, terminated, truncated, info
