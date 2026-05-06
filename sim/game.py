"""Pure simulation stepping — owns mutable game state (Gymnasium-style skeleton).

Convention: **LEFT_TEAM** roster (``ATTACK_RIGHT_GOAL_*`` spawn) scores **+x (right)**;
**RIGHT_TEAM** roster (``ATTACK_LEFT_GOAL_*``) scores **−x (left)**.
Four ``Player`` slots: ``left_team_player{1,2}`` and ``right_team_player{1,2}``.
``controlled_scores_into_right_net`` selects which team's pair is RL-controlled (+x scoring = LEFT_TEAM).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics import Ball, create_ball, resolve_ball_player, resolve_player_player
from physics.ball import REAPPLY_PITCH_AFTER_PHYS_LOOP, REAPPLY_PITCH_PER_PHYS_ITERATION
from field import attack_target_goal_xy, left_attack_target_goal_xy

from player import (
    HEURISTIC_FROZEN_TEAM_CHASE_SCALE,
    Player,
    create_attack_left_goal_pair,
    create_attack_right_goal_pair,
    place_attack_left_goal_team_no_overlap,
    place_attack_right_goal_team_no_overlap,
)

from sim.actions import TickActions
from sim.constants import (
    DEFAULT_MAX_TICKS,
    KICK_COOLDOWN_TICKS,
    KICK_FLASH_TICKS,
    KICK_IMPULSE,
    FROZEN_TEAM_HEURISTIC_CHASE,
)
from sim.observation import build_observation
from sim.policy_mirror import mirror_observation_for_right_trained_policy
from sim.types import StepInfo


@dataclass(slots=True)
class GameState:
    ball: Ball
    left_team_player1: Player
    left_team_player2: Player
    right_team_player1: Player
    right_team_player2: Player
    tick_count: int = 0
    max_ticks: int = DEFAULT_MAX_TICKS
    score: tuple[int, int] = (0, 0)
    opponent: Player | None = None
    kick_cool_opponent_expire_at: int = 0
    left_team_p1_kick_cool_expire_at: int = 0
    left_team_p2_kick_cool_expire_at: int = 0
    right_team_p1_kick_cool_expire_at: int = 0
    right_team_p2_kick_cool_expire_at: int = 0
    # ``True``: RL-controlled pair is LEFT_TEAM (+x primary). ``False``: RIGHT_TEAM (−x primary).
    controlled_scores_into_right_net: bool = True

    def controlled_team_players(self) -> tuple[Player, Player]:
        if self.controlled_scores_into_right_net:
            return self.left_team_player1, self.left_team_player2
        return self.right_team_player1, self.right_team_player2

    def opposing_team_players(self) -> tuple[Player, Player]:
        if self.controlled_scores_into_right_net:
            return self.right_team_player1, self.right_team_player2
        return self.left_team_player1, self.left_team_player2

    @classmethod
    def new(
        cls,
        *,
        max_ticks: int = DEFAULT_MAX_TICKS,
        controlled_scores_into_right_net: bool = True,
    ) -> GameState:
        minus_x_roster_a, minus_x_roster_b = create_attack_left_goal_pair()
        plus_x_roster_a, plus_x_roster_b = create_attack_right_goal_pair()
        return cls(
            ball=create_ball(),
            left_team_player1=plus_x_roster_a,
            left_team_player2=plus_x_roster_b,
            right_team_player1=minus_x_roster_a,
            right_team_player2=minus_x_roster_b,
            tick_count=0,
            max_ticks=max_ticks,
            score=(0, 0),
            opponent=None,
            kick_cool_opponent_expire_at=0,
            controlled_scores_into_right_net=controlled_scores_into_right_net,
        )

    def _controlled_attack_goal(self) -> tuple[float, float]:
        return (
            attack_target_goal_xy()
            if self.controlled_scores_into_right_net
            else left_attack_target_goal_xy()
        )

    def _opposing_attack_goal(self) -> tuple[float, float]:
        return (
            left_attack_target_goal_xy()
            if self.controlled_scores_into_right_net
            else attack_target_goal_xy()
        )

    def _clear_kick_cooldowns_after_restart(self) -> None:
        self.left_team_p1_kick_cool_expire_at = 0
        self.left_team_p2_kick_cool_expire_at = 0
        self.right_team_p1_kick_cool_expire_at = 0
        self.right_team_p2_kick_cool_expire_at = 0
        self.kick_cool_opponent_expire_at = 0
        for pl in (
            self.left_team_player1,
            self.left_team_player2,
            self.right_team_player1,
            self.right_team_player2,
        ):
            pl.kick_flash_expire_at = 0
        if self.opponent is not None:
            self.opponent.kick_flash_expire_at = 0

    def _pulse_kick_flash(self, player: Player) -> None:
        """Render hint: yellow ring until ``tick_count`` reaches newly set exclusive bound."""
        player.kick_flash_expire_at = self.tick_count + KICK_FLASH_TICKS

    def reset(self) -> GameState:
        self.tick_count = 0
        self.score = (0, 0)
        self._clear_kick_cooldowns_after_restart()
        self.ball.reset_center()
        place_attack_right_goal_team_no_overlap(self.left_team_player1, self.left_team_player2)
        place_attack_left_goal_team_no_overlap(self.right_team_player1, self.right_team_player2)
        if self.opponent is not None:
            self.opponent.reset_opponent_to_start()
        return self

    def _restart_after_goal(self) -> None:
        self.ball.reset_center()
        place_attack_right_goal_team_no_overlap(self.left_team_player1, self.left_team_player2)
        place_attack_left_goal_team_no_overlap(self.right_team_player1, self.right_team_player2)
        if self.opponent is not None:
            self.opponent.reset_opponent_to_start()
        self._clear_kick_cooldowns_after_restart()

    def get_obs_controlled_primary(self) -> np.ndarray:
        cp1, cp2 = self.controlled_team_players()
        ot1, ot2 = self.opposing_team_players()
        return build_observation(
            ego_player=cp1,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=(
                self.left_team_p1_kick_cool_expire_at
                if self.controlled_scores_into_right_net
                else self.right_team_p1_kick_cool_expire_at
            ),
            teammate_player=cp2,
            opposing_team_other_slot1_player=ot1,
            opposing_team_other_slot2_player=ot2,
            attack_goal_xy=self._controlled_attack_goal(),
        )

    def get_obs_controlled_partner(self) -> np.ndarray:
        cp1, cp2 = self.controlled_team_players()
        ot1, ot2 = self.opposing_team_players()
        return build_observation(
            ego_player=cp2,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=(
                self.left_team_p2_kick_cool_expire_at
                if self.controlled_scores_into_right_net
                else self.right_team_p2_kick_cool_expire_at
            ),
            teammate_player=cp1,
            opposing_team_other_slot1_player=ot1,
            opposing_team_other_slot2_player=ot2,
            attack_goal_xy=self._controlled_attack_goal(),
        )

    def get_obs_left_team_primary(self) -> np.ndarray:
        """25-D ego for ``left_team_player1`` (+x scoring), partner L2 vs RIGHT_TEAM."""
        lt1, lt2 = self.left_team_player1, self.left_team_player2
        rt1, rt2 = self.right_team_player1, self.right_team_player2
        return build_observation(
            ego_player=lt1,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=self.left_team_p1_kick_cool_expire_at,
            teammate_player=lt2,
            opposing_team_other_slot1_player=rt1,
            opposing_team_other_slot2_player=rt2,
            attack_goal_xy=attack_target_goal_xy(),
        )

    def get_obs_left_team_partner(self) -> np.ndarray:
        """25-D ego for ``left_team_player2`` (+x scoring), partner L1."""
        lt1, lt2 = self.left_team_player1, self.left_team_player2
        rt1, rt2 = self.right_team_player1, self.right_team_player2
        return build_observation(
            ego_player=lt2,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=self.left_team_p2_kick_cool_expire_at,
            teammate_player=lt1,
            opposing_team_other_slot1_player=rt1,
            opposing_team_other_slot2_player=rt2,
            attack_goal_xy=attack_target_goal_xy(),
        )

    def get_obs_right_team_primary(self) -> np.ndarray:
        """25-D ego for ``right_team_player1`` (−x scoring), partner R2 vs LEFT_TEAM."""
        lt1, lt2 = self.left_team_player1, self.left_team_player2
        rt1, rt2 = self.right_team_player1, self.right_team_player2
        return build_observation(
            ego_player=rt1,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=self.right_team_p1_kick_cool_expire_at,
            teammate_player=rt2,
            opposing_team_other_slot1_player=lt1,
            opposing_team_other_slot2_player=lt2,
            attack_goal_xy=left_attack_target_goal_xy(),
        )

    def get_obs_right_team_partner(self) -> np.ndarray:
        """25-D ego for ``right_team_player2`` (−x scoring), partner R1 vs LEFT_TEAM."""
        lt1, lt2 = self.left_team_player1, self.left_team_player2
        rt1, rt2 = self.right_team_player1, self.right_team_player2
        return build_observation(
            ego_player=rt2,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=self.right_team_p2_kick_cool_expire_at,
            teammate_player=rt1,
            opposing_team_other_slot1_player=lt1,
            opposing_team_other_slot2_player=lt2,
            attack_goal_xy=left_attack_target_goal_xy(),
        )

    def opposing_team_mirror_policy_observations(self) -> tuple[np.ndarray, np.ndarray]:
        """Ego vectors for opposing-team pawns running a mirrored +x-trained policy."""
        cp1, cp2 = self.controlled_team_players()
        ot1, ot2 = self.opposing_team_players()
        og = self._opposing_attack_goal()
        raw_ot1 = build_observation(
            ego_player=ot1,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=(
                self.right_team_p1_kick_cool_expire_at
                if self.controlled_scores_into_right_net
                else self.left_team_p1_kick_cool_expire_at
            ),
            teammate_player=ot2,
            opposing_team_other_slot1_player=cp1,
            opposing_team_other_slot2_player=cp2,
            attack_goal_xy=og,
        )
        raw_ot2 = build_observation(
            ego_player=ot2,
            ball=self.ball,
            tick_count=self.tick_count,
            kick_cool_expire_at=(
                self.right_team_p2_kick_cool_expire_at
                if self.controlled_scores_into_right_net
                else self.left_team_p2_kick_cool_expire_at
            ),
            teammate_player=ot1,
            opposing_team_other_slot1_player=cp1,
            opposing_team_other_slot2_player=cp2,
            attack_goal_xy=og,
        )
        return (
            mirror_observation_for_right_trained_policy(raw_ot1),
            mirror_observation_for_right_trained_policy(raw_ot2),
        )

    def step(
        self,
        controlled_player1_actions: TickActions,
        dt: float,
        *,
        human_actions: TickActions | None = None,
        controlled_player2_actions: TickActions | None = None,
        opposing_team_player1_actions: TickActions | None = None,
        opposing_team_player2_actions: TickActions | None = None,
    ) -> StepInfo:
        self.tick_count += 1

        cp1, cp2 = self.controlled_team_players()
        op1, op2 = self.opposing_team_players()

        cp1.apply_movement(float(controlled_player1_actions.move_x), float(controlled_player1_actions.move_y), dt)
        if controlled_player2_actions is not None:
            cp2.apply_movement(
                float(controlled_player2_actions.move_x),
                float(controlled_player2_actions.move_y),
                dt,
            )
        else:
            cp2.apply_movement(0.0, 0.0, dt)

        opp = self.opponent
        if opp is not None and human_actions is not None:
            opp.apply_movement(float(human_actions.move_x), float(human_actions.move_y), dt)

        if controlled_player1_actions.reset_ball_only or (
            human_actions is not None and human_actions.reset_ball_only
        ):
            self.ball.reset_center()

        kp = float(max(0.0, min(1.0, controlled_player1_actions.kick_power)))
        if kp > 0.0:
            if self.controlled_scores_into_right_net:
                if self.tick_count >= self.left_team_p1_kick_cool_expire_at and cp1.try_kick(
                    self.ball, impulse=KICK_IMPULSE * kp
                ):
                    self.left_team_p1_kick_cool_expire_at = self.tick_count + KICK_COOLDOWN_TICKS
                    self._pulse_kick_flash(cp1)
            else:
                if self.tick_count >= self.right_team_p1_kick_cool_expire_at and cp1.try_kick(
                    self.ball, impulse=KICK_IMPULSE * kp
                ):
                    self.right_team_p1_kick_cool_expire_at = self.tick_count + KICK_COOLDOWN_TICKS
                    self._pulse_kick_flash(cp1)

        if controlled_player2_actions is not None:
            k2 = float(max(0.0, min(1.0, controlled_player2_actions.kick_power)))
            if k2 > 0.0:
                if self.controlled_scores_into_right_net:
                    if self.tick_count >= self.left_team_p2_kick_cool_expire_at and cp2.try_kick(
                        self.ball, impulse=KICK_IMPULSE * k2
                    ):
                        self.left_team_p2_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(cp2)
                else:
                    if self.tick_count >= self.right_team_p2_kick_cool_expire_at and cp2.try_kick(
                        self.ball, impulse=KICK_IMPULSE * k2
                    ):
                        self.right_team_p2_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(cp2)

        if (
            opp is not None
            and human_actions is not None
            and self.tick_count >= self.kick_cool_opponent_expire_at
        ):
            kpo = float(max(0.0, min(1.0, human_actions.kick_power)))
            if kpo > 0.0:
                if opp.try_kick(self.ball, impulse=KICK_IMPULSE * kpo):
                    self.kick_cool_opponent_expire_at = self.tick_count + KICK_COOLDOWN_TICKS
                    self._pulse_kick_flash(opp)

        if opposing_team_player1_actions is not None:
            op1.apply_movement(
                float(opposing_team_player1_actions.move_x),
                float(opposing_team_player1_actions.move_y),
                dt,
            )
        elif FROZEN_TEAM_HEURISTIC_CHASE:
            op1.apply_chase_toward_ball(self.ball, HEURISTIC_FROZEN_TEAM_CHASE_SCALE, dt)

        if opposing_team_player2_actions is not None:
            op2.apply_movement(
                float(opposing_team_player2_actions.move_x),
                float(opposing_team_player2_actions.move_y),
                dt,
            )
        elif FROZEN_TEAM_HEURISTIC_CHASE:
            op2.apply_chase_toward_ball(self.ball, HEURISTIC_FROZEN_TEAM_CHASE_SCALE, dt)

        if opposing_team_player1_actions is not None:
            okp = float(max(0.0, min(1.0, opposing_team_player1_actions.kick_power)))
            if okp > 0.0:
                if self.controlled_scores_into_right_net:
                    if self.tick_count >= self.right_team_p1_kick_cool_expire_at and op1.try_kick(
                        self.ball, impulse=KICK_IMPULSE * okp
                    ):
                        self.right_team_p1_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(op1)
                else:
                    if self.tick_count >= self.left_team_p1_kick_cool_expire_at and op1.try_kick(
                        self.ball, impulse=KICK_IMPULSE * okp
                    ):
                        self.left_team_p1_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(op1)

        if opposing_team_player2_actions is not None:
            ok2 = float(max(0.0, min(1.0, opposing_team_player2_actions.kick_power)))
            if ok2 > 0.0:
                if self.controlled_scores_into_right_net:
                    if self.tick_count >= self.right_team_p2_kick_cool_expire_at and op2.try_kick(
                        self.ball, impulse=KICK_IMPULSE * ok2
                    ):
                        self.right_team_p2_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(op2)
                else:
                    if self.tick_count >= self.left_team_p2_kick_cool_expire_at and op2.try_kick(
                        self.ball, impulse=KICK_IMPULSE * ok2
                    ):
                        self.left_team_p2_kick_cool_expire_at = (
                            self.tick_count + KICK_COOLDOWN_TICKS
                        )
                        self._pulse_kick_flash(op2)

        self.ball.update(dt)

        lt1, lt2 = self.left_team_player1, self.left_team_player2
        rt1, rt2 = self.right_team_player1, self.right_team_player2

        for _ in range(6):
            for pl in (lt1, lt2, rt1, rt2):
                resolve_ball_player(self.ball, pl)
            if opp is not None:
                resolve_ball_player(self.ball, opp)
                for pl in (lt1, lt2, rt1, rt2):
                    resolve_player_player(opp, pl)
            pairs = (
                (lt1, lt2),
                (lt1, rt1),
                (lt1, rt2),
                (lt2, rt1),
                (lt2, rt2),
                (rt1, rt2),
            )
            for a, b in pairs:
                resolve_player_player(a, b)

            self.ball.reapply_pitch_collisions(REAPPLY_PITCH_PER_PHYS_ITERATION)

        self.ball.reapply_pitch_collisions(REAPPLY_PITCH_AFTER_PHYS_LOOP)
        self.ball._cap_speed()

        terminated = False
        net = self.ball.which_goal_scored()
        if net == "left":
            lg, rg = self.score
            self.score = (lg + 1, rg)
            self._restart_after_goal()
            terminated = True
        elif net == "right":
            lg, rg = self.score
            self.score = (lg, rg + 1)
            self._restart_after_goal()
            terminated = True

        truncated = (not terminated) and self.tick_count >= self.max_ticks
        scoring = net if terminated else None
        return StepInfo(terminated=terminated, truncated=truncated, scoring_net=scoring)
