"""Observation vector — pure numpy; no pygame imports."""

from __future__ import annotations

import math

import numpy as np

from field import FIELD_DIAG, FIELD_H, FIELD_W, attack_target_goal_xy
from player import PLAYER_MAX_SPEED, Player

from sim.constants import KICK_COOLDOWN_TICKS, MATCH_WIN_GOALS


def _four_relative_to_ego(ego: Player, other: Player | None) -> tuple[float, float, float, float]:
    if other is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        (other.x - ego.x) / FIELD_W,
        (other.y - ego.y) / FIELD_H,
        other.vx / PLAYER_MAX_SPEED,
        other.vy / PLAYER_MAX_SPEED,
    )


def build_observation(
    *,
    ego_player: Player,
    ball,
    tick_count: int,
    kick_cool_expire_at: int,
    score: tuple[int, int],
    teammate_player: Player | None = None,
    opposing_team_other_slot1_player: Player | None = None,
    opposing_team_other_slot2_player: Player | None = None,
    attack_goal_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    First 13 dims: ego kinematics + ball-relative + mouth vector + cooldown.
    Then teammate + two opposing-team anchors (relative kinematics × 4 each). Zeros when missing.
    Last 2 dims: ``score[0] / MATCH_WIN_GOALS``, ``score[1] / MATCH_WIN_GOALS`` (``GameState.score``:
    left-net tally, right-net tally — see README).

    Default ``attack_goal_xy`` (+x mouth) matches LEFT_TEAM training; callers pass −x mouth for RIGHT_TEAM.
    """
    if attack_goal_xy is None:
        gx, gy = attack_target_goal_xy()
    else:
        gx, gy = float(attack_goal_xy[0]), float(attack_goal_xy[1])

    ox = gx - ball.x
    oy = gy - ball.y
    rel_px = (ball.x - ego_player.x) / FIELD_W
    rel_py = (ball.y - ego_player.y) / FIELD_H
    d_pb = math.hypot(ball.x - ego_player.x, ball.y - ego_player.y) / FIELD_DIAG
    d_bg = math.hypot(ox, oy) / FIELD_DIAG

    kd = min(max(0, kick_cool_expire_at - tick_count), KICK_COOLDOWN_TICKS) / KICK_COOLDOWN_TICKS

    head = (
        ego_player.x / FIELD_W,
        ego_player.y / FIELD_H,
        ego_player.vx / PLAYER_MAX_SPEED,
        ego_player.vy / PLAYER_MAX_SPEED,
        rel_px,
        rel_py,
        ball.vx / ball.max_speed,
        ball.vy / ball.max_speed,
        ox / FIELD_W,
        oy / FIELD_H,
        d_pb,
        d_bg,
        kd,
    )
    g = float(MATCH_WIN_GOALS)
    s0, s1 = float(score[0]) / g, float(score[1]) / g
    tail = (
        *_four_relative_to_ego(ego_player, teammate_player),
        *_four_relative_to_ego(ego_player, opposing_team_other_slot1_player),
        *_four_relative_to_ego(ego_player, opposing_team_other_slot2_player),
        s0,
        s1,
    )
    return np.asarray(head + tail, dtype=np.float64)
