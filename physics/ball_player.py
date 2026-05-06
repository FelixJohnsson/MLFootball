"""Ball–player capsule: separation + contact-only impulses (player is duck-typed)."""

from __future__ import annotations

import math

from .ball import Ball

BALL_PHYSICAL_MASS = 1.0
PLAYER_PHYSICAL_MASS = 78.0
BALL_PLAYER_RESTITUTION = 0.48
BALL_PLAYER_TOUCH_EXTRA = 0.5


def resolve_ball_player(ball: Ball, player) -> bool:
    mb = BALL_PHYSICAL_MASS
    mp = PLAYER_PHYSICAL_MASS
    pr = float(player.radius)
    min_dist = ball.r + pr
    touch_limit = min_dist + BALL_PLAYER_TOUCH_EXTRA
    touch_limit_sq = touch_limit * touch_limit

    px, py = float(player.x), float(player.y)
    pvx, pvy = float(player.vx), float(player.vy)

    dx = ball.x - px
    dy = ball.y - py
    dist_sq = dx * dx + dy * dy

    if dist_sq > touch_limit_sq:
        return False

    if dist_sq < 1e-12:
        dx, dy = 1.0, 0.0
        dist = 1.0
    else:
        dist = math.sqrt(dist_sq)

    nx = dx / dist
    ny = dy / dist

    if dist < min_dist:
        penetration = min_dist - dist
        inv = 1.0 / (mb + mp)
        corr_b = penetration * mp * inv
        corr_p = penetration * mb * inv
        ball.x += nx * corr_b
        ball.y += ny * corr_b
        player.x = px - nx * corr_p
        player.y = py - ny * corr_p

    dx = ball.x - player.x
    dy = ball.y - player.y
    dist_sq = dx * dx + dy * dy
    if dist_sq > touch_limit_sq or dist_sq < 1e-12:
        player._clamp_to_pitch()
        return True

    dist_touch = math.sqrt(dist_sq)
    nx = dx / dist_touch
    ny = dy / dist_touch

    bxv = ball.vx - pvx
    byv = ball.vy - pvy
    vn_rel = bxv * nx + byv * ny
    if vn_rel < 0.0:
        inv_sum = 1.0 / mb + 1.0 / mp
        impulse = -(1.0 + BALL_PLAYER_RESTITUTION) * vn_rel / inv_sum
        ball.vx += (impulse / mb) * nx
        ball.vy += (impulse / mb) * ny
        player.vx -= (impulse / mp) * nx
        player.vy -= (impulse / mp) * ny

    player._clamp_to_pitch()
    return True
