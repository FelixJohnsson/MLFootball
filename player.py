"""Controllable player — force-based locomotion & contact kicks.

Project team rule — **LEFT** team (typically ``ATTACK_RIGHT_GOAL_*`` spawn geometry) shoots into the **+x (right)** net;
**RIGHT** team (``ATTACK_LEFT_GOAL_*``) shoots into the **−x (left)** net. Spawn blocks are named by *mouth direction*, not side-of-pitch nickname.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from physics import BALL_PLAYER_TOUCH_EXTRA, Ball
from field import FIELD_H, FIELD_W, pitch_wall_collision_edges
import random

def _closest_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float, float]:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-18:
        dx, dy = px - ax, py - ay
        return ax, ay, dx * dx + dy * dy
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    sx, sy = ax + t * abx, ay + t * aby
    dx, dy = px - sx, py - sy
    return sx, sy, dx * dx + dy * dy



# -----------------------------------------------------------------------------
# Movement
# -----------------------------------------------------------------------------
PLAYER_ACCEL = 2000.0  # px / s²
PLAYER_LINEAR_DRAG = 7.8  # 1/s
PLAYER_MAX_SPEED = 220.0  # px / s
PLAYER_NEAR_STOP_SPEED = 2.5  # post-drag cutoff; velocities zeroed below this

# -----------------------------------------------------------------------------
# Heuristic chase for frozen team (when no policy): scale × PLAYER_MAX_SPEED
# -----------------------------------------------------------------------------
HEURISTIC_FROZEN_TEAM_CHASE_SCALE = 0

# -----------------------------------------------------------------------------
# Roster shooting into the +x goal (often drawn as the ``left'' team — starts −x half).
# -----------------------------------------------------------------------------
ATTACK_RIGHT_GOAL_TEAM1_SPAWN_X_FRACS = (0.10, 0.40)
ATTACK_RIGHT_GOAL_TEAM2_SPAWN_X_FRACS = (0.10, 0.40)
ATTACK_RIGHT_GOAL_SPAWN_Y_JITTER = (-400.0, 400.0)
ATTACK_RIGHT_GOAL_PAIR_CENTER_DIST_PAD = 4.0
ATTACK_RIGHT_GOAL_PAIR_PLACE_TRIES = 80
ATTACK_RIGHT_GOAL_FALLBACK_X_FRAC = 200  # fallback: spread horizontally along this band

# -----------------------------------------------------------------------------
# Roster shooting into the −x goal (often drawn as the ``right'' team — starts +x half).
# -----------------------------------------------------------------------------
ATTACK_LEFT_GOAL_TEAM1_SPAWN_X_FRACS = (0.50, 0.75)
ATTACK_LEFT_GOAL_TEAM2_SPAWN_X_FRACS = (0.50, 0.75)
ATTACK_LEFT_GOAL_SPAWN_Y_JITTER = (-200.0, 200.0)
ATTACK_LEFT_GOAL_PAIR_CENTER_DIST_PAD = 4.0
ATTACK_LEFT_GOAL_PAIR_PLACE_TRIES = 80
ATTACK_LEFT_GOAL_FALLBACK_X_FRAC = 0.51  # fallback: spread horizontally along this band

# -----------------------------------------------------------------------------
# Spawn: opponent (fractions of FIELD_W; Y offset from midfield)
# -----------------------------------------------------------------------------

OPPONENT_RESET_X_FRACS = (0.40, 0.41)
OPPONENT_RESET_Y_JITTER = (0,1)

OPPONENT_FACTORY_SEED_X_FRAC = 0.40  # replaced immediately by reset_opponent_to_start()
OPPONENT_FACTORY_Y_JITTER = (0.0, 1.0)


@dataclass
class Player:
    x: float
    y: float
    radius: float = 16.0
    vx: float = 0.0
    vy: float = 0.0
    #: Exclusive-end sim tick: draw kick ring while ``tick_count < kick_flash_expire_at`` (see ``sim.constants.KICK_FLASH_TICKS``).
    kick_flash_expire_at: int = 0

    def _clamp_to_pitch(self) -> None:
        rp = self.radius
        rp_sq = (rp - 1e-6) ** 2
        edges = pitch_wall_collision_edges()
        for _ in range(2):
            for ax, ay, bx, by, *_ in edges:
                cx, cy, dist_sq = _closest_on_segment(self.x, self.y, ax, ay, bx, by)
                if dist_sq >= rp_sq or dist_sq < 1e-18:
                    continue
                dist = math.sqrt(dist_sq)
                nx = (self.x - cx) / dist
                ny = (self.y - cy) / dist
                self.x += nx * (rp - dist)
                self.y += ny * (rp - dist)

        eps = 0.001
        if self.x <= rp + eps:
            self.vx = max(0.0, self.vx)
        if self.x >= FIELD_W - rp - eps:
            self.vx = min(0.0, self.vx)
        if self.y <= rp + eps:
            self.vy = max(0.0, self.vy)
        if self.y >= FIELD_H - rp - eps:
            self.vy = min(0.0, self.vy)

    def set_pos(self, px: float, py: float) -> None:
        self.x = px
        self.y = py
        self.vx = self.vy = 0.0
        self._clamp_to_pitch()

    def apply_movement(self, move_x: float, move_y: float, dt: float) -> None:
        """
        Steering: (move_x, move_y) approximate a stick; clamped axes are typical
        in [-1, 1]. Direction is unit-normalized; stick length scales thrust (cap 1).
        """
        dx = float(move_x)
        dy = float(move_y)
        mag = math.hypot(dx, dy)
        if mag > 1e-6:
            thrust = min(1.0, mag) * PLAYER_ACCEL * dt
            self.vx += (dx / mag) * thrust
            self.vy += (dy / mag) * thrust

        kd = math.exp(-PLAYER_LINEAR_DRAG * dt)
        self.vx *= kd
        self.vy *= kd

        spd = math.hypot(self.vx, self.vy)
        if spd > PLAYER_MAX_SPEED and spd > 0:
            s = PLAYER_MAX_SPEED / spd
            self.vx *= s
            self.vy *= s

        if spd < PLAYER_NEAR_STOP_SPEED:
            self.vx = 0.0
            self.vy = 0.0

        self.x += self.vx * dt
        self.y += self.vy * dt
        self._clamp_to_pitch()

    def reset_to_start(self) -> None:
        """Solo spawn for a player attacking the +x net (paired resets use ``place_attack_right_goal_team_no_overlap``)."""
        self.x = FIELD_W * random.uniform(*ATTACK_RIGHT_GOAL_TEAM1_SPAWN_X_FRACS)
        self.y = (FIELD_H / 2) + random.uniform(*ATTACK_RIGHT_GOAL_SPAWN_Y_JITTER)
        self.vx = self.vy = 0.0
        self._clamp_to_pitch()

    def reset_opponent_to_start(self) -> None:
        """Spawn near the left goal (mirror of RL agent kicking toward right)."""
        self.x = FIELD_W * random.uniform(*OPPONENT_RESET_X_FRACS)
        self.y = (FIELD_H / 2) + random.uniform(*OPPONENT_RESET_Y_JITTER)
        self.vx = self.vy = 0.0
        self._clamp_to_pitch()

    def apply_chase_toward_ball(self, ball: Ball, speed_scale: float, dt: float) -> None:
        """Move toward ``ball`` at ``speed_scale * PLAYER_MAX_SPEED`` (simple pursuit)."""
        dx = ball.x - self.x
        dy = ball.y - self.y
        mag = math.hypot(dx, dy)
        if mag < 1e-9:
            self.vx = self.vy = 0.0
            return
        spd = float(speed_scale) * PLAYER_MAX_SPEED
        ux, uy = dx / mag, dy / mag
        self.vx = ux * spd
        self.vy = uy * spd
        self.x += self.vx * dt
        self.y += self.vy * dt
        self._clamp_to_pitch()

    def distance_to_ball(self, ball: Ball) -> float:
        return math.hypot(ball.x - self.x, ball.y - self.y)

    def in_contact_with(self, ball: Ball) -> bool:
        sep = self.radius + ball.r + BALL_PLAYER_TOUCH_EXTRA
        return self.distance_to_ball(ball) <= sep

    def try_kick(self, ball: Ball, impulse: float = 560.0) -> bool:
        if not self.in_contact_with(ball):
            return False
        ball.kick_radial_from(self.x, self.y, impulse=impulse)
        return True


def attack_right_goal_pair_min_center_distance(d1: Player, d2: Player) -> float:
    return float(d1.radius + d2.radius + ATTACK_RIGHT_GOAL_PAIR_CENTER_DIST_PAD)


def attack_left_goal_pair_min_center_distance(a1: Player, a2: Player) -> float:
    return float(a1.radius + a2.radius + ATTACK_LEFT_GOAL_PAIR_CENTER_DIST_PAD)


def place_attack_left_goal_team_no_overlap(a1: Player, a2: Player) -> None:
    """Pair shooting into −x goal: sample until separated; fallback horizontal."""
    mid = FIELD_H / 2.0
    min_d = attack_left_goal_pair_min_center_distance(a1, a2)

    for _ in range(ATTACK_LEFT_GOAL_PAIR_PLACE_TRIES):
        a1.x = FIELD_W * random.uniform(*ATTACK_LEFT_GOAL_TEAM1_SPAWN_X_FRACS)
        a1.y = mid + random.uniform(*ATTACK_LEFT_GOAL_SPAWN_Y_JITTER)
        a2.x = FIELD_W * random.uniform(*ATTACK_LEFT_GOAL_TEAM2_SPAWN_X_FRACS)
        a2.y = mid + random.uniform(*ATTACK_LEFT_GOAL_SPAWN_Y_JITTER)
        a1.vx = a1.vy = a2.vx = a2.vy = 0.0
        a1._clamp_to_pitch()
        a2._clamp_to_pitch()
        if math.hypot(a2.x - a1.x, a2.y - a1.y) >= min_d:
            return

    r = float(a1.radius)
    margin = 8.0
    x_anchor = FIELD_W * ATTACK_LEFT_GOAL_FALLBACK_X_FRAC
    x_left = max(r + margin, x_anchor - min_d * 0.5)
    x_right = min(x_left + min_d, FIELD_W - r - margin)
    if x_right - x_left < min_d:
        x_right = min(FIELD_W * 0.58, FIELD_W - r - margin)
        x_left = max(r + margin, x_right - min_d)
    a1.set_pos(x_left, mid + random.uniform(*ATTACK_LEFT_GOAL_SPAWN_Y_JITTER))
    a2.set_pos(x_right, mid + random.uniform(*ATTACK_LEFT_GOAL_SPAWN_Y_JITTER))


def place_attack_right_goal_team_no_overlap(d1: Player, d2: Player) -> None:
    """Pair shooting into +x goal: sample until separated; fallback toward +x."""
    mid = FIELD_H / 2.0
    min_d = attack_right_goal_pair_min_center_distance(d1, d2)

    for _ in range(ATTACK_RIGHT_GOAL_PAIR_PLACE_TRIES):
        d1.x = FIELD_W * random.uniform(*ATTACK_RIGHT_GOAL_TEAM1_SPAWN_X_FRACS)
        d1.y = mid + random.uniform(*ATTACK_RIGHT_GOAL_SPAWN_Y_JITTER)
        d2.x = FIELD_W * random.uniform(*ATTACK_RIGHT_GOAL_TEAM2_SPAWN_X_FRACS)
        d2.y = mid + random.uniform(*ATTACK_RIGHT_GOAL_SPAWN_Y_JITTER)
        d1.vx = d1.vy = d2.vx = d2.vy = 0.0
        d1._clamp_to_pitch()
        d2._clamp_to_pitch()
        if math.hypot(d2.x - d1.x, d2.y - d1.y) >= min_d:
            return

    r = float(d1.radius)
    margin = 8.0
    x_right = min(FIELD_W * ATTACK_RIGHT_GOAL_FALLBACK_X_FRAC, FIELD_W - r - margin)
    x_left = x_right - min_d
    if x_left <= r + margin:
        x_left = r + margin
        x_right = min(x_left + min_d, FIELD_W - r - margin)
        if x_right - x_left < min_d:
            x_right = FIELD_W - r - margin
            x_left = max(r + margin, x_right - min_d)
    d1.set_pos(x_right, mid)
    d2.set_pos(x_left, mid)


def create_attack_left_goal_pair() -> tuple[Player, Player]:
    """Two players attacking the −x net; non-overlapping."""
    a1 = Player(x=0.0, y=0.0)
    a2 = Player(x=0.0, y=0.0)
    place_attack_left_goal_team_no_overlap(a1, a2)
    return a1, a2


def create_opponent_player() -> Player:
    """Human-controlled pawn for watch mode; appears near defending (left) goal."""
    p = Player(
        x=FIELD_W * OPPONENT_FACTORY_SEED_X_FRAC,
        y=(FIELD_H / 2) + random.uniform(*OPPONENT_FACTORY_Y_JITTER),
    )
    p.reset_opponent_to_start()
    return p


def create_attack_right_goal_pair() -> tuple[Player, Player]:
    """Two players attacking the +x net; non-overlapping."""
    d1 = Player(x=0.0, y=0.0)
    d2 = Player(x=0.0, y=0.0)
    place_attack_right_goal_team_no_overlap(d1, d2)
    return d1, d2

