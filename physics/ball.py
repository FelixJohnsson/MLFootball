"""Ball kinematics, kicks, and collisions with the pitch (no player here)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from field import FIELD_H, FIELD_W, GOAL_DEPTH_PX, goal_opening_y_range, pitch_wall_collision_edges


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# After ``resolve_ball_player`` / iterative physics, snap ball vs pitch — tuning.
REAPPLY_PITCH_PER_PHYS_ITERATION = 1
REAPPLY_PITCH_AFTER_PHYS_LOOP = 2


def _closest_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-18:
        return ax, ay
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    return ax + t * abx, ay + t * aby


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    r: float = 9.0
    restitution: float = 0.78
    linear_drag: float = 1.35  # s^-1; higher = stops sooner on grass
    max_speed: float = 2600.0

    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def _apply_drag(self, dt: float) -> None:
        k = math.exp(-self.linear_drag * dt)
        self.vx *= k
        self.vy *= k
        if self.speed() < 8.0:
            self.vx = 0.0
            self.vy = 0.0

    def _cap_speed(self) -> None:
        sp = self.speed()
        if sp > self.max_speed and sp > 0:
            s = self.max_speed / sp
            self.vx *= s
            self.vy *= s

    def integrate(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

    def kick_radial_from(self, ox: float, oy: float, impulse: float = 560.0) -> None:
        dx, dy = self.x - ox, self.y - oy
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return
        self.vx += impulse * dx / d
        self.vy += impulse * dy / d
        self._cap_speed()

    def reset_center(self) -> None:
        self.x = FIELD_W / 2
        self.y = FIELD_H / 2
        self.vx = self.vy = 0.0

    def which_goal_scored(self) -> str | None:
        """``left`` tag = −x-line mouth scored; ``right`` tag = +x-line mouth (+x tally is LEFT team's goal)."""
        gt, gb = goal_opening_y_range()
        if not (gt < self.y < gb):
            return None
        w = float(FIELD_W)
        r = float(self.r)
        # Whole ball has crossed opening plane x=0 (from the right) / x=FIELD_W (from the left).
        if self.x + r < 0.0:
            return "left"
        if self.x - r > w:
            return "right"
        return None

    def crossed_goal_plane(self) -> bool:
        return self.which_goal_scored() is not None

    def _bounce_x(self, wall_x: float) -> None:
        self.x = wall_x
        self.vx = -self.vx * self.restitution

    def _bounce_y(self, wall_y: float) -> None:
        self.y = wall_y
        self.vy = -self.vy * self.restitution

    def _resolve_wall_segment(
        self,
        ax: float,
        ay: float,
        bx: float,
        by: float,
        nin_x: float,
        nin_y: float,
    ) -> None:
        """
        Keep the ball at least ``r`` inside the touchline half-plane (along ``nin``).
        Works when the centre is **outside** the field: Euclidean distance to the
        segment can exceed ``r``, so the old ``dist < r`` test never fired.
        """
        cx, cy = _closest_on_segment(self.x, self.y, ax, ay, bx, by)
        r = self.r
        vx = self.x - cx
        vy = self.y - cy
        s_proj = vx * nin_x + vy * nin_y
        eps = 1e-5
        if s_proj >= r - eps:
            return
        need = r - s_proj
        self.x += need * nin_x
        self.y += need * nin_y
        vn = self.vx * nin_x + self.vy * nin_y
        if vn < 0.0:
            self.vx -= 2.0 * vn * nin_x
            self.vy -= 2.0 * vn * nin_y
            self.vx *= self.restitution
            self.vy *= self.restitution

    def resolve_collisions(self) -> None:
        r = self.r
        gt, gb = goal_opening_y_range()
        depth = float(GOAL_DEPTH_PX)
        back_l = -depth + r
        back_r = FIELD_W + depth - r
        mouth = gt < self.y < gb

        if self.x < 0.0 and mouth:
            if self.x < back_l:
                self._bounce_x(back_l)
            if self.y < gt + r:
                self._bounce_y(gt + r)
            elif self.y > gb - r:
                self._bounce_y(gb - r)

        if self.x > FIELD_W and mouth:
            if self.x > back_r:
                self._bounce_x(back_r)
            if self.y < gt + r:
                self._bounce_y(gt + r)
            elif self.y > gb - r:
                self._bounce_y(gb - r)

        # In the mouth, once the ball's silhouette crosses the vertical goal plane at
        # FIELD_W (or 0), the chamfer/pillar segments at that x sit *inside* or behind
        # the sphere; resolving them snaps the ball back onto the pitch before the centre
        # can cross — no goal even with x + r > FIELD_W skipped by old rule. Skip outfield
        # edges whenever mouth is true AND any part of the ball has crossed those planes.
        w = float(FIELD_W)
        skip_outfield_edges = mouth and (
            self.x > w
            or (self.x + r) > w
            or self.x < 0.0
            or (self.x - r) < 0.0
        )

        if not skip_outfield_edges:
            for ax, ay, bx, by, nin_x, nin_y in pitch_wall_collision_edges():
                self._resolve_wall_segment(ax, ay, bx, by, nin_x, nin_y)

        post_r = 4.25
        for gx in (0.0, float(FIELD_W)):
            for gy in (gt, gb):
                self._resolve_circle(gx, gy, post_r)

        self.y = _clamp(self.y, r, FIELD_H - r)
        self.x = _clamp(self.x, back_l - r, back_r + r)

    def reapply_pitch_collisions(
        self, passes: int = REAPPLY_PITCH_AFTER_PHYS_LOOP
    ) -> None:
        """
        Resolve ball vs pitch (and posts) without integration. Call after external
        position changes — e.g. ``resolve_ball_player`` — which can push the ball
        past touchlines before the next ``update``.
        """
        for _ in range(passes):
            self.resolve_collisions()

    def _resolve_circle(self, cx: float, cy: float, rad: float) -> None:
        dx = self.x - cx
        dy = self.y - cy
        dist = math.hypot(dx, dy)
        min_dist = self.r + rad
        if dist < 1e-6 or dist >= min_dist:
            return
        nx, ny = dx / dist, dy / dist
        overlap = min_dist - dist
        self.x += nx * overlap
        self.y += ny * overlap
        vn = self.vx * nx + self.vy * ny
        if vn < 0:
            self.vx -= 2 * vn * nx
            self.vy -= 2 * vn * ny
            self.vx *= self.restitution
            self.vy *= self.restitution

    def update(self, dt: float) -> None:
        self._apply_drag(dt)
        self.integrate(dt)
        for _ in range(2):
            self.resolve_collisions()
        self._cap_speed()


def create_ball() -> Ball:
    return Ball(x=FIELD_W / 2, y=FIELD_H / 2, r=9.0)
