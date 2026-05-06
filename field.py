"""Pitch geometry & constants — no Pygame (sim + render share this).

Team rule reminder: −x mouth = “left net” tally index 0; +x mouth = “right net” tally index 1. LEFT-team players attack +x (``attack_target_goal_xy``).
"""

from __future__ import annotations

import math

# Playing surface (~105 × 68 m proportions)
FIELD_W = 1200
FIELD_H = 800
FIELD_DIAG = math.hypot(FIELD_W, FIELD_H)

LENGTH_M = 105.0
WIDTH_M = 68.0

GOAL_DEPTH_PX = 34
CENTER_CIRCLE_R_M = 9.15
PENALTY_ARC_R_M = 9.15
CORNER_ARC_R_M = 1.0
CORNER_CHAMFER_M = 3.2
GOAL_OPENING_M = 7.32
LINE_W = 3


def xm(m: float) -> float:
    return m / LENGTH_M * FIELD_W


def ym(m: float) -> float:
    return m / WIDTH_M * FIELD_H


def rx_rad(r_m: float) -> float:
    return r_m / LENGTH_M * FIELD_W


def ry_rad(r_m: float) -> float:
    return r_m / WIDTH_M * FIELD_H


def _compute_goal_opening_y_range() -> tuple[float, float]:
    """One-time goal mouth Y bounds (same math as public API)."""
    cy_m = WIDTH_M / 2
    half = GOAL_OPENING_M / 2
    y0 = ym(cy_m - half)
    y1 = ym(cy_m + half)
    return (min(y0, y1), max(y0, y1))


CACHED_GOAL_OPENING_Y: tuple[float, float] = _compute_goal_opening_y_range()


def goal_opening_y_range() -> tuple[float, float]:
    """World-space Y extent of goal mouth opening (pixels, Y-down). Cached — constants only."""
    return CACHED_GOAL_OPENING_Y


def attack_target_goal_xy() -> tuple[float, float]:
    """Mouth centre of the +x (right-line) goal — ``ATTACK_RIGHT_GOAL_*`` roster shoots here."""
    gt, gb = goal_opening_y_range()
    return float(FIELD_W), (gt + gb) * 0.5


def left_attack_target_goal_xy() -> tuple[float, float]:
    """Mouth centre of the −x (left-line) goal — ``ATTACK_LEFT_GOAL_*`` roster shoots here."""
    gt, gb = goal_opening_y_range()
    return 0.0, (gt + gb) * 0.5


def corner_chamfer_px() -> tuple[float, float]:
    """Chamfer size in pixels along ±length and ±width from each corner vertex."""
    return xm(CORNER_CHAMFER_M), ym(CORNER_CHAMFER_M)


def touchline_octagon_vertices() -> tuple[tuple[float, float], ...]:
    """Touchline as an octagon: vertices clockwise starting at inner end of top edge."""
    cx, cy = corner_chamfer_px()
    w, h = float(FIELD_W), float(FIELD_H)
    return (
        (cx, 0.0),
        (w - cx, 0.0),
        (w, cy),
        (w, h - cy),
        (w - cx, h),
        (cx, h),
        (0.0, h - cy),
        (0.0, cy),
    )


def _segment_inward_normal(ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    ex, ey = bx - ax, by - ay
    el = math.hypot(ex, ey)
    if el < 1e-12:
        return 0.0, 1.0
    nx, ny = -ey / el, ex / el
    mx, my = 0.5 * FIELD_W, 0.5 * FIELD_H
    tcx, tcy = 0.5 * (ax + bx), 0.5 * (ay + by)
    if nx * (mx - tcx) + ny * (my - tcy) < 0:
        nx, ny = -nx, -ny
    return nx, ny


def _build_pitch_wall_collision_edges() -> tuple[tuple[float, float, float, float, float, float], ...]:
    """
    Touchline segments for rigid collision; goal mouth gaps on x=0 and x=FIELD_W.
    Each tuple: ax, ay, bx, by, inward_nx, inward_ny.
    """
    gt, gb = CACHED_GOAL_OPENING_Y
    cx, cy = corner_chamfer_px()
    w, h = float(FIELD_W), float(FIELD_H)

    raw: list[tuple[tuple[float, float], tuple[float, float]]] = [
        ((cx, 0.0), (w - cx, 0.0)),
        ((w - cx, 0.0), (w, cy)),
    ]
    if gt > cy:
        raw.append(((w, cy), (w, gt)))
    if gb < h - cy:
        raw.append(((w, gb), (w, h - cy)))
    raw.extend(
        (
            ((w, h - cy), (w - cx, h)),
            ((w - cx, h), (cx, h)),
            ((cx, h), (0.0, h - cy)),
        )
    )
    if gb < h - cy:
        raw.append(((0.0, h - cy), (0.0, gb)))
    if gt > cy:
        raw.append(((0.0, gt), (0.0, cy)))
    raw.append(((0.0, cy), (cx, 0.0)))

    out: list[tuple[float, float, float, float, float, float]] = []
    for (ax, ay), (bx, by) in raw:
        nx, ny = _segment_inward_normal(ax, ay, bx, by)
        out.append((ax, ay, bx, by, nx, ny))
    return tuple(out)


CACHED_PITCH_WALL_EDGES: tuple[
    tuple[float, float, float, float, float, float], ...
] = _build_pitch_wall_collision_edges()


def pitch_wall_collision_edges() -> tuple[tuple[float, float, float, float, float, float], ...]:
    """Touchline segment list for rigid collision. Built once — field geometry is static."""
    return CACHED_PITCH_WALL_EDGES


__all__ = [
    "FIELD_W",
    "FIELD_H",
    "FIELD_DIAG",
    "LENGTH_M",
    "WIDTH_M",
    "GOAL_DEPTH_PX",
    "GOAL_OPENING_M",
    "CENTER_CIRCLE_R_M",
    "PENALTY_ARC_R_M",
    "CORNER_ARC_R_M",
    "CORNER_CHAMFER_M",
    "corner_chamfer_px",
    "LINE_W",
    "xm",
    "ym",
    "rx_rad",
    "ry_rad",
    "goal_opening_y_range",
    "attack_target_goal_xy",
    "left_attack_target_goal_xy",
    "touchline_octagon_vertices",
    "pitch_wall_collision_edges",
]
