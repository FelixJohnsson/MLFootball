"""Field lines, grass, arcs — metre grid from `field`, colours from `colors`."""

from __future__ import annotations

import math

import pygame

from field import (
    CENTER_CIRCLE_R_M,
    FIELD_H,
    FIELD_W,
    LENGTH_M,
    LINE_W,
    PENALTY_ARC_R_M,
    WIDTH_M,
    corner_chamfer_px,
    touchline_octagon_vertices,
    xm,
    ym,
    rx_rad,
    ry_rad,
)

from .colors import GRASS_A, GRASS_B, LINE


def draw_grass(surface: pygame.Surface) -> None:
    surface.fill(GRASS_A)
    stripes = 12
    band = FIELD_W / stripes
    for i in range(stripes):
        if i % 2:
            x0 = int(i * band)
            pygame.draw.rect(surface, GRASS_B, pygame.Rect(x0, 0, int(math.ceil(band)) + 1, FIELD_H))


def _fill_corner_cutouts(surface: pygame.Surface) -> None:
    """Regions outside the chamfered touchline — slightly darker grass."""
    cx_i = int(round(corner_chamfer_px()[0]))
    cy_i = int(round(corner_chamfer_px()[1]))
    w, h = FIELD_W - 1, FIELD_H - 1
    triangles = (
        [(0, 0), (cx_i, 0), (0, cy_i)],
        [(w, 0), (FIELD_W - cx_i, 0), (w, cy_i)],
        [(w, h), (FIELD_W - cx_i, h), (w, FIELD_H - cy_i)],
        [(0, h), (cx_i, h), (0, FIELD_H - cy_i)],
    )
    for tri in triangles:
        pygame.draw.polygon(surface, GRASS_B, tri)


def draw_touchlines(surface: pygame.Surface) -> None:
    _fill_corner_cutouts(surface)
    verts = touchline_octagon_vertices()
    pts = [(int(round(x)), int(round(y))) for x, y in verts]
    pygame.draw.lines(surface, LINE, True, pts, LINE_W)
    hx = int(round(xm(LENGTH_M / 2)))
    pygame.draw.line(surface, LINE, (hx, 0), (hx, FIELD_H), LINE_W)


def draw_center_circle(surface: pygame.Surface) -> None:
    cx, cy = xm(LENGTH_M / 2), ym(WIDTH_M / 2)
    rect = pygame.Rect(
        cx - rx_rad(CENTER_CIRCLE_R_M),
        cy - ry_rad(CENTER_CIRCLE_R_M),
        2 * rx_rad(CENTER_CIRCLE_R_M),
        2 * ry_rad(CENTER_CIRCLE_R_M),
    )
    pygame.draw.ellipse(surface, LINE, rect, LINE_W)


def draw_boxes(surface: pygame.Surface) -> None:
    pd, pw = 16.5, 40.32
    half = pw / 2
    cy_m = WIDTH_M / 2
    y_top, y_bot = cy_m - half, cy_m + half

    pygame.draw.rect(surface, LINE, pygame.Rect(0, ym(y_top), xm(pd), ym(y_bot) - ym(y_top)), LINE_W)
    pygame.draw.rect(
        surface,
        LINE,
        pygame.Rect(FIELD_W - xm(pd), ym(y_top), xm(pd), ym(y_bot) - ym(y_top)),
        LINE_W,
    )

    gd, gw = 5.5, 18.32
    hg = gw / 2
    gy0, gy1 = cy_m - hg, cy_m + hg
    pygame.draw.rect(surface, LINE, pygame.Rect(0, ym(gy0), xm(gd), ym(gy1) - ym(gy0)), LINE_W)
    pygame.draw.rect(
        surface,
        LINE,
        pygame.Rect(FIELD_W - xm(gd), ym(gy0), xm(gd), ym(gy1) - ym(gy0)),
        LINE_W,
    )


def _penalty_d_angles(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    x_penalty_edge: float,
    *,
    goal_left: bool,
) -> tuple[float, float]:
    pts: list[tuple[float, float]] = []
    steps = 2000
    for i in range(steps + 1):
        t = math.tau * i / steps
        xs = cx + rx * math.cos(t)
        if goal_left:
            if xs >= x_penalty_edge - 0.25:
                pts.append((t, xs))
        else:
            if xs <= x_penalty_edge + 0.25:
                pts.append((t, xs))

    if len(pts) < 2:
        return 0.0, math.pi / 8

    segs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = [pts[0]]
    for item in pts[1:]:
        a_prev = cur[-1][0]
        dt = abs(item[0] - a_prev)
        if dt < 0.12 or dt > math.tau - 0.12:
            cur.append(item)
        else:
            segs.append(cur)
            cur = [item]
    segs.append(cur)

    longest = max(segs, key=len)
    if len(longest) < 8:
        return 0.0, math.pi / 4

    ts = [x[0] for x in longest]
    t0, t1 = min(ts), max(ts)
    if t1 - t0 < 0.2:
        t0, t1 = longest[0][0], longest[-1][0]
    if t1 < t0:
        t1 += math.tau
    return t0, t1


def draw_penalty_ds(surface: pygame.Surface) -> None:
    cy = ym(WIDTH_M / 2)
    rxp, ryp = rx_rad(PENALTY_ARC_R_M), ry_rad(PENALTY_ARC_R_M)
    edge_l = xm(16.5)
    edge_r = xm(LENGTH_M - 16.5)

    cx_l = xm(11.0)
    rl = pygame.Rect(cx_l - rxp, cy - ryp, 2 * rxp, 2 * ryp)
    a0, a1 = _penalty_d_angles(cx_l, cy, rxp, ryp, edge_l, goal_left=True)
    pygame.draw.arc(surface, LINE, rl, a0, a1, LINE_W)

    cx_r = xm(LENGTH_M - 11.0)
    rr = pygame.Rect(cx_r - rxp, cy - ryp, 2 * rxp, 2 * ryp)
    b0, b1 = _penalty_d_angles(cx_r, cy, rxp, ryp, edge_r, goal_left=False)
    pygame.draw.arc(surface, LINE, rr, b0, b1, LINE_W)


def draw_spots(surface: pygame.Surface) -> None:
    cy = int(ym(WIDTH_M / 2))
    mid = int(xm(LENGTH_M / 2))
    r = 5
    pygame.draw.circle(surface, LINE, (mid, cy), r)
    pygame.draw.circle(surface, LINE, (int(xm(11.0)), cy), r)
    pygame.draw.circle(surface, LINE, (int(xm(LENGTH_M - 11.0)), cy), r)
