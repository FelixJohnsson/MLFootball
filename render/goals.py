"""Goals, netting, mouth markers (see `Ball.which_goal_scored` + touchline openings)."""

from __future__ import annotations

import pygame

from field import FIELD_W, GOAL_DEPTH_PX, GOAL_OPENING_M, LINE_W, WIDTH_M, ym
from .colors import (
    GOAL_INTERIOR,
    GOAL_LINE_TEAM1,
    GOAL_LINE_TEAM1_EDGE,
    GOAL_LINE_TEAM2,
    GOAL_LINE_TEAM2_EDGE,
    GOAL_LINE_W,
    GOAL_NET,
    GOAL_POST,
    LINE,
)


def _goal_opening_vertical_px() -> tuple[int, int]:
    cy_m = WIDTH_M / 2
    half = GOAL_OPENING_M / 2
    y0 = int(round(ym(cy_m - half)))
    y1 = int(round(ym(cy_m + half)))
    return min(y0, y1), max(y0, y1)


def draw_goal_line_markers(surface: pygame.Surface) -> None:
    top, bottom = _goal_opening_vertical_px()

    # Align with opening geometry in `draw_goal`: posts/line sit on x≈0 and x≈FIELD_W-1.
    lx = 0
    rx = FIELD_W - 1

    pygame.draw.line(surface, GOAL_LINE_TEAM2_EDGE, (lx, top), (lx, bottom), GOAL_LINE_W + 4)
    pygame.draw.line(surface, GOAL_LINE_TEAM2, (lx, top), (lx, bottom), GOAL_LINE_W)

    pygame.draw.line(surface, GOAL_LINE_TEAM1_EDGE, (rx, top), (rx, bottom), GOAL_LINE_W + 4)
    pygame.draw.line(surface, GOAL_LINE_TEAM1, (rx, top), (rx, bottom), GOAL_LINE_W)


def draw_goal(surface: pygame.Surface, *, left: bool) -> None:
    cy = WIDTH_M / 2
    half = GOAL_OPENING_M / 2
    y0 = int(ym(cy - half))
    y1 = int(ym(cy + half))
    top, bottom = min(y0, y1), max(y0, y1)
    depth = GOAL_DEPTH_PX
    post = max(5, LINE_W + 2)

    if left:
        box = pygame.Rect(-depth, top, depth, bottom - top)
        pygame.draw.rect(surface, GOAL_INTERIOR, box)
        step = max(8, depth // 4)
        for i in range(0, depth, step):
            x = -depth + i
            pygame.draw.line(surface, GOAL_NET, (x, top + 6), (x, bottom - 6), 1)
        for y in range(top + 10, bottom, 12):
            pygame.draw.line(surface, GOAL_NET, (-depth + 4, y), (-2, y), 1)
        pygame.draw.rect(surface, GOAL_POST, pygame.Rect(-post, top - post // 2, post, post))
        pygame.draw.rect(surface, GOAL_POST, pygame.Rect(-post, bottom - post // 2, post, post))
        pygame.draw.line(surface, GOAL_POST, (0, top), (0, bottom), post)
        pygame.draw.line(surface, (18, 44, 22), (-1, top), (-1, bottom), post + 2)
    else:
        box = pygame.Rect(FIELD_W, top, depth, bottom - top)
        pygame.draw.rect(surface, GOAL_INTERIOR, box)
        step = max(8, depth // 4)
        for i in range(0, depth, step):
            x = FIELD_W + i
            pygame.draw.line(surface, GOAL_NET, (x, top + 6), (x, bottom - 6), 1)
        for y in range(top + 10, bottom, 12):
            pygame.draw.line(surface, GOAL_NET, (FIELD_W + depth - 4, y), (FIELD_W + 2, y), 1)
        pygame.draw.rect(surface, GOAL_POST, pygame.Rect(FIELD_W - 1, top - post // 2, post, post))
        pygame.draw.rect(surface, GOAL_POST, pygame.Rect(FIELD_W - 1, bottom - post // 2, post, post))
        pygame.draw.line(surface, GOAL_POST, (FIELD_W - 1, top), (FIELD_W - 1, bottom), post)
        pygame.draw.line(surface, (18, 44, 22), (FIELD_W, bottom), (FIELD_W, top), post + 2)
