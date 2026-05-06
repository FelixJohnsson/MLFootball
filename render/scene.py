"""Assemble pitch layers onto a FIXED_SIZE surface matching `FIELD_W` × `FIELD_H`."""

from __future__ import annotations

import pygame

from field import FIELD_H, FIELD_W
from .goals import draw_goal, draw_goal_line_markers
from .markings import (
    draw_boxes,
    draw_center_circle,
    draw_grass,
    draw_penalty_ds,
    draw_spots,
    draw_touchlines,
)


def draw_pitch(surface: pygame.Surface) -> None:
    w, h = surface.get_size()
    if (w, h) != (FIELD_W, FIELD_H):
        raise ValueError(f"pitch surface must be {FIELD_W}×{FIELD_H}, got {w}×{h}")

    draw_grass(surface)
    draw_goal(surface, left=True)
    draw_goal(surface, left=False)
    draw_touchlines(surface)
    draw_center_circle(surface)
    draw_boxes(surface)
    draw_penalty_ds(surface)
    draw_spots(surface)
    draw_goal_line_markers(surface)
