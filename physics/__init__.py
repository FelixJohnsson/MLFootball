"""Deterministic rigid-body primitives for pitch + actors (pure Python + field math)."""

from .ball import Ball, create_ball
from .ball_player import (
    BALL_PHYSICAL_MASS,
    BALL_PLAYER_RESTITUTION,
    BALL_PLAYER_TOUCH_EXTRA,
    PLAYER_PHYSICAL_MASS,
    resolve_ball_player,
)
from .player_player import resolve_player_player

__all__ = [
    "BALL_PHYSICAL_MASS",
    "BALL_PLAYER_RESTITUTION",
    "BALL_PLAYER_TOUCH_EXTRA",
    "Ball",
    "PLAYER_PHYSICAL_MASS",
    "create_ball",
    "resolve_ball_player",
    "resolve_player_player",
]
