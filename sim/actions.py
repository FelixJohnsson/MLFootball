"""Per-tick input for GameState.step — no UI library."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TickActions:
    """
    Continuous move on each axis ∈ [-1, 1]; direction is normalized, magnitude
    scales acceleration (see ``Player.apply_movement``).
    kick_power ∈ [0, 1]: attempt contact kick scaled by ``KICK_IMPULSE * kick_power``.
    """

    move_x: float
    move_y: float
    reset_ball_only: bool
    kick_power: float


def idle_actions() -> TickActions:
    return TickActions(0.0, 0.0, False, 0.0)
