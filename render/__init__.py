"""
Pygame layer: markings, goals, and window I/O. Simulation logic stays in ``sim``.
"""

from render.pygame_shell import poll_pygame_tick_actions, render_game
from render.scene import draw_pitch

__all__ = ["draw_pitch", "poll_pygame_tick_actions", "render_game"]
