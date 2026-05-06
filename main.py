"""MLFootball — thin Pygame harness: poll → simulate → render."""

from __future__ import annotations

import pygame

from field import FIELD_H, FIELD_W
from render import poll_pygame_tick_actions, render_game
from sim.game import GameState


def run_windowed() -> None:
    pygame.init()

    screen = pygame.display.set_mode((FIELD_W, FIELD_H))
    pygame.display.set_caption("MLFootball")

    clock = pygame.time.Clock()
    state = GameState.new()

    running = True
    while running:
        dt_ms = clock.tick(60)
        dt = min(dt_ms / 1000.0, 1.0 / 30.0)

        events = pygame.event.get()
        keys = pygame.key.get_pressed()
        actions, quit_req = poll_pygame_tick_actions(events, keys)
        if quit_req:
            running = False

        state.step(actions, dt)

        render_game(screen, state)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_windowed()
