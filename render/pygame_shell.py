"""Pygame: OS input → TickActions, draw frame on the field-sized surface."""

from __future__ import annotations

import pygame

from field import FIELD_W
from physics import Ball
from player import Player
from .scene import draw_pitch

from sim.actions import TickActions
from sim.game import GameState

_HUD_FONT: pygame.font.Font | None = None
_ROSTER_LABEL_FONT: pygame.font.Font | None = None

_KICK_FLASH_RING_RGBA = (255, 226, 60)
_KICK_FLASH_RING_PAD_PX = 5
_KICK_FLASH_RING_WIDTH_PX = 3


def _roster_label_font() -> pygame.font.Font:
    global _ROSTER_LABEL_FONT
    if _ROSTER_LABEL_FONT is None:
        pygame.font.init()
        _ROSTER_LABEL_FONT = pygame.font.Font(None, 22)
    return _ROSTER_LABEL_FONT


def _hud_font() -> pygame.font.Font:
    global _HUD_FONT
    if _HUD_FONT is None:
        pygame.font.init()
        _HUD_FONT = pygame.font.Font(None, 46)
    return _HUD_FONT


def poll_pygame_tick_actions(events: list[pygame.event.Event], keys_pressed) -> tuple[TickActions, bool]:
    quit_req = False
    kick = False
    reset_ball_only = False

    for ev in events:
        if ev.type == pygame.QUIT:
            quit_req = True
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
            reset_ball_only = True
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            kick = True

    move_x = int(keys_pressed[pygame.K_d]) - int(keys_pressed[pygame.K_a])
    move_y = int(keys_pressed[pygame.K_s]) - int(keys_pressed[pygame.K_w])

    kick_power = 1.0 if kick else 0.0
    return TickActions(float(move_x), float(move_y), reset_ball_only, kick_power), quit_req


def draw_ball(surface: pygame.Surface, ball: Ball) -> None:
    ix, iy = int(round(ball.x)), int(round(ball.y))
    ir = int(round(ball.r))
    shadow = (15, 25, 18, 110)
    s = pygame.Surface((ir * 2 + 6, ir * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(s, shadow, (ir + 3, ir + 5), ir)
    surface.blit(s, (ix - ir - 3, iy - ir - 3))
    pygame.draw.circle(surface, (250, 250, 252), (ix, iy), ir)
    pygame.draw.circle(surface, (30, 32, 38), (ix, iy), ir, 2)
    p0 = (ix - int(ir * 0.45), iy - int(ir * 0.15))
    p1 = (ix + int(ir * 0.5), iy - int(ir * 0.35))
    pygame.draw.line(surface, (190, 192, 198), p0, p1, 2)


def draw_player(
    surface: pygame.Surface,
    player: Player,
    *,
    jersey: tuple[int, int, int] = (42, 98, 180),
    trim: tuple[int, int, int] = (230, 210, 80),
) -> None:
    ix, iy = int(round(player.x)), int(round(player.y))
    ir = int(round(player.radius))
    shadow = (12, 22, 14, 100)
    pad = 4
    s = pygame.Surface((ir * 2 + pad * 2, ir * 2 + pad * 2), pygame.SRCALPHA)
    pygame.draw.circle(s, shadow, (ir + pad, ir + pad + 2), ir)
    surface.blit(s, (ix - ir - pad, iy - ir - pad))
    pygame.draw.circle(surface, jersey, (ix, iy), ir)
    pygame.draw.circle(surface, trim, (ix, iy), ir, 3)
    pygame.draw.circle(surface, (28, 32, 38), (ix, iy), ir, 2)


def _draw_kick_flash_ring_if_active(surface: pygame.Surface, gs: GameState, player: Player) -> None:
    if gs.tick_count >= player.kick_flash_expire_at:
        return
    ix, iy = int(round(player.x)), int(round(player.y))
    rr = max(3, int(round(player.radius)) + _KICK_FLASH_RING_PAD_PX)
    pygame.draw.circle(surface, _KICK_FLASH_RING_RGBA, (ix, iy), rr, width=_KICK_FLASH_RING_WIDTH_PX)


def _draw_team_roster_label(surface: pygame.Surface, gs: GameState, player: Player) -> None:
    """L1/L2 on ``left_team_player*``, R1/R2 on ``right_team_player*``."""
    if player is gs.left_team_player1:
        text = "L1"
    elif player is gs.left_team_player2:
        text = "L2"
    elif player is gs.right_team_player1:
        text = "R1"
    elif player is gs.right_team_player2:
        text = "R2"
    else:
        return
    ix, iy = int(round(player.x)), int(round(player.y))
    font = _roster_label_font()
    fg = (255, 255, 250)
    shadow = (18, 22, 28)
    main = font.render(text, True, fg)
    shade = font.render(text, True, shadow)
    rect = main.get_rect(center=(ix, iy))
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        surface.blit(shade, rect.move(ox, oy))
    surface.blit(main, rect)


def draw_score_hud(surface: pygame.Surface, gs: GameState) -> None:
    # gs.score == (−x mouth count, +x mouth count). Team 1 label shows +x goals (LEFT_TEAM default).
    left_net_goals, right_net_goals = gs.score
    score_team1 = right_net_goals
    score_team2 = left_net_goals

    bar = pygame.Surface((FIELD_W, 48), pygame.SRCALPHA)
    bar.fill((10, 18, 24, 175))
    surface.blit(bar, (0, 0))

    font = _hud_font()
    text = f"Team 1     {score_team1}  —  {score_team2}     Team 2"
    fg = (248, 252, 255)
    shadow = (16, 22, 30)

    main = font.render(text, True, fg)
    shade = font.render(text, True, shadow)
    rect = main.get_rect(center=(FIELD_W // 2, 24))
    for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        surface.blit(shade, rect.move(ox, oy))
    surface.blit(main, rect)


def render_game(surface: pygame.Surface, gs: GameState) -> None:
    draw_pitch(surface)
    draw_ball(surface, gs.ball)

    opp1, opp2 = gs.opposing_team_players()
    draw_player(surface, opp1, jersey=(48, 128, 86), trim=(200, 235, 210))
    _draw_kick_flash_ring_if_active(surface, gs, opp1)
    _draw_team_roster_label(surface, gs, opp1)
    draw_player(surface, opp2, jersey=(32, 112, 74), trim=(190, 228, 205))
    _draw_kick_flash_ring_if_active(surface, gs, opp2)
    _draw_team_roster_label(surface, gs, opp2)

    ctl1, ctl2 = gs.controlled_team_players()
    draw_player(surface, ctl1, jersey=(42, 98, 180), trim=(230, 210, 80))
    _draw_kick_flash_ring_if_active(surface, gs, ctl1)
    _draw_team_roster_label(surface, gs, ctl1)
    draw_player(surface, ctl2, jersey=(36, 86, 170), trim=(210, 200, 80))
    _draw_kick_flash_ring_if_active(surface, gs, ctl2)
    _draw_team_roster_label(surface, gs, ctl2)

    if gs.opponent is not None:
        draw_player(
            surface,
            gs.opponent,
            jersey=(182, 58, 52),
            trim=(240, 225, 200),
        )
        _draw_kick_flash_ring_if_active(surface, gs, gs.opponent)
    draw_score_hud(surface, gs)
