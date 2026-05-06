"""Player–player overlap resolution (watch / local multi-entity)."""

from __future__ import annotations

import math


def resolve_player_player(a, b) -> None:
    """Push two players apart if their circles overlap; clamp to pitch."""
    dx = float(b.x - a.x)
    dy = float(b.y - a.y)
    dist = math.hypot(dx, dy)
    ra, rb = float(a.radius), float(b.radius)
    min_d = ra + rb
    if dist >= min_d or dist < 1e-9:
        return
    nx, ny = dx / dist, dy / dist
    penetration = min_d - dist
    half = penetration * 0.5
    a.x -= nx * half
    a.y -= ny * half
    b.x += nx * half
    b.y += ny * half
    a._clamp_to_pitch()
    b._clamp_to_pitch()
