"""Player–player overlap resolution (watch / local multi-entity)."""

from __future__ import annotations

import math


def resolve_player_player(a, b) -> bool:
    dx = float(b.x - a.x)
    dy = float(b.y - a.y)
    ra, rb = float(a.radius), float(b.radius)
    min_d = ra + rb
    dist_sq = dx * dx + dy * dy
    min_d_sq = min_d * min_d
    if dist_sq >= min_d_sq or dist_sq < 1e-18:
        return False
    dist = math.sqrt(dist_sq)
    nx, ny = dx / dist, dy / dist
    penetration = min_d - dist
    half = penetration * 0.5
    a.x -= nx * half
    a.y -= ny * half
    b.x += nx * half
    b.y += ny * half
    a._clamp_to_pitch()
    b._clamp_to_pitch()
    return True
