"""Reuse a +x-goal–trained policy when the ego must score on −x (mirror across pitch length).

Observation: negate every x-position and x–velocity scalar in the training vector so the
network still sees ``attack_target`` semantics as trained. Action (after predict): negate
``move_x`` only before passing to ``TickActions``.

Applying any mirror helper twice restores the original (involution).

See ``OBS_X_MIRROR_DIMS`` / ``OBS_ACTION_MIRROR_AXES`` — keep in sync with
``sim/observation.py`` ordering.
"""

from __future__ import annotations

import numpy as np

from sim.constants import OBS_DIM

# 27-D obs (see ``build_observation``): negate ego x,vx; ball rel_x, vx;
# goal-vector x; teammate + each opposing-team slot (Δx/FIELD_W and vx/max).
# Last two dims (match score / ``MATCH_WIN_GOALS``) are not mirrored — world tallies are shared.
OBS_X_MIRROR_DIMS: tuple[int, ...] = (
    0,
    2,  # player x/W, vx
    4,
    6,  # rel_px; ball vx
    8,  # ox / FIELD_W
    13,
    15,  # teammate
    17,
    19,  # opposing_team slot 1 block
    21,
    23,  # opposing_team slot 2 block
)


def mirror_observation_for_right_trained_policy(obs: np.ndarray) -> np.ndarray:
    """Return a copy suitable as input for a ``right-side`` attacker policy."""
    a = np.asarray(obs, dtype=np.float64).reshape(-1)
    if a.shape[-1] != OBS_DIM:
        raise ValueError(f"expected obs dim {OBS_DIM}, got shape {obs.shape}")
    out = np.array(a, copy=True)
    for i in OBS_X_MIRROR_DIMS:
        out[..., i] = -out[..., i]
    return out


# Action slice [move_x, move_y, kick]: horizontal stick only mirrored.
OBS_ACTION_MIRROR_AXES: tuple[int, ...] = (0,)


def mirror_action_for_right_trained_policy(action: np.ndarray) -> np.ndarray:
    """Map policy output → sim controls for a mirrored (left-attacking) agent."""
    a = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    for i in OBS_ACTION_MIRROR_AXES:
        a[i] = -a[i]
    return a


__all__ = [
    "OBS_ACTION_MIRROR_AXES",
    "OBS_X_MIRROR_DIMS",
    "mirror_action_for_right_trained_policy",
    "mirror_observation_for_right_trained_policy",
]
