"""Tune sim horizon, kicking, and vector obs size.

Team rule: LEFT team → +x goal; RIGHT team → −x (see ``GameState.controlled_scores_into_right_net``).
"""

KICK_IMPULSE = 560.0
DEFAULT_MAX_TICKS = 5400  # 90 s wall-clock at 60 Hz with dt = 1/60
KICK_COOLDOWN_TICKS = 45  # cooldown after successful contact kick (~0.75 s @ 60 Hz)
KICK_FLASH_TICKS = 12  # pygame ring duration: visible while ``tick_count < player.kick_flash_expire_at``
OBS_DIM = 25  # Single ego-attacker observation (includes teammate slot)
TEAM_ATTACKER_ACT_DIM = 6  # Two 3-D TickActions flattened (not a policy I/O bundle)

FROZEN_TEAM_HEURISTIC_CHASE = True