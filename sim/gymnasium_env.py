"""RL training helpers: team sim + vec wrapper (27-D ego, 3-D act per policy call)."""

from __future__ import annotations

from sim.attacker_team_env import AttackerTeamRlEnv

# Backwards-compatible name: not a Gymnasium Env; use PartnerAttackersVecEnv for PPO.
MLFootballEnv = AttackerTeamRlEnv

__all__ = [
    "AttackerTeamRlEnv",
    "MLFootballEnv",
]
