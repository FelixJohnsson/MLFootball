"""Standard Gymnasium wrapper around AttackerTeamRlEnv for single-ego training.

Exposes the primary ego's 27-D observation and 3-D action (move_x, move_y,
kick_power) through the standard ``reset()`` / ``step()`` API so that
SB3's ``SubprocVecEnv`` can run each match in its own process.

The partner slot receives idle (0, 0, 0) actions every step.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.attacker_team_env import AttackerTeamRlEnv
from sim.constants import OBS_DIM

_IDLE_PARTNER = np.array([0.0, 0.0, 0.0], dtype=np.float64)


class SingleEgoGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs) -> None:
        super().__init__()
        self._env = AttackerTeamRlEnv(**env_kwargs)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._env.reset(seed=seed)
        return self._env.observation(0), {}

    def step(self, action):
        a_team = np.concatenate([
            np.asarray(action, dtype=np.float64), _IDLE_PARTNER,
        ])
        reward, terminated, truncated, info = self._env.step_team(a_team)
        obs = self._env.observation(0)
        return obs, float(reward), terminated, truncated, info