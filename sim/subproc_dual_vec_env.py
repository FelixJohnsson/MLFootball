"""SubprocDualVecEnv — dual-ego shared-weights training across subprocesses.

Each worker subprocess manages **one** match (a ``TeamVsOpponentEnv`` or
``AttackerTeamRlEnv``).  The parent VecEnv exposes **two** logical env slots
per match (one per controlled ego), so ``num_envs = 2 * n_matches``.

Both egos share the same PPO policy (27-D obs → 3-D action).  Each receives
``0.5 * team_reward``.  This preserves the shared-weights architecture from
``PartnerAttackersVecEnv`` but runs matches in parallel.

Usage::

    from sim.team_vs_opponent_env import TeamVsOpponentEnv

    env_fns = [
        lambda: TeamVsOpponentEnv(
            opponent_model_path="models/opp.zip",
            controlled_scores_into_right_net=True,
        )
        for _ in range(4)
    ]
    vec = SubprocDualVecEnv(env_fns)  # num_envs = 8
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from sim.constants import OBS_DIM

# Worker commands
_CMD_STEP = 0
_CMD_RESET = 1
_CMD_CLOSE = 2


def _worker(conn: mp.connection.Connection, env_fn: Callable) -> None:
    """Event loop for one match subprocess."""
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()

            if cmd == _CMD_STEP:
                act_ego0, act_ego1 = data
                a_team = np.concatenate([
                    act_ego0.astype(np.float64),
                    act_ego1.astype(np.float64),
                ])
                reward, terminated, truncated, info = env.step_team(a_team)
                done = terminated or truncated

                obs0 = env.observation(0)
                obs1 = env.observation(1)

                result_info = {
                    k: v for k, v in info.items() if k != "TimeLimit.truncated"
                }
                tl = bool(info.get("TimeLimit.truncated", truncated and not terminated))

                terminal_obs0 = None
                terminal_obs1 = None
                if done:
                    terminal_obs0 = np.copy(obs0)
                    terminal_obs1 = np.copy(obs1)
                    env.reset()
                    obs0 = env.observation(0)
                    obs1 = env.observation(1)

                conn.send((obs0, obs1, reward, done, tl, result_info, terminal_obs0, terminal_obs1))

            elif cmd == _CMD_RESET:
                seed = data
                env.reset(seed=seed)
                conn.send((env.observation(0), env.observation(1)))

            elif cmd == _CMD_CLOSE:
                break
    finally:
        conn.close()


class SubprocDualVecEnv(VecEnv):
    """Two logical env slots per subprocess worker (shared-weights dual ego)."""

    def __init__(
        self,
        env_fns: list[Callable],
        *,
        start_method: str = "spawn",
    ) -> None:
        n_matches = len(env_fns)
        if n_matches < 1:
            raise ValueError("need at least one env factory")
        num_envs = n_matches * 2

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        super().__init__(num_envs, obs_space, act_space)

        self.n_matches = n_matches
        ctx = mp.get_context(start_method)
        self._parent_conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for fn in env_fns:
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child_conn, fn), daemon=True)
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

        self.actions: np.ndarray | None = None
        self.buf_obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
        self.buf_rews = np.zeros(num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(num_envs, dtype=bool)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    # -- VecEnv interface ------------------------------------------------------

    def reset(self) -> VecEnvObs:
        for k, conn in enumerate(self._parent_conns):
            seed = self._seeds[2 * k] if 2 * k < len(self._seeds) else k
            conn.send((_CMD_RESET, seed))
        for k, conn in enumerate(self._parent_conns):
            obs0, obs1 = conn.recv()
            self.buf_obs[2 * k] = obs0
            self.buf_obs[2 * k + 1] = obs1
            self.reset_infos[2 * k] = {}
            self.reset_infos[2 * k + 1] = {}
        self._reset_seeds()
        self._reset_options()
        return np.copy(self.buf_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        assert self.actions is not None

        # Send paired actions
        for k, conn in enumerate(self._parent_conns):
            i0, i1 = 2 * k, 2 * k + 1
            conn.send((_CMD_STEP, (self.actions[i0], self.actions[i1])))

        # Receive results
        for k, conn in enumerate(self._parent_conns):
            i0, i1 = 2 * k, 2 * k + 1
            obs0, obs1, reward, done, tl, raw_info, t_obs0, t_obs1 = conn.recv()

            r_each = 0.5 * float(reward)
            self.buf_obs[i0] = obs0
            self.buf_obs[i1] = obs1
            self.buf_rews[i0] = r_each
            self.buf_rews[i1] = r_each
            self.buf_dones[i0] = done
            self.buf_dones[i1] = done

            info0 = dict(raw_info)
            info1 = dict(raw_info)
            info0["TimeLimit.truncated"] = tl
            info1["TimeLimit.truncated"] = tl

            if done and t_obs0 is not None:
                info0["terminal_observation"] = t_obs0
                info1["terminal_observation"] = t_obs1

            self.buf_infos[i0] = info0
            self.buf_infos[i1] = info1

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        for conn in self._parent_conns:
            try:
                conn.send((_CMD_CLOSE, None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    # -- Required ABC stubs ----------------------------------------------------

    def get_images(self):
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        if attr_name in ("render_mode", "spec"):
            return [None for _ in self._get_indices(indices)]
        if attr_name == "metadata":
            return [{"render_modes": []} for _ in self._get_indices(indices)]
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        raise NotImplementedError(f"SubprocDualVecEnv does not support env_method({method_name!r})")

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
