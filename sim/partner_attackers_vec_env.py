"""VecEnv: one or two RL ego rows per ``AttackerTeamRlEnv`` (27-D obs, 3-D act per ego).

LEFT controlled team ↔ +x scoring; RIGHT ↔ −x (`controlled_scores_into_right_net` on ``GameState``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.attacker_team_env import AttackerTeamRlEnv
from sim.constants import OBS_DIM

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


_IDLE_PARTNER_ACT = np.array([0.0, 0.0, 0.0], dtype=np.float64)


class PartnerAttackersVecEnv(VecEnv):
    """
    Each factory builds one ``AttackerTeamRlEnv`` (one 2v2 **controlled** pairing per world).

    **Dual** (default): ``num_envs = 2 * num_pairs`` — consecutive rows share weights on both
    teammates; joint 6-D action; reward ``0.5 * team_reward`` per row.

    **Single**: ``num_envs = num_pairs`` — one row per match controlling **primary** ego only;
    partner gets idle stick/kick; that row receives the **full** team reward.
    """

    def __init__(
        self,
        pair_env_factories: list[Callable[[], AttackerTeamRlEnv]],
        *,
        dual_ego_training: bool = True,
    ) -> None:
        self.dual_ego_training = dual_ego_training
        self.pair_envs = [fn() for fn in pair_env_factories]
        num_pairs = len(self.pair_envs)
        if num_pairs == 0:
            raise ValueError("need at least one team environment")
        num_envs = num_pairs * 2 if dual_ego_training else num_pairs

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        super().__init__(num_envs, obs_space, act_space)

        self.actions: np.ndarray | None = None
        self.buf_obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
        self.buf_rews = np.zeros((num_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((num_envs,), dtype=bool)
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    def reset(self) -> VecEnvObs:
        if self.dual_ego_training:
            for k, env in enumerate(self.pair_envs):
                seed = self._seeds[2 * k]
                env.reset(seed=seed)
                self.buf_obs[2 * k] = env.observation(0)
                self.buf_obs[2 * k + 1] = env.observation(1)
                self.reset_infos[2 * k] = {}
                self.reset_infos[2 * k + 1] = {}
        else:
            for k, env in enumerate(self.pair_envs):
                seed = self._seeds[k]
                env.reset(seed=seed)
                self.buf_obs[k] = env.observation(0)
                self.reset_infos[k] = {}

        self._reset_seeds()
        self._reset_options()
        return np.copy(self.buf_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        assert self.actions is not None
        if self.dual_ego_training:
            return self._step_wait_dual()
        return self._step_wait_single()

    def _step_wait_dual(self) -> VecEnvStepReturn:
        for k, env in enumerate(self.pair_envs):
            i0, i1 = 2 * k, 2 * k + 1
            a_team = np.concatenate(
                [self.actions[i0].astype(np.float64), self.actions[i1].astype(np.float64)]
            )
            r_team, terminated, truncated, raw_info = env.step_team(a_team)
            done = terminated or truncated

            o0_u = env.observation(0)
            o1_u = env.observation(1)

            r_each = 0.5 * float(r_team)
            self.buf_rews[i0] = r_each
            self.buf_rews[i1] = r_each
            self.buf_dones[i0] = done
            self.buf_dones[i1] = done

            base_info = {
                kk: v for kk, v in raw_info.items() if kk != "TimeLimit.truncated"
            }
            tl = bool(
                raw_info.get("TimeLimit.truncated", truncated and not terminated)
            )
            self.buf_infos[i0] = deepcopy(base_info)
            self.buf_infos[i1] = deepcopy(base_info)
            self.buf_infos[i0]["TimeLimit.truncated"] = tl
            self.buf_infos[i1]["TimeLimit.truncated"] = tl

            if done:
                self.buf_infos[i0]["terminal_observation"] = np.copy(o0_u)
                self.buf_infos[i1]["terminal_observation"] = np.copy(o1_u)
                env.reset()
                o0_u = env.observation(0)
                o1_u = env.observation(1)

            self.buf_obs[i0] = o0_u
            self.buf_obs[i1] = o1_u

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def _step_wait_single(self) -> VecEnvStepReturn:
        for k, env in enumerate(self.pair_envs):
            a_team = np.concatenate([self.actions[k].astype(np.float64), _IDLE_PARTNER_ACT])
            r_team, terminated, truncated, raw_info = env.step_team(a_team)
            done = terminated or truncated

            o0_u = env.observation(0)

            self.buf_rews[k] = float(r_team)
            self.buf_dones[k] = done

            base_info = {
                kk: v for kk, v in raw_info.items() if kk != "TimeLimit.truncated"
            }
            tl = bool(
                raw_info.get("TimeLimit.truncated", truncated and not terminated)
            )
            self.buf_infos[k] = deepcopy(base_info)
            self.buf_infos[k]["TimeLimit.truncated"] = tl

            if done:
                self.buf_infos[k]["terminal_observation"] = np.copy(o0_u)
                env.reset()
                o0_u = env.observation(0)

            self.buf_obs[k] = o0_u

        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        pass

    def get_images(self) -> Sequence[np.ndarray | None]:
        raise NotImplementedError

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        indices = list(self._get_indices(indices))
        out: list[Any] = []
        for i in indices:
            pi = i // 2 if self.dual_ego_training else i
            pair = self.pair_envs[pi]
            if attr_name == "render_mode":
                out.append(pair.render_mode)
            elif attr_name == "metadata":
                out.append(pair.metadata)
            elif attr_name == "spec":
                out.append(None)
            else:
                out.append(getattr(pair, attr_name))
        return out

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        for i in self._get_indices(indices):
            pi = i // 2 if self.dual_ego_training else i
            setattr(self.pair_envs[pi], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support env_method ({method_name})."
        )

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
