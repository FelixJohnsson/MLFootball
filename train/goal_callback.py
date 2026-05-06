"""TensorBoard: goals and steps-to-score (Monitor); left vs right net are world −x / +x mouths."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GoalCountCallback(BaseCallback):
    """
    - Per rollout: goal counts per net; optional steps-to-score length when episode
      ends on a goal (one list per net).
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._right = 0
        self._left = 0
        self._steps_to_score_left: list[float] = []
        self._steps_to_score_right: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            sn = info.get("scoring_net")
            if sn == "right":
                self._right += 1
            elif sn == "left":
                self._left += 1

            episode = info.get("episode")
            if episode is not None:
                length = episode.get("l")
                if length is not None:
                    if sn == "left":
                        self._steps_to_score_left.append(float(length))
                    elif sn == "right":
                        self._steps_to_score_right.append(float(length))

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("goals/right_net", float(self._right))
        self.logger.record("goals/left_net", float(self._left))
        self.logger.record("goals/total", float(self._right + self._left))
        self._right = 0
        self._left = 0

        for label, xs in (
            ("goals/steps_to_score_left", self._steps_to_score_left),
            ("goals/steps_to_score_right", self._steps_to_score_right),
        ):
            n = len(xs)
            key_ep = label + "_rollout_episodes"
            self.logger.record(key_ep, float(n))
            if n:
                arr = np.array(xs, dtype=np.float64)
                self.logger.record(label + "_mean", float(np.mean(arr)))
                self.logger.record(label + "_min", float(np.min(arr)))
                self.logger.record(label + "_max", float(np.max(arr)))
            xs.clear()
