"""Settings for ``train/train_ppo_single.py`` (primary ego only).

Reward shaping for **all** trainers is ``sim.reward_weights.DEFAULT_TEAM_RL_REWARD_WEIGHTS``
(``TeamRlRewardWeights``). Edit ``sim/reward_weights.py`` to change dense / goal coefficients.

This module holds single-run PPO hyperparameters + env timing (``frame_skip``, stagnant ball,
logging paths).
"""

from __future__ import annotations

from sim.constants import DEFAULT_MAX_TICKS

# ---------------------------------------------------------------------------
# AttackerTeamRlEnv timing (not part of TeamRlRewardWeights)
# ---------------------------------------------------------------------------
STAGNANT_BALL_ENV_STEPS = 20
STAGNANT_BALL_PENALTY = -0.01
STAGNANT_BALL_SPEED_EPS = 12.0

# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
FRAME_SKIP = 4
MAX_TICKS = DEFAULT_MAX_TICKS

# ---------------------------------------------------------------------------
# PPO (Stable-Baselines3)
# ---------------------------------------------------------------------------
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01

# ---------------------------------------------------------------------------
# Logging / checkpoints (relative to project root or cwd)
# ---------------------------------------------------------------------------
LOG_DIR = "./logs/ppo_football_single"
MODEL_DIR = "./models"
# File prefix: ppo_football_1ego_{save_tag}_*  or ppo_football_1ego_* when save_tag omitted
MODEL_NAME_PREFIX = "1ego"

EVAL_N_EPISODES = 10
# Eval every N *goal* env-steps / width; same shape as dual script: 25_000 // n_envs
EVAL_FREQ_NUMERATOR = 500_000
CKPT_FREQ_NUMERATOR = 1_000_000

TOTAL_TIMESTEPS_DEFAULT = 1_000_000
N_ENVS_DEFAULT = 12
