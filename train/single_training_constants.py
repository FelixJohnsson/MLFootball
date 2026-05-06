"""Settings for ``train/train_ppo_single.py`` (primary ego only).

Dual shared-weights training uses defaults in ``sim.reward_weights.DEFAULT_TEAM_RL_REWARD_WEIGHTS`` unless overridden.

Edit values here — they are wired into ``TeamRlRewardWeights`` for single-ego runs only.
"""

from __future__ import annotations

from sim.reward_weights import TeamRlRewardWeights
from sim.constants import DEFAULT_MAX_TICKS

# ---------------------------------------------------------------------------
# Rewards (applied inside ``AttackerTeamRlEnv.step_team`` each frame_skip tick accumulation)
# ---------------------------------------------------------------------------
STEP_PENALTY_PER_ENV_STEP = 0.01

BALL_TO_SCORING_TARGET_COEF = 8.0
BALL_APPROACH_OPPOSITE_NET_PENALTY_COEF = 4.0

# Chase shaping: ego ball proximity Δ; partner slot defaults to **0** in single-ego runs.
CTRL_PRIMARY_TO_BALL_COEF = 1.0
CTRL_PARTNER_TO_BALL_COEF = 0.0

GOAL_CORRECT_NET_BONUS = 50.0
GOAL_WRONG_NET_PENALTY = 50.0

STAGNANT_BALL_ENV_STEPS = 20
STAGNANT_BALL_PENALTY = -0.01
STAGNANT_BALL_SPEED_EPS = 12.0

SINGLE_REWARD_WEIGHTS = TeamRlRewardWeights(
    step_penalty_per_env_step=STEP_PENALTY_PER_ENV_STEP,
    ball_to_scoring_target_coef=BALL_TO_SCORING_TARGET_COEF,
    ball_opposite_net_penalty_coef=BALL_APPROACH_OPPOSITE_NET_PENALTY_COEF,
    ctrl_primary_to_ball_coef=CTRL_PRIMARY_TO_BALL_COEF,
    ctrl_partner_to_ball_coef=CTRL_PARTNER_TO_BALL_COEF,
    goal_correct_net_bonus=GOAL_CORRECT_NET_BONUS,
    goal_wrong_net_penalty=GOAL_WRONG_NET_PENALTY,
)

# ---------------------------------------------------------------------------
# Env construction (AttackerTeamRlEnv kwargs)
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
MODEL_NAME_PREFIX = "ppo_football_1ego"

EVAL_N_EPISODES = 20
# Eval every N *goal* env-steps / width; same shape as dual script: 25_000 // n_envs
EVAL_FREQ_NUMERATOR = 25_000
CKPT_FREQ_NUMERATOR = 50_000

TOTAL_TIMESTEPS_DEFAULT = 1_000_000
N_ENVS_DEFAULT = 8
