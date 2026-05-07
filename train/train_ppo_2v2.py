"""PPO 2v2: shared-weights dual ego vs a frozen opponent model.

Rewards: ``sim.reward_weights.DEFAULT_TEAM_RL_REWARD_WEIGHTS``.

Each match runs in its own subprocess.  Both controlled teammates share a
single 27→3 PPO policy; the opposing team is driven by a separately-trained
checkpoint loaded inside each worker process (no mirroring — native obs).

Usage
-----
    # Train LEFT team vs frozen RIGHT opponent
    python train/train_ppo_2v2.py --left_team \\
        --opponent_model models/ppo_football_1ego_RIGHT_TEAM_v3_final \\
        --steps 1000000 --n_envs 8 --save_tag LEFT_vs_RIGHT_v1

    # Train RIGHT team vs frozen LEFT opponent
    python train/train_ppo_2v2.py --right_team \\
        --opponent_model models/ppo_football_2v2_v5_final_LEFT_TEAM \\
        --steps 1000000 --n_envs 8 --save_tag RIGHT_vs_LEFT_v1

    # Resume a previous run
    python train/train_ppo_2v2.py --left_team \\
        --opponent_model models/ppo_football_1ego_RIGHT_TEAM_v3_final \\
        --resume models/checkpoints_LEFT_vs_RIGHT_v1/ppo_football_2v2_LEFT_vs_RIGHT_v1_500000_steps \\
        --save_tag LEFT_vs_RIGHT_v1

n_envs must be **even** (two rollout rows per match).
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.append(str(_TRAIN_DIR))

from goal_callback import GoalCountCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from field import attack_target_goal_xy, left_attack_target_goal_xy

from sim.reward_weights import DEFAULT_TEAM_RL_REWARD_WEIGHTS
from sim.subproc_dual_vec_env import SubprocDualVecEnv
from sim.team_vs_opponent_env import TeamVsOpponentEnv
from sim.single_ego_gym_env import SingleEgoGymEnv


def print_reward_structure_verification(controlled_scores_into_right_net: bool) -> None:
    """Echo how ``AttackerTeamRlEnv.step_team`` maps nets to +/- reward (default weights).

    ``TeamVsOpponentEnv`` does not override ``reward_weights``; defaults apply.
    """
    rw = DEFAULT_TEAM_RL_REWARD_WEIGHTS
    tcx, tcy = attack_target_goal_xy()
    lcx, lcy = left_attack_target_goal_xy()
    sep = "=" * 70
    print(sep)
    print("Reward structure verification (AttackerTeamRlEnv / TeamVsOpponentEnv)")
    print(sep)

    if controlled_scores_into_right_net:
        roster = "LEFT_TEAM  (--left_team)"
        target_label = "+x mouth • GameState scoring_net tag 'right' (attack_target_goal_xy)"
        opposite_label = "−x mouth • tag 'left' (left_attack_target_goal_xy)"
        target_xy: tuple[float, float] = (tcx, tcy)
        opp_xy: tuple[float, float] = (lcx, lcy)
        sparse_lines = (
            f"  +{rw.goal_correct_net_bonus:g} when scoring_net == 'right' (ball in +x / attack goal).",
            f"  −{rw.goal_wrong_net_penalty:g} when scoring_net == 'left' (ball in −x net).",
        )
    else:
        roster = "RIGHT_TEAM  (--right_team)"
        target_label = "−x mouth • GameState scoring_net tag 'left' (left_attack_target_goal_xy)"
        opposite_label = "+x mouth • tag 'right' (attack_target_goal_xy)"
        target_xy = (lcx, lcy)
        opp_xy = (tcx, tcy)
        sparse_lines = (
            f"  +{rw.goal_correct_net_bonus:g} when scoring_net == 'left' (ball in −x / attack goal).",
            f"  −{rw.goal_wrong_net_penalty:g} when scoring_net == 'right' (ball in +x net).",
        )

    print(f"Controlled roster: {roster}")
    print(f"  Dense: reward ball moving closer to — {target_label}")
    print(f"         mouth centre ≈ ({target_xy[0]:.1f}, {target_xy[1]:.1f}) px")
    print(f"  Dense: penalize ball approaching — {opposite_label}")
    print(f"         mouth centre ≈ ({opp_xy[0]:.1f}, {opp_xy[1]:.1f}) px")
    print()
    print("Sparse goal terms (per sim tick inside frame_skip when a goal registers):")
    for line in sparse_lines:
        print(line)
    print()
    print("Other default shaping (TeamRlRewardWeights):")
    print(f"  step_penalty_per_env_step ....... −{rw.step_penalty_per_env_step:g}")
    print(f"  ball_to_scoring_target_coef ..... {rw.ball_to_scoring_target_coef:g}")
    print(f"  ball_opposite_net_penalty_coef .. {rw.ball_opposite_net_penalty_coef:g}")
    print(f"  chase_nearest_to_ball_coef ...... {rw.chase_nearest_to_ball_coef:g}")
    print(f"  partner_support_ball_potential ... {rw.partner_support_ball_potential_coef:g} (ideal≈{rw.partner_support_ideal_ball_dist_px:g}px)")
    print(f"  teammate_spread ................. thresh {rw.teammate_spread_threshold_px:g}px coef {rw.teammate_spread_penalty_coef:g}")
    print(
        f"  stagnant_crowding ............... touch {rw.stagnant_crowding_ball_touch_px:g}px "
        f"tick {rw.stagnant_crowding_tick_penalty:g} stall×{rw.stagnant_crowding_stall_multiplier:g}"
    )
    print(f"  off_pitch_penalty_coef .......... {rw.off_pitch_penalty_coef:g} (controlled only; /FIELD_DIAG)")
    print(f"  losing_per_goal_behind_coef ..... {rw.losing_per_goal_behind_coef:g} (− per goal behind, each env step)")
    print()


# ---------------------------------------------------------------------------
# Defaults (mirror single_training_constants structure)
# ---------------------------------------------------------------------------
FRAME_SKIP = 4
MAX_TICKS = 5400
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 10
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01

LOG_DIR = "./logs/ppo_football_2v2"
MODEL_DIR = "./models"
MODEL_NAME_PREFIX = "ppo_football_2v2"
EVAL_N_EPISODES = 10
EVAL_FREQ_NUMERATOR = 500_000
CKPT_FREQ_NUMERATOR = 1_000_000


# ---------------------------------------------------------------------------
# Env factories (must be picklable for spawn)
# ---------------------------------------------------------------------------

def _env_kwargs(
    controlled_scores_into_right_net: bool,
    opponent_model_path: str,
    frame_skip: int,
) -> dict:
    return dict(
        max_ticks=MAX_TICKS,
        dt=1.0 / 60.0,
        frame_skip=frame_skip,
        controlled_scores_into_right_net=controlled_scores_into_right_net,
        opponent_model_path=opponent_model_path,
        reward_weights=DEFAULT_TEAM_RL_REWARD_WEIGHTS,
    )


def _init_match_env(seed: int, **env_kwargs):
    env = TeamVsOpponentEnv(**env_kwargs)
    env.reset(seed=seed)
    return env

def _make_match_env(seed: int, env_kwargs: dict):
    return functools.partial(_init_match_env, seed, **env_kwargs)


def _init_eval_env(seed: int, **env_kwargs):
    env = TeamVsOpponentEnv(**env_kwargs)
    env.reset(seed=seed)
    return _EvalWrapper(env)

def _make_eval_env(seed: int, env_kwargs: dict):
    return functools.partial(_init_eval_env, seed, **env_kwargs)


class _EvalWrapper(SingleEgoGymEnv):
    """Thin wrapper: reuse SingleEgoGymEnv but with a pre-built env."""

    def __init__(self, env: TeamVsOpponentEnv) -> None:
        # Skip SingleEgoGymEnv.__init__ — set fields directly
        import gymnasium as gym
        from gymnasium import spaces
        from sim.constants import OBS_DIM
        import numpy as np

        gym.Env.__init__(self)
        self._env = env
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO 2v2 · shared weights vs frozen opponent")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    p.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Rollout width (must be even: 2 ego rows per match subprocess).",
    )
    p.add_argument("--frame_skip", type=int, default=FRAME_SKIP, help="Sim ticks per agent step")
    p.add_argument(
        "--opponent_model",
        type=str,
        required=True,
        metavar="PATH",
        help="Frozen opponent checkpoint (without .zip)",
    )
    p.add_argument("--log_dir", type=str, default=LOG_DIR)
    p.add_argument("--model_dir", type=str, default=MODEL_DIR)
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Continue from SB3 checkpoint (without .zip)",
    )
    p.add_argument("--save_tag", type=str, default=None, metavar="LABEL")

    team = p.add_mutually_exclusive_group(required=True)
    team.add_argument(
        "--left_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=True,
        help="Train LEFT_TEAM (+x net) vs frozen RIGHT opponent.",
    )
    team.add_argument(
        "--right_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=False,
        help="Train RIGHT_TEAM (−x net) vs frozen LEFT opponent.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.n_envs < 2 or args.n_envs % 2 != 0:
        raise SystemExit("--n_envs must be >= 2 and even (two ego rows per match).")

    # Resolve opponent path (accept with or without .zip)
    opp_path = Path(args.opponent_model)
    if opp_path.suffix.lower() != ".zip":
        opp_zip = opp_path.with_suffix(".zip")
        if opp_zip.exists():
            opp_path = opp_zip
    if not opp_path.exists():
        raise SystemExit(f"Opponent model not found: {opp_path}")
    opponent_model_str = str(opp_path.with_suffix(""))  # SB3 wants no .zip

    side = "LEFT" if args.controlled_scores_into_right_net else "RIGHT"
    print(f"2v2 dual-ego training: {side} team vs frozen opponent ({opp_path.name})")
    print_reward_structure_verification(args.controlled_scores_into_right_net)

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)

    model_base = f"{MODEL_NAME_PREFIX}_{args.save_tag}" if args.save_tag else MODEL_NAME_PREFIX
    if args.save_tag:
        best_dir = model_dir / f"best_model_{args.save_tag}"
        ckpt_dir = model_dir / f"checkpoints_{args.save_tag}"
    else:
        best_dir = model_dir / "best_model_2v2"
        ckpt_dir = model_dir / "checkpoints_2v2"
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_matches = args.n_envs // 2
    ekw = _env_kwargs(args.controlled_scores_into_right_net, opponent_model_str, args.frame_skip)

    # --- Training: SubprocDualVecEnv (2 slots per worker) ---
    train_env = SubprocDualVecEnv(
        [_make_match_env(k, ekw) for k in range(n_matches)],
        start_method="spawn",
    )
    train_envs = VecMonitor(train_env)

    # --- Eval: single-ego in main process (just 1 env) ---
    eval_env = DummyVecEnv([_make_eval_env(10_000, ekw)])
    eval_env = VecMonitor(eval_env)

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=EVAL_N_EPISODES,
        eval_freq=max(EVAL_FREQ_NUMERATOR // args.n_envs, 1),
        best_model_save_path=str(best_dir),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(CKPT_FREQ_NUMERATOR // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=model_base,
        verbose=1,
    )
    goals_cb = GoalCountCallback()

    if args.resume:
        rp = Path(args.resume)
        base = rp.parent / rp.stem if rp.suffix.lower() == ".zip" else rp
        print(f"Resuming from {base}.zip …")
        model = PPO.load(str(base), env=train_envs, device="cpu", tensorboard_log=str(log_dir))
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_envs,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            learning_rate=PPO_LEARNING_RATE,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            clip_range=PPO_CLIP_RANGE,
            ent_coef=PPO_ENT_COEF,
            device="cpu",
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    print(
        f"Training for {args.steps:,} steps, "
        f"n_envs={args.n_envs} ({n_matches} matches × 2 egos), "
        f"SubprocDualVecEnv"
    )
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([eval_cb, ckpt_cb, goals_cb]),
        progress_bar=True,
        reset_num_timesteps=bool(args.resume is None),
    )

    final_path = model_dir / f"{model_base}_final_{side}_TEAM"
    model.save(str(final_path))
    print(f"Final model saved -> {final_path}.zip")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
