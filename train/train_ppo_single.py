"""PPO on **one** controlled ego per match (primary slot); teammate idle.

Hyperparameters, reward shaping, and logging paths live in ``train/single_training_constants.py``.
For shared-weights **dual** training use ``train/train_ppo.py``.

Usage
-----
    python train/train_ppo_single.py --steps 500000
    python train/train_ppo_single.py --right_team --n_envs 8
    python train/train_ppo_single.py --save_tag run1 --resume models/ppo_football_1ego_run1_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.append(str(_TRAIN_DIR))

import single_training_constants as stc

from goal_callback import GoalCountCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from sim.single_ego_gym_env import SingleEgoGymEnv


def _env_kwargs(controlled_scores_into_right_net: bool, frame_skip: int) -> dict:
    """Shared env constructor kwargs (picklable plain dict)."""
    return dict(
        max_ticks=stc.MAX_TICKS,
        dt=1.0 / 60.0,
        frame_skip=frame_skip,
        stagnant_ball_env_steps=stc.STAGNANT_BALL_ENV_STEPS,
        stagnant_ball_penalty=stc.STAGNANT_BALL_PENALTY,
        stagnant_ball_speed_eps=stc.STAGNANT_BALL_SPEED_EPS,
        controlled_scores_into_right_net=controlled_scores_into_right_net,
        reward_weights=stc.SINGLE_REWARD_WEIGHTS,
    )


def _make_env(seed: int, env_kwargs: dict):
    """Return a zero-arg factory that SubprocVecEnv calls in the child process."""
    def _init():
        env = SingleEgoGymEnv(**env_kwargs)
        env.reset(seed=seed)
        return env
    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO · MLFootball (single primary ego)")
    p.add_argument("--steps", type=int, default=stc.TOTAL_TIMESTEPS_DEFAULT, help="Total training timesteps")
    p.add_argument(
        "--n_envs",
        type=int,
        default=stc.N_ENVS_DEFAULT,
        help="Parallel matches (one per subprocess).",
    )
    p.add_argument("--frame_skip", type=int, default=stc.FRAME_SKIP, help="Sim ticks per agent step")
    p.add_argument("--log_dir", type=str, default=stc.LOG_DIR, help="TensorBoard log dir")
    p.add_argument("--model_dir", type=str, default=stc.MODEL_DIR, help="Model save dir")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Continue from SB3 checkpoint (no .zip suffix)",
    )
    p.add_argument(
        "--save_tag",
        type=str,
        default=None,
        metavar="LABEL",
        help=f"Save under {stc.MODEL_NAME_PREFIX}_<tag>_* and models/best_model_<tag>/ …",
    )
    team = p.add_mutually_exclusive_group()
    team.add_argument(
        "--left_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=True,
        help="Train LEFT_TEAM (+x net primary). Same as default.",
    )
    team.add_argument(
        "--right_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=False,
        help="Train RIGHT_TEAM (−x net primary). Mirrored opposing policy disabled.",
    )
    p.set_defaults(controlled_scores_into_right_net=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_envs < 1:
        raise SystemExit("--n_envs must be >= 1")

    roster = (
        "LEFT team (+x primary)"
        if args.controlled_scores_into_right_net
        else "RIGHT team (−x primary) — no mirrored opposing policy"
    )
    print(f"Single-ego training: {roster} (see train/single_training_constants.py for reward/PPO knobs)")

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)

    model_base = f"{stc.MODEL_NAME_PREFIX}_{args.save_tag}" if args.save_tag else stc.MODEL_NAME_PREFIX
    if args.save_tag:
        best_dir = model_dir / f"best_model_{args.save_tag}_1ego"
        ckpt_dir = model_dir / f"checkpoints_{args.save_tag}_1ego"
    else:
        best_dir = model_dir / "best_model_1ego"
        ckpt_dir = model_dir / "checkpoints_1ego"
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ekw = _env_kwargs(args.controlled_scores_into_right_net, args.frame_skip)

    # --- Training envs: one subprocess per match ---
    train_env = SubprocVecEnv(
        [_make_env(k, ekw) for k in range(args.n_envs)],
        start_method="spawn",
    )
    train_envs = VecMonitor(train_env)

    # --- Eval env: single process is fine for 1 env ---
    eval_env = DummyVecEnv([_make_env(10_000, ekw)])
    eval_env = VecMonitor(eval_env)

    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=stc.EVAL_N_EPISODES,
        eval_freq=max(stc.EVAL_FREQ_NUMERATOR // args.n_envs, 1),
        best_model_save_path=str(best_dir),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(stc.CKPT_FREQ_NUMERATOR // args.n_envs, 1),
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
            n_steps=stc.PPO_N_STEPS,
            batch_size=stc.PPO_BATCH_SIZE,
            n_epochs=stc.PPO_N_EPOCHS,
            learning_rate=stc.PPO_LEARNING_RATE,
            gamma=stc.PPO_GAMMA,
            gae_lambda=stc.PPO_GAE_LAMBDA,
            clip_range=stc.PPO_CLIP_RANGE,
            ent_coef=stc.PPO_ENT_COEF,
            device="cpu",
            verbose=1,
            tensorboard_log=str(log_dir),
        )

    if args.controlled_scores_into_right_net:
        print("LEFT_TEAM: opposing team uses heuristic chase.")
    else:
        print("RIGHT_TEAM: LEFT_TEAM heuristic (no mirrored policy).")

    print(f"Training for {args.steps:,} steps, n_envs={args.n_envs} (SubprocVecEnv) …")
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([eval_cb, ckpt_cb, goals_cb]),
        progress_bar=True,
        reset_num_timesteps=bool(args.resume is None),
    )

    final_path = model_dir / f"{model_base}_final"
    model.save(str(final_path))
    print(rf"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║   __  __ _      _____           _   _           _ _          ║
        ║  |  \/  | |    |  ___|__   ___ | |_| |__   __ _| | |         ║
        ║  | |\/| | |    | |_ / _ \ / _ \| __| '_ \ / _` | | |         ║
        ║  | |  | | |___ |  _| (_) | (_) | |_| |_) | (_| | | |         ║
        ║  |_|  |_|_____||_|  \___/ \___/ \__|_.__/ \__,_|_|_|         ║
        ║                                                              ║
        ║                 2v2 AGENT TRAINING COMPLETE                  ║
        ║                                                              ║
        ║        O     O                         O     O               ║
        ║       /|\   /|\          ●            /|\   /|\              ║
        ║       / \   / \        _/ \_          / \   / \              ║
        ║                                                              ║
        ║             FINAL MODEL SAVED TO THE LOCKER ROOM             ║
        ║                                                              ║
        ║             → {final_path}.zip                               ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """)

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()