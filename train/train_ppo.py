"""Train PPO with **dual** rollout rows per match — shared weights on both controlled teammates.

Reward defaults are ``sim.reward_weights.DEFAULT_TEAM_RL_REWARD_WEIGHTS`` — edit ``sim/reward_weights.py``
to change shaping for this script and all other trainers.

For **single** primary-ego training use ``train/train_ppo_single.py`` and ``train/single_training_constants.py`` (PPO + paths only).

Default **LEFT_TEAM** run mirrors the current policy onto **RIGHT_TEAM** (two frozen ``predict`` calls per sim tick).
Use ``--opposing_heuristic`` with ``--left_team`` to drive **RIGHT_TEAM** with deterministic chase instead (no PPO).
``--right_team`` keeps **LEFT_TEAM** heuristic (no mirrored policy); ``--opposing_heuristic`` is then redundant.

Usage
-----
    python train/train_ppo.py
    python train/train_ppo.py --left_team
    python train/train_ppo.py --left_team --opposing_heuristic
    python train/train_ppo.py --right_team --steps 2000000
    python train/train_ppo.py --right_team --opponent_model models/1ego_LEFT_TEAM_final
    python train/train_ppo.py --n_envs 8
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
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from sim import AttackerTeamRlEnv
from sim.frozen_opposing_team_policy import SharedPolicyRef
from sim.partner_attackers_vec_env import PartnerAttackersVecEnv
from sim.reward_weights import DEFAULT_TEAM_RL_REWARD_WEIGHTS


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def team_factory(
    seed: int,
    *,
    frame_skip: int,
    opposing_team_policy_ref: SharedPolicyRef,
    controlled_scores_into_right_net: bool,
) -> AttackerTeamRlEnv:
    e = AttackerTeamRlEnv(
        frame_skip=frame_skip,
        opposing_team_policy_ref=opposing_team_policy_ref,
        controlled_scores_into_right_net=controlled_scores_into_right_net,
        reward_weights=DEFAULT_TEAM_RL_REWARD_WEIGHTS,
    )
    e.reset(seed=seed)
    return e


def build_partner_vec(
    n_slots: int,
    frame_skip: int,
    *,
    seed_base: int = 0,
    opposing_team_policy_ref: SharedPolicyRef,
    controlled_scores_into_right_net: bool,
) -> PartnerAttackersVecEnv:
    if n_slots < 2 or n_slots % 2 != 0:
        raise ValueError("--n_envs must be >= 2 and even (dual-ego rows per rollout).")
    n_pairs = n_slots // 2
    fns = [
        functools.partial(
            team_factory,
            seed_base + k,
            frame_skip=frame_skip,
            opposing_team_policy_ref=opposing_team_policy_ref,
            controlled_scores_into_right_net=controlled_scores_into_right_net,
        )
        for k in range(n_pairs)
    ]
    return PartnerAttackersVecEnv(fns, dual_ego_training=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO · MLFootball (dual shared teammate weights)")
    p.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    p.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Vectorized rollout width (must be even): two ego rows per match",
    )
    p.add_argument("--frame_skip", type=int, default=4, help="Sim ticks per agent step")
    p.add_argument("--log_dir", type=str, default="./logs/ppo_football", help="TensorBoard log dir")
    p.add_argument("--model_dir", type=str, default="./models", help="Model save dir")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Continue training from this SB3 checkpoint (no .zip suffix, same as --watch)",
    )
    p.add_argument(
        "--save_tag",
        type=str,
        default=None,
        metavar="LABEL",
        help="Save under ppo_football_LABEL_* and models/best_model_LABEL/, checkpoints_LABEL/ "
        "(omit to keep legacy best_model/, checkpoints/).",
    )
    team = p.add_mutually_exclusive_group()
    team.add_argument(
        "--left_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=True,
        help="Train **LEFT** team: ``ATTACK_RIGHT_GOAL_*`` spawn bands, score **+x (right)** net "
        "(``controlled_scores_into_right_net=True``). Same default as omitting both flags.",
    )
    team.add_argument(
        "--right_team",
        dest="controlled_scores_into_right_net",
        action="store_const",
        const=False,
        help="Train **RIGHT** team: ``ATTACK_LEFT_GOAL_*`` spawn, score **−x (left)** net "
        "(``controlled_scores_into_right_net=False``).",
    )
    p.set_defaults(controlled_scores_into_right_net=True)
    p.add_argument(
        "--opposing_heuristic",
        action="store_true",
        help=(
            "For --left_team (default): opposing RIGHT_TEAM uses deterministic ball chase only — "
            "do not attach the learner to ``SharedPolicyRef`` (no mirrored PPO). "
            "With --right_team the opponent is already heuristic; this flag does nothing extra. "
            "Ignored when --opponent_model is set (frozen checkpoint drives the opposing pair)."
        ),
    )
    p.add_argument(
        "--opponent_model",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Frozen PPO path for the **opposing** roster (without .zip). "
            "Expected: **LEFT_TEAM** (+x) checkpoint. When training --left_team it is mirrored onto "
            "RIGHT; when training --right_team blues use native LEFT observations."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.n_envs % 2 != 0:
        raise SystemExit("--n_envs must be even (dual-ego: two rollout rows per match).")

    if args.controlled_scores_into_right_net:
        roster = "LEFT team (+x primary) [controlled_scores_into_right_net=True]"
    else:
        roster = "RIGHT team (−x primary) [controlled_scores_into_right_net=False]"
    print(f"Dual-ego training: {roster}")

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)

    model_base = f"ppo_football_2v2_{args.save_tag}" if args.save_tag else "ppo_football_2v2"
    if args.save_tag:
        best_dir = model_dir / f"best_model_{args.save_tag}"
        ckpt_dir = model_dir / f"checkpoints_{args.save_tag}"
    else:
        best_dir = model_dir / "best_model"
        ckpt_dir = model_dir / "checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opposing_policy_shared = SharedPolicyRef()
    train_env = build_partner_vec(
        args.n_envs,
        args.frame_skip,
        seed_base=0,
        opposing_team_policy_ref=opposing_policy_shared,
        controlled_scores_into_right_net=args.controlled_scores_into_right_net,
    )
    train_envs = VecMonitor(train_env)

    eval_partner = build_partner_vec(
        2,
        args.frame_skip,
        seed_base=10_000,
        opposing_team_policy_ref=opposing_policy_shared,
        controlled_scores_into_right_net=args.controlled_scores_into_right_net,
    )
    eval_env = VecMonitor(eval_partner)

    # -- callbacks ---------------------------------------------------------
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=10,
        eval_freq=max(500_000 // args.n_envs, 1),
        best_model_save_path=str(best_dir),
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1_000_000 // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=model_base,
        verbose=1,
    )
    goals_cb = GoalCountCallback()

    # -- PPO ---------------------------------------------------------------
    if args.resume:
        rp = Path(args.resume)
        base = rp.parent / rp.stem if rp.suffix.lower() == ".zip" else rp
        print(f"Resuming from {base}.zip …")
        model = PPO.load(str(base), env=train_envs, tensorboard_log=str(log_dir))
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_envs,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(log_dir),
            device="cpu"
        )

    frozen_opponent = None
    if args.opponent_model:
        opp = Path(args.opponent_model)
        ob = opp.parent / opp.stem if opp.suffix.lower() == ".zip" else opp
        print(f"Loading frozen opponent from {ob}.zip …")
        frozen_opponent = PPO.load(str(ob), device="cpu")

    if frozen_opponent is not None:
        opposing_policy_shared.model = frozen_opponent
        if args.controlled_scores_into_right_net:
            print(
                "LEFT_TEAM training: RIGHT_TEAM driven by frozen --opponent_model "
                "(mirrored +x checkpoint)."
            )
        else:
            print(
                "RIGHT_TEAM training: LEFT_TEAM driven by frozen --opponent_model "
                "(native LEFT observations)."
            )
        if args.opposing_heuristic:
            print(
                "Note: --opposing_heuristic ignored because --opponent_model is set.",
                file=sys.stderr,
            )
    elif args.controlled_scores_into_right_net and not args.opposing_heuristic:
        opposing_policy_shared.model = model
        print("LEFT_TEAM training: RIGHT_TEAM uses mirrored current policy (frozen predict).")
    elif args.controlled_scores_into_right_net:
        print("LEFT_TEAM training: RIGHT_TEAM heuristic chase (no opposing PPO; --opposing_heuristic).")
    else:
        if args.opposing_heuristic:
            print(
                "Note: --opposing_heuristic redundant with --right_team (LEFT_TEAM opposing is already heuristic).",
                file=sys.stderr,
            )
        print("RIGHT_TEAM training: LEFT_TEAM heuristic override (no mirrored policy attached).")

    print(f"Training for {args.steps:,} steps across {args.n_envs} envs …")
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([eval_cb, ckpt_cb, goals_cb]),
        progress_bar=True,
        reset_num_timesteps=bool(args.resume is None),
    )

    final_path = model_dir / f"{model_base}_final"
    model.save(str(final_path))
    print(f"Final model saved -> {final_path}.zip")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
