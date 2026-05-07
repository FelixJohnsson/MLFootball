"""Watch trained PPO agents play MLFootball in the Pygame window.

Policies are **27-D obs / 3-D act** per ego slot; teammates share weights per roster.

Pick checkpoints per geometric team (**LEFT_TEAM** scores +x / **RIGHT_TEAM** scores −x):

* ``--model_left`` (+x-trained default layout) overrides ``--model`` for LEFT_TEAM only.
* ``--model_right`` loads a **RIGHT_TEAM** (--right_team) checkpoint with native −x observations.

If ``--model_right`` is omitted: **either** heuristic chase (**default**) **or** mirrored +x weights
via ``--opposing_team_policy`` (+ optional ``--opposing_team_model``).

**Asymmetry:** there is **no** ``--right_team_heuristic`` flag. Greens already use deterministic
chase whenever ``--model_right`` and ``--opposing_team_policy`` are both off. Blues instead
default to **PPO** from ``--model`` / ``--model_left``, so use ``--left_team_heuristic`` when you
only have a **RIGHT_TEAM** checkpoint and no LEFT save (same idea as ``train_ppo_single --watch``).

``--left_team_heuristic`` drives **blue** (LEFT) players with the same deterministic ball chase as
default greens — **no** ``--model`` / ``--model_left`` checkpoint needed. Combine with default
greens (omit ``--model_right`` and ``--opposing_team_policy``) for an all-heuristic demo.

Pass paths **without** ``.zip`` (same convention as Stable-Baselines3 ``PPO.save``).

You control a solo **PLAYER** (red): WASD, **R** resets ball, left mouse kicks.

Examples
--------
    python train/watch.py --left_team_heuristic
    python train/watch.py --model ./models/checkpoints/ppo_football_50000_steps
    python train/watch.py --model_left ./m/LEFT_final --model_right ./m/RIGHT_final
    python train/watch.py --opposing_team_policy
    python train/watch.py --opposing_team_policy --opposing_team_model ./other_ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pygame
from stable_baselines3 import PPO

from sim.actions import TickActions, idle_actions
from sim.frozen_opposing_team_policy import opposing_team_tick_actions_shared_policy
from sim.game import GameState
from render.pygame_shell import poll_pygame_tick_actions, render_game

from player import create_opponent_player
from field import FIELD_W, FIELD_H


def _load_ppo(path: str) -> PPO:
    p = Path(path)
    stem = str(p.with_suffix("")) if p.suffix.lower() == ".zip" else str(p)
    return PPO.load(stem)


def _sb3_checkpoint_exists(path_str: str) -> bool:
    p = Path(path_str)
    if p.is_file():
        return True
    return p.with_suffix(".zip").is_file()


def _clip_obs(vec: np.ndarray) -> np.ndarray:
    return np.clip(vec, -1.0, 1.0).astype(np.float32)


def _actions_to_tick(actions: np.ndarray) -> TickActions:
    a = np.asarray(actions, dtype=np.float32).reshape(-1)
    return TickActions(float(a[0]), float(a[1]), False, float(a[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch · MLFootball")
    p.add_argument(
        "--left_team_heuristic",
        action="store_true",
        help="Drive LEFT_TEAM (blues) with heuristic chase (no PPO). Use when you only trained "
        "RIGHT_TEAM and have no LEFT checkpoint — greens are already heuristic by default "
        "(no --right_team_heuristic; omit --model_right). Ignores --model / --model_left.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="./models/best_model/best_model",
        help="Default checkpoint for LEFT_TEAM if --model_left omitted (no .zip). Unused with --left_team_heuristic.",
    )
    p.add_argument(
        "--model_left",
        "--model-left",
        type=str,
        default=None,
        metavar="PATH",
        help="PPO checkpoint for LEFT_TEAM (+x scoring). Default: same as --model.",
    )
    p.add_argument(
        "--model_right",
        "--model-right",
        type=str,
        default=None,
        metavar="PATH",
        help="PPO checkpoint for RIGHT_TEAM (−x scoring); native RIGHT obs. "
        "If omitted: heuristic RIGHT_TEAM unless --opposing_team_policy.",
    )
    p.add_argument("--episodes", type=int, default=3, help="Episodes to play")
    p.add_argument("--fps", type=int, default=60, help="Render FPS cap")
    p.add_argument("--frame_skip", type=int, default=4, help="Must match training")
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="Sim dt")
    p.add_argument(
        "--opposing_team_policy",
        action="store_true",
        help="Drive RIGHT_TEAM with mirrored +x-trained PPO (--opposing_team_model or LEFT checkpoint). "
        "Ignored if --model_right is set.",
    )
    p.add_argument(
        "--opposing_team_model",
        type=str,
        default=None,
        help="Mirrored-policy checkpoint for greens (no .zip). Default: LEFT checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.left_team_heuristic and args.model_left is not None:
        print("Note: --model_left ignored because --left_team_heuristic is set.", file=sys.stderr)

    left_path = args.model_left if args.model_left is not None else args.model
    model_left: PPO | None = None

    if args.left_team_heuristic:
        print("LEFT_TEAM: heuristic chase (same scale as frozen / default greens)")
    else:
        if not _sb3_checkpoint_exists(left_path):
            zp = Path(left_path).with_suffix(".zip")
            print(
                "No LEFT_TEAM checkpoint found.\n"
                f"  Looked for: {left_path!s}  or  {zp!s}\n"
                "  Pass --model PATH or --model_left PATH to a +x-trained (--left_team) SB3 save, "
                "or use --left_team_heuristic for deterministic blues without a checkpoint.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        model_left = _load_ppo(left_path)
        print(f"LEFT_TEAM checkpoint: {left_path}")

    model_right_native: PPO | None = None
    mirror_model: PPO | None = None

    if args.model_right is not None:
        if args.opposing_team_policy:
            print("Note: --opposing_team_policy ignored because --model_right is set.", file=sys.stderr)
        if not _sb3_checkpoint_exists(args.model_right):
            zp = Path(args.model_right).with_suffix(".zip")
            print(
                f"RIGHT_TEAM checkpoint not found: {args.model_right!s}  or  {zp!s}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        model_right_native = _load_ppo(args.model_right)
        print(f"RIGHT_TEAM checkpoint (native −x obs): {args.model_right}")
    elif args.opposing_team_policy:
        if args.opposing_team_model is not None:
            mirror_ckpt = args.opposing_team_model
        elif not args.left_team_heuristic and _sb3_checkpoint_exists(left_path):
            mirror_ckpt = left_path
        else:
            print(
                "Mirrored greens need a +x checkpoint: pass --opposing_team_model PATH, "
                "or use --model / --model_left with a LEFT save.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not _sb3_checkpoint_exists(mirror_ckpt):
            zp = Path(mirror_ckpt).with_suffix(".zip")
            print(
                f"Mirrored opposing checkpoint not found: {mirror_ckpt!s}  or  {zp!s}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        mirror_model = _load_ppo(mirror_ckpt)
        print(f"RIGHT_TEAM: mirrored +x policy from {mirror_ckpt}")
    else:
        print("RIGHT_TEAM: heuristic chase (no RL)")

    pygame.init()
    screen = pygame.display.set_mode((FIELD_W, FIELD_H))
    pygame.display.set_caption(
        "MLFootball · Watch "
        "(L1,L2=blue [PPO | heuristic chase] · R1,R2=green [heuristic | native PPO | mirrored PPO] · red=solo PLAYER)"
    )
    clock = pygame.time.Clock()

    state = GameState.new()
    state.opponent = create_opponent_player()

    for ep in range(1, args.episodes + 1):
        state.reset()
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            idle = idle_actions()
            if args.left_team_heuristic:
                tick_l1 = idle
                tick_l2 = idle
            else:
                assert model_left is not None
                o_l1 = _clip_obs(state.get_obs_left_team_primary())
                o_l2 = _clip_obs(state.get_obs_left_team_partner())
                al1, _ = model_left.predict(o_l1, deterministic=True)
                al2, _ = model_left.predict(o_l2, deterministic=True)

                tick_l1 = _actions_to_tick(al1)
                tick_l2 = _actions_to_tick(al2)

            tick_r1: TickActions | None = None
            tick_r2: TickActions | None = None

            if model_right_native is not None:
                o_r1 = _clip_obs(state.get_obs_right_team_primary())
                o_r2 = _clip_obs(state.get_obs_right_team_partner())
                ar1, _ = model_right_native.predict(o_r1, deterministic=True)
                ar2, _ = model_right_native.predict(o_r2, deterministic=True)
                tick_r1 = _actions_to_tick(ar1)
                tick_r2 = _actions_to_tick(ar2)
            elif mirror_model is not None:
                tick_r1, tick_r2 = opposing_team_tick_actions_shared_policy(
                    state,
                    shared_policy_model=mirror_model,
                    deterministic=True,
                )

            for _ in range(args.frame_skip):
                events = pygame.event.get()
                keys = pygame.key.get_pressed()
                human_tick, pygame_quit = poll_pygame_tick_actions(events, keys)

                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                if pygame_quit:
                    pygame.quit()
                    sys.exit()

                si = state.step(
                    tick_l1,
                    args.dt,
                    human_actions=human_tick,
                    controlled_player2_actions=tick_l2,
                    opposing_team_player1_actions=tick_r1,
                    opposing_team_player2_actions=tick_r2,
                    controlled_team_heuristic_chase=args.left_team_heuristic,
                )
                terminated = si.terminated
                truncated = si.truncated

                render_game(screen, state)
                pygame.display.flip()
                clock.tick(args.fps)

                if terminated or truncated:
                    break

            steps += 1

        print(f"Episode {ep}: score {state.score}, steps {steps}, ticks {state.tick_count}")

    pygame.quit()


if __name__ == "__main__":
    main()
