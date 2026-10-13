"""Watch trained PPO agents play MLFootball in the Pygame window.

Policies are **42-D obs / 3-D act** per ego slot; teammates share weights per roster.

Pick checkpoints per geometric team (**LEFT_TEAM** scores +x / **RIGHT_TEAM** scores −x):

* ``--model_left`` (+x-trained default layout) overrides ``--model`` for LEFT_TEAM only.
* ``--model_right`` loads a **RIGHT_TEAM** (--right_team) checkpoint with native −x observations.

If ``--model_right`` is omitted: **either** geometric shooting heuristic (**default**) **or** mirrored +x weights
via ``--opposing_team_policy`` (+ optional ``--opposing_team_model``).

**Asymmetry:** there is **no** ``--right_team_heuristic`` flag. Greens already use deterministic
chase whenever ``--model_right`` and ``--opposing_team_policy`` are both off. Blues instead
default to **PPO** from ``--model`` / ``--model_left``, so use ``--left_team_heuristic`` when you
only have a **RIGHT_TEAM** checkpoint and no LEFT save (same idea as ``train_ppo_single --watch``).

``--left_team_heuristic`` drives **blue** (LEFT) players with the same deterministic ball chase as
default greens — **no** ``--model`` / ``--model_left`` checkpoint needed. Combine with default
greens (omit ``--model_right`` and ``--opposing_team_policy``) for an all-heuristic demo.

Pass paths **without** ``.zip`` (same convention as Stable-Baselines3 ``PPO.save``).

By default you get a fifth **solo PLAYER** (red) for demo control (WASD, **R**, click kick).
Training envs never include this pawn — collisions differ. Use ``--no_human_pawn`` to match training (4 outfield players only).

Training-time differences to know about:
Use ``--no_human_pawn`` (no 5th collision body). For **train_ppo_single** checkpoints, use
``--single_ego_partner_idle`` — that script only drives the **primary** teammate; the partner
is zero-action, but default watch runs PPO on **both** blues (or both greens with two native models).

Examples
--------
    python train/watch.py --left_team_heuristic
    python train/watch.py --model ./models/checkpoints/ppo_football_50000_steps
    python train/watch.py --model_left ./m/LEFT_final --model_right ./m/RIGHT_final
    python train/watch.py --opposing_team_policy
    python train/watch.py --opposing_team_policy --opposing_team_model ./other_ckpt
    python train/watch.py --model_left ./LEFT --model_right ./RIGHT --no_human_pawn --stochastic
    python train/watch.py --model_left ./models/1ego_LEFT_final --single_ego_partner_idle --no_human_pawn
    python train/watch.py --model_left ./models/best_model_LEFT_TEAM_v1.2_against_model/best_model --no_human_pawn --gif docs/demo.gif
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


# Jersey and pitch colors are a small fraction of the pixels. A palette built only
# from pixel counts drops them, so blue players quantize into the grass.
_GIF_RESERVED_COLORS: tuple[tuple[int, int, int], ...] = (
    (58, 134, 64),
    (48, 116, 54),
    (248, 252, 255),
    (168, 186, 198),
    (28, 78, 36),
    (110, 220, 255),
    (255, 210, 40),
    (42, 98, 180),
    (36, 86, 170),
    (230, 210, 80),
    (210, 200, 80),
    (48, 128, 86),
    (32, 112, 74),
    (200, 235, 210),
    (190, 228, 205),
    (250, 250, 252),
    (30, 32, 38),
    (255, 255, 250),
    (18, 22, 28),
    (10, 18, 24),
    (182, 58, 52),
    (255, 226, 60),
)


def _frame_image(screen: pygame.Surface, width: int):
    from PIL import Image

    w, h = screen.get_size()
    image = Image.frombytes("RGB", (w, h), pygame.image.tobytes(screen, "RGB"))
    if width != w:
        height = max(1, round(h * width / w))
        image = image.resize((width, height), Image.Resampling.BOX)
    return image


def _gif_palette(frames: list):
    from PIL import Image

    reserved = list(_GIF_RESERVED_COLORS)
    sample = frames[len(frames) // 2]
    n_adapt = 256 - len(reserved)
    adapted = sample.quantize(colors=n_adapt, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    raw = adapted.getpalette()
    extra = [(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]) for i in range(n_adapt)]
    pal: list[int] = []
    for r, g, b in reserved + extra:
        pal.extend((r, g, b))
    image = Image.new("P", (1, 1))
    image.putpalette(pal)
    return image


def _write_gif(frames: list, path: Path, duration_ms: int) -> Path:
    from PIL import Image

    palette = _gif_palette(frames)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


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
    p.add_argument(
        "--no_human_pawn",
        action="store_true",
        help="Do not spawn the solo red PLAYER (training matches 4 outfield agents only).",
    )
    p.add_argument(
        "--single_ego_partner_idle",
        action="store_true",
        help=(
            "Match ``train_ppo_single`` / ``SingleEgoGymEnv``: only the **primary** roster "
            "slot (L1 or R1) gets PPO actions from a native checkpoint; the partner gets idle "
            "stick/kick. Omit for ``train_ppo.py`` / ``train_ppo_2v2`` dual-ego saves where both "
            "teammates are controlled."
        ),
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions like PPO rollout collection (deterministic=False). "
        "Default matches EvalCallback greedy actions.",
    )
    p.add_argument(
        "--gif",
        type=str,
        default=None,
        metavar="PATH",
        help="Save a short GIF of the rendered match to PATH and exit. Requires Pillow.",
    )
    p.add_argument("--gif-seconds", type=float, default=12.0, help="Simulation seconds to record")
    p.add_argument("--gif-fps", type=int, default=12, help="GIF playback frame rate")
    p.add_argument("--gif-width", type=int, default=640, help="GIF width in pixels")
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
        print("RIGHT_TEAM: geometric shooting heuristic (no RL)")

    if args.single_ego_partner_idle:
        print(
            "single_ego_partner_idle: native PPO controls only L1 / R1; partners idle "
            "(matches train_ppo_single)."
        )

    gif_frames: list = []
    gif_stride = 1
    gif_limit = 0
    if args.gif:
        try:
            import PIL  # noqa: F401
        except ImportError as exc:
            raise SystemExit("GIF export needs Pillow: pip install pillow") from exc
        if args.gif_fps < 1 or args.gif_seconds <= 0 or args.gif_width < 2:
            raise SystemExit("--gif-fps must be >= 1, --gif-seconds > 0, and --gif-width >= 2")
        gif_stride = max(1, int(round(1.0 / (args.dt * args.gif_fps))))
        gif_limit = max(1, int(round(args.gif_seconds * args.gif_fps)))

    pygame.init()
    display_flags = pygame.HIDDEN if args.gif else 0
    screen = pygame.display.set_mode((FIELD_W, FIELD_H), display_flags)
    pygame.display.set_caption(
        "MLFootball · Watch "
        "(L1,L2=blue [PPO | heuristic chase] · R1,R2=green [heuristic | native PPO | mirrored PPO] · red=solo PLAYER)"
    )
    clock = pygame.time.Clock()

    state = GameState.new()
    if args.no_human_pawn:
        state.opponent = None
    else:
        state.opponent = create_opponent_player()

    record_done = False
    gif_score = (0, 0)
    for ep in range(1, args.episodes + 1):
        state.reset()
        terminated = False
        truncated = False
        steps = 0
        rendered = 0

        while not (terminated or truncated):
            idle = idle_actions()
            if args.left_team_heuristic:
                tick_l1 = idle
                tick_l2 = idle
            else:
                assert model_left is not None
                o_l1 = _clip_obs(state.get_obs_left_team_primary())
                o_l2 = _clip_obs(state.get_obs_left_team_partner())
                det = not args.stochastic
                al1, _ = model_left.predict(o_l1, deterministic=det)
                al2, _ = model_left.predict(o_l2, deterministic=det)

                tick_l1 = _actions_to_tick(al1)
                tick_l2 = _actions_to_tick(al2)
                if args.single_ego_partner_idle:
                    tick_l2 = idle

            tick_r1: TickActions | None = None
            tick_r2: TickActions | None = None

            if model_right_native is not None:
                o_r1 = _clip_obs(state.get_obs_right_team_primary())
                o_r2 = _clip_obs(state.get_obs_right_team_partner())
                det_r = not args.stochastic
                ar1, _ = model_right_native.predict(o_r1, deterministic=det_r)
                ar2, _ = model_right_native.predict(o_r2, deterministic=det_r)
                tick_r1 = _actions_to_tick(ar1)
                tick_r2 = _actions_to_tick(ar2)
                if args.single_ego_partner_idle:
                    tick_r2 = idle
            elif mirror_model is not None:
                tick_r1, tick_r2 = opposing_team_tick_actions_shared_policy(
                    state,
                    shared_policy_model=mirror_model,
                    deterministic=not args.stochastic,
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

                human_for_step = human_tick if state.opponent is not None else None
                si = state.step(
                    tick_l1,
                    args.dt,
                    human_actions=human_for_step,
                    controlled_player2_actions=tick_l2,
                    opposing_team_player1_actions=tick_r1,
                    opposing_team_player2_actions=tick_r2,
                    controlled_team_heuristic_chase=args.left_team_heuristic,
                )
                terminated = si.terminated
                truncated = si.truncated

                render_game(screen, state)
                if not args.gif:
                    pygame.display.flip()
                    clock.tick(args.fps)
                if args.gif and rendered % gif_stride == 0:
                    gif_frames.append(_frame_image(screen, args.gif_width))
                    if state.score != gif_score:
                        gif_score = state.score
                        print(f"score {state.score} at gif frame {len(gif_frames)}")
                    if len(gif_frames) >= gif_limit:
                        record_done = True
                rendered += 1

                if record_done or terminated or truncated:
                    break

            steps += 1
            if record_done:
                break

        print(f"Episode {ep}: score {state.score}, steps {steps}, ticks {state.tick_count}")
        if record_done:
            break

    if args.gif:
        if not gif_frames:
            pygame.quit()
            raise SystemExit("No frames captured for GIF")
        out = _write_gif(gif_frames, Path(args.gif), duration_ms=max(1, int(round(1000 / args.gif_fps))))
        print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(gif_frames)} frames)")

    pygame.quit()


if __name__ == "__main__":
    main()
