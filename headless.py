"""
Headless runner — no Pygame window. Prints ball position each tick.

  python headless.py --steps 120
"""

from __future__ import annotations

import argparse

from sim.actions import TickActions, idle_actions
from sim.game import GameState


def main() -> None:
    p = argparse.ArgumentParser(description="Step sim without Pygame; print ball (x,y).")
    p.add_argument("--steps", type=int, default=120, help="number of dt steps")
    p.add_argument("--dt", type=float, default=1.0 / 60.0, help="seconds per step")
    args = p.parse_args()

    state = GameState.new()
    actions: TickActions = idle_actions()

    print("step\tticks\tscore(L,R)\tterm\ttrunc\tball_x\tball_y")
    for i in range(args.steps):
        info = state.step(actions, args.dt)
        b = state.ball
        print(
            f"{i}\t{state.tick_count}\t{state.score}"
            f"\t{info.terminated}\t{info.truncated}"
            f"\t{b.x:.4f}\t{b.y:.4f}"
        )


if __name__ == "__main__":
    main()
