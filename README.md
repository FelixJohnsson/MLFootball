# MLFootball

Top-down pitch sim (Pygame demo + headless stepping + PPO training for 2-vs-2 with shared weights).

**Team rule:** spatial **LEFT** team (pitch −x half, `ATTACK_RIGHT_GOAL_*` spawn bands) always attacks the **+x (right)** goal mouth; spatial **RIGHT** team (+x half, `ATTACK_LEFT_GOAL_*`) always attacks **−x (left)**. In code this is selected by `GameState.controlled_scores_into_right_net`: `True` = RL controls **LEFT_TEAM** (scores +x); `False` = RL controls **RIGHT_TEAM** (scores −x).

## Layout

| Area               | Modules                                                                           | Depends on pygame? |
| ------------------ | --------------------------------------------------------------------------------- | ------------------ |
| **Field geometry** | `field.py`                                                                        | No                 |
| **Physics**        | `physics/` (`ball.py`, `ball_player.py`)                                          | No                 |
| **Player**         | `player.py`                                                                       | No                 |
| **Simulation**     | `sim/` (`game.py`, `actions.py`, `attacker_team_env.py`, …)                       | No                 |
| **Rendering**      | `render/` (`scene.py`, `markings.py`, `goals.py`, `colors.py`, `pygame_shell.py`) | Yes                |

Train or script against `sim` only; attach `render` when you want a window.

## RL env (training)

`sim.MLFootballEnv` / `sim.AttackerTeamRlEnv`: two RL-controlled teammates vs frozen **RIGHT_TEAM** (when RL is LEFT_TEAM) or **LEFT_TEAM** (when RL is RIGHT_TEAM); optional mirrored frozen **PPO** via `SharedPolicyRef`. **`OBS_DIM == 27`**, **`TEAM_ATTACKER_ACT_DIM == 6`** (two flattened `TickActions`: move + kick each). This is not a stock `gymnasium.Env` reset/step API; training uses `sim.partner_attackers_vec_env.PartnerAttackersVecEnv` with Stable-Baselines3. All training scripts share **`sim.reward_weights.DEFAULT_TEAM_RL_REWARD_WEIGHTS`** for `TeamRlRewardWeights` (`train/train_ppo_single.py` pulls PPO/logging from `train/single_training_constants.py` only).

```python
from sim import AttackerTeamRlEnv
from sim.frozen_opposing_team_policy import SharedPolicyRef

env = AttackerTeamRlEnv(
    frame_skip=4,
    opposing_team_policy_ref=SharedPolicyRef(),
    # True (default): LEFT team (+x scoring). False: RIGHT team (−x scoring).
    controlled_scores_into_right_net=True,
)
env.reset(seed=0)
ego0 = env.observation(0)
reward, terminated, truncated, info = env.step_team(action_vec6)
# info keys include scoring_net ("left"|"right"|None), score, tick_count, …
```

## Run

```bash
pip install -r requirements.txt
python main.py              # windowed ~60 FPS, no CLI flags
python headless.py --steps 120
```

## CLI reference

### `python train/train_ppo_single.py`

Primary ego only (teammate idle). **Rewards:** `sim/reward_weights.py` (`DEFAULT_TEAM_RL_REWARD_WEIGHTS`). **PPO + stagnant timing + paths:** `train/single_training_constants.py`. Saves default to `ppo_football_1ego_*`, `best_model_1ego/`, etc.

```bash
python train/train_ppo_single.py --steps 500000 --n_envs 8
python train/train_ppo_single.py --right_team --save_tag solo_r
```

### `python train/train_ppo.py`

PPO with parameter sharing across the two controlled teammates. **`--left_team` (default):** frozen **opposing** roster is usually the **mirrored current learner**; use **`--opposing_heuristic`** to keep them on **deterministic ball chase only** (same as `--right_team`'s built-in heuristic opponent). Use **`--opponent_model PATH`** to freeze a **LEFT_TEAM** checkpoint against the opposing pair instead (mirrored on greens when you train LEFT; **native LEFT obs** on blues when you train **`--right_team`**). **`--right_team`:** opposing **LEFT** roster is **heuristic** unless **`--opponent_model`** is set.

**Pitch “left/right team” ≠ “shoot at left/right net”.** Matches `player.py` spawn bands (`ATTACK_RIGHT_GOAL_*`, `ATTACK_LEFT_GOAL_*`):

| Colloquial     | Spawn side (approx.)                   | Scoring mouth      | Flag / ctor                                                                                      |
| -------------- | -------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------ |
| **Left team**  | Left / −x half (`ATTACK_RIGHT_GOAL_*`) | **+x (right) net** | Default if you omit flags; **`--left_team`** (explicit). `controlled_scores_into_right_net=True` |
| **Right team** | Right / +x half (`ATTACK_LEFT_GOAL_*`) | **−x (left) net**  | **`--right_team`**. `controlled_scores_into_right_net=False`                                     |

Pygame HUD colors still call the RL pair “blue” and the frozen pair “green”; naming above is geometric. **`--left_team`** and **`--right_team`** are mutually exclusive.

| Flag                | Default               | Description                                                                                        |
| ------------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| `--steps`           | `1000000`             | Total training timesteps                                                                           |
| `--n_envs`          | `8`                   | Vec width (must be **even**: two policy rows per match)                                            |
| `--frame_skip`      | `4`                   | Sim ticks per agent step (match eval / watch)                                                      |
| `--log_dir`         | `./logs/ppo_football` | TensorBoard log directory                                                                          |
| `--model_dir`       | `./models`            | Checkpoints, `best_model`, and final save root                                                     |
| `--resume`          | _(none)_              | SB3 checkpoint path **without** `.zip` (same style as `--model` in watch)                          |
| `--save_tag`        | _(none)_              | Suffix for `ppo_football_<tag>_*`, `best_model_<tag>/`, `checkpoints_<tag>/`                       |
| _(omit team flags)_ | —                     | **`LEFT`** team training (`controlled_scores_into_right_net=True`, `scoring_net=="right"` is good) |
| `--left_team`       | —                     | Same as omitting flags — explicit LEFT team (**+x** primary)                                       |
| `--right_team`      | —                     | **RIGHT** team (−x primary; `scoring_net=="left"` is good)                                         |
| `--opposing_heuristic` | —                  | Only with **`--left_team`** / default: opposing pair uses deterministic ball chase (**no** mirrored learner). Redundant with `--right_team`. **Ignored** if `--opponent_model` is set. |
| `--opponent_model`     | _(none)_           | Frozen **LEFT_TEAM** (+x) SB3 path (no `.zip`). Opponent uses mirror when training LEFT, **native LEFT obs** when training `--right_team`. |

`GameState.controlled_scores_into_right_net` mirrors that: **`True`** = controlled slots are LEFT team (score **right**).

Examples (each flag exercised at least once):

```bash
python train/train_ppo.py
python train/train_ppo.py --left_team
python train/train_ppo.py --left_team --opposing_heuristic
python train/train_ppo.py --right_team
python train/train_ppo.py --steps 2000000
python train/train_ppo.py --n_envs 4
python train/train_ppo.py --frame_skip 4
python train/train_ppo.py --log_dir ./logs/my_run
python train/train_ppo.py --model_dir ./my_models
python train/train_ppo.py --resume ./models/checkpoints/ppo_football_2v2_50000_steps
python train/train_ppo.py --save_tag exp1
python train/train_ppo.py --right_team --steps 500000 --n_envs 8 --save_tag train_right_roster
python train/train_ppo.py --right_team --opponent_model models/1ego_LEFT_TEAM_v1 --save_tag dual_r
```

Resume with the **same** team mode as when the checkpoint was trained (default / `--left_team` vs `--right_team`, and mirrored vs **`--opposing_heuristic`** opponent) so observations and dynamics stay aligned.

### `python train/watch.py`

Geometry is fixed: **LEFT_TEAM** agents (blue) vs **RIGHT_TEAM** (green) vs solo **PLAYER** (red).

| Flag                             | Default                          | Description                                                                                                                     |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--model`                        | `./models/best_model/best_model` | Fallback LEFT checkpoint (**without** `.zip`) if `--model_left` omitted                                                         |
| `--model_left`, `--model-left`   | _(use `--model`)_                | PPO checkpoint for LEFT_TEAM (**+x** net; typical default training layout)                                                      |
| `--model_right`, `--model-right` | _(none)_                         | PPO checkpoint for RIGHT_TEAM (**−x** net; `--right_team` training). Omit for heuristic greens, unless `--opposing_team_policy` |
| `--left_team_heuristic`          | off                              | Heuristic **blues** (no LEFT checkpoint). **Greens** are heuristic by default when `--model_right` is omitted — there is no `--right_team_heuristic`. |
| `--opposing_team_policy`         | off                              | RIGHT_TEAM uses **mirrored** +x policy (see `--opposing_team_model`); **ignored** if `--model_right` is set                     |
| `--opposing_team_model`          | _(none)_                         | Mirrored-policy checkpoint (defaults to LEFT checkpoint)                                                                        |
| `--episodes`                     | `3`                              | Episodes before exit                                                                                                            |
| `--fps`                          | `60`                             | Pygame FPS cap                                                                                                                  |
| `--frame_skip`                   | `4`                              | Must match training `frame_skip`                                                                                                |
| `--dt`                           | `1/60`                           | Simulation timestep (seconds)                                                                                                   |

Examples:

```bash
python train/watch.py
python train/watch.py --model ./models/checkpoints/ppo_football_2v2_50000_steps
python train/watch.py --model_left models/ppo_football_2v2_LEFT_TEAM_v1_final --model_right models/ppo_football_2v2_RIGHT_TEAM_v1_final
python train/watch.py --opposing_team_policy
python train/watch.py --opposing_team_policy --opposing_team_model ./models/other_run/best_model
```

### `python headless.py`

| Flag      | Default | Description                               |
| --------- | ------- | ----------------------------------------- |
| `--steps` | `120`   | Number of fixed-`dt` steps (idle actions) |
| `--dt`    | `1/60`  | Seconds per step                          |

Examples:

```bash
python headless.py
python headless.py --steps 500
python headless.py --dt 0.02
```

## Scoring convention

Tuple `score == (left_net_goals, right_net_goals)` — **left/right here are world-space goal lines (−x vs +x), not “left team” spawn names** (see training table).

- Ball **left net** → `score[0] += 1` (shown as Team 2 in the HUD).
- Ball **right net** → `score[1] += 1` (Team 1).

After each goal, rosters restart with `GameState._restart_after_goal()`; **`tick_count` keeps increasing** until the RL episode finishes. **`MATCH_WIN_GOALS`** (`sim.constants`, default **5**) — one episode ends (**`terminated=True`**) once **either** tally reaches this value **or** `max_ticks` is hit (**truncation**).

## Observation (`OBS_DIM == 27`)

Built in `sim/observation.py`. Normalised positions/velocities, relative ball, ball→attack-goal offsets (see `GameState._controlled_attack_goal`; primary mouth follows `controlled_scores_into_right_net`), distances over `FIELD_DIAG`, teammate and opposing-team context, cooldown fill. **Last two components** are **`score[0] / MATCH_WIN_GOALS`**, **`score[1] / MATCH_WIN_GOALS`** (same tuple as `GameState.score` / `info["score"]`: left-net tally, right-net tally).

## Policy MLP sizing (SB3 `net_arch`)

**27‑D obs, continuous 3‑D control** × two teammates (`MlpPolicy` default ~`[64,64]`) usually trains. Prefer **`policy_kwargs=dict(net_arch=[128, 128])`** once you tune rewards or lengthen episodes (multi-goal rallies add temporal credit-assignment blur). Bump the **value net** separately if critics lag: ``net_arch=dict(pi=[128, 128], vf=[256, 256])``. Go deeper (`[128,128,128]`) only if you add **more** sensory dimensions.

## Step order

`GameState.step`: increment `tick_count` → player movement → optional R ball reset → kick (if off cooldown) → ball integration/collisions → multiple `resolve_ball_player` passes → goal check (**score++**, reposition, **`terminated`** iff **`MATCH_WIN_GOALS`**) → truncation at `max_ticks`.

python train/train_ppo_single.py  --steps 3_000_000 --n_envs 16 --save_tag RIGHT_TEAM_v1.1_9_000_000 --right_team --resume models\1ego_RIGHT_TEAM_v1.1_6_000_000