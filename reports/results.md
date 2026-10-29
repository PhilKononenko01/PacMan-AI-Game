# Acceptance report (issue #18)

**Status:** DRAFT. Training and validation measured; the held-out test evaluation waits for the knob freeze (§4).

## 1. What was built

Issues #4 to #17 implement everything in the PRD (v0.11) and architecture (v0.3): the fixed 5x5 grid and planning primitives, bounded map generation with hashed train/validation/test splits, the ghost, the environment with Pac-Man's partial wall memory, the referee and the one episode loop, the random and pathfinding baselines, the ADR-2 feature encoder, tabular Q-learning, versioned policy files, statistics, paired evaluation, training with learning-curve checkpoints, terminal rendering, and the five commands. The per-ticket entries in `review.md` record what each one built and decided.

## 2. Runtime and dependencies (NFR-1, NFR-5)

| Item | Value |
|---|---|
| Machine | Intel Core i5-8259U @ 2.30 GHz, 16 GB, macOS 15.8.1 |
| Python | 3.14.10 (python.org build) |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 (only `pacman/plot.py` imports it, Agg backend) |
| pytest | 9.2.0 |
| Dependencies | exactly those of ADR-1: NumPy, Matplotlib; pytest for tests |

Reproducibility is within this runtime, as NFR-1 allows: the same seed, configuration and policy give the same games in one process and in a subprocess with a different `PYTHONHASHSEED` (`tests/test_determinism.py`), and a 200-episode training run gives a byte-identical table twice.

## 3. Test suite (NFR-3, NFR-7)

| Measure | Value |
|---|---|
| Tests | 472, one file per module plus `test_determinism.py` and `test_integration.py` |
| Wall time | about 12 s idle, 14.5 s with a training run competing for the CPU; budget 30 s |
| Slowest test | the 1,000-seed validation run of both baselines (about 3.4 s), kept as the regression guard for the documented figures |

Every test is headless. The §17 table of the architecture maps each file to the requirements it covers; §6 below traces the requirements the other way.

## 4. Knob freeze (PRD §5)

The configuration proposed for the final evaluation is the PRD default, unchanged by anything measured on validation:

```
game:    ghost_move_prob=1.0  wall_density=0.2  max_walls=6  min_loops=4  max_steps=100
learner: reward_win=100  reward_loss=-100  reward_corner=1  reward_step=-0.05
         learning_rate=0.1  discount=0.99  epsilon 1.0 -> 0.05 over the first half  episodes=50000
         tie_update=bootstrap
```

**Sign-off:** pending. Christian and Phillip record the freeze in `review.md` before any test-split game is played. No test-split episode has been run while writing this draft.

## 5. Results

### 5.1 Training at the defaults (NFR-4, FR-9, FR-12)

`python -m pacman train --out runs/default` at the configuration of §4, timed with `/usr/bin/time -p` (`reports/results/train-default.log`):

| Measure | Value |
|---|---|
| Wall time, whole command | **234.6 s (3 min 55 s)**; user 217.9 s, sys 2.3 s. NFR-4 target: 10 min |
| What that includes | 50,000 training episodes, 20 learning-curve checkpoints of 200 greedy validation games, the final 1,000-game validation, and writing the four files |
| Per training episode | about 4.4 ms, with the checkpoints and the final validation costing about 15 s together |
| Policy file | `reports/results/policy.npz`, 3.9 MB, format version 1, training seeds 0 to 49,999 |

**Learning curve** (`reports/results/curve.csv`, plotted in `curve.png`): the greedy win rate on validation seeds 1,000,000 to 1,000,199 every 2,500 episodes.

| Episodes | 2,500 | 5,000 | 7,500 | 10,000 | 12,500 | 15,000 | 17,500 | 20,000 | 22,500 | 25,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Win rate | 0.5% | 1.0% | 2.0% | 0.0% | 3.5% | 10.5% | 26.0% | 41.0% | 52.5% | 59.0% |

| Episodes | 27,500 | 30,000 | 32,500 | 35,000 | 37,500 | 40,000 | 42,500 | 45,000 | 47,500 | 50,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Win rate | 60.0% | 66.0% | 67.5% | 68.5% | 67.0% | 67.5% | 69.5% | 70.0% | 70.5% | 72.5% |

The curve is flat while exploration is high, rises steeply between 15,000 and 30,000 episodes as epsilon falls towards 0.05 (reached at episode 25,000), passes 50% at 22,500, and is still creeping up at 50,000. The final table is the one saved; no checkpoint was chosen by its score.

**Against the design probe.** Architecture §11.3 recorded an exploratory run of the same configuration on Python 3.9.6: 5,239 of 122,880 rows updated, 59.0% at 25,000 and 72.5% at 50,000 episodes, validation 69.1% (66.2–71.9) with 26.7% losses and 4.2% ties, 37.4 steps per win, McNemar 691/0 and 570/21, in 340 s. The production run reproduces every one of those figures to the digit, in 234.6 s on Python 3.14.10. The reviewer session read the same values independently from `policy.npz` and the time log.

### 5.2 Validation (FR-10, FR-11, FR-15)

`python -m pacman evaluate --split val --policy runs/default/policy.npz`, all three agents on all 1,000 validation seeds under the policy's configuration (9.5 s; `reports/results/eval-val.json` holds every record):

```
split val, 1000 games per agent, knobs: ghost_move_prob=1.0 wall_density=0.2 max_walls=6 min_loops=4 max_steps=100
agent         wins                  losses                ties                  steps/win
random          0   0.0% [ 0.0,  0.4]  1000 100.0% [99.6,100.0]    0   0.0% [ 0.0,  0.4]   n/a
pathfinding   142  14.2% [12.2, 16.5]   858  85.8% [83.5, 87.8]    0   0.0% [ 0.0,  0.4]   13.6
learned       691  69.1% [66.2, 71.9]   267  26.7% [24.1, 29.5]   42   4.2% [ 3.1,  5.6]   37.4
learned vs random:       b=691 c=0 one-sided exact McNemar p=0.0000
learned vs pathfinding:  b=570 c=21 one-sided exact McNemar p=0.0000
```

The exact p-values are 2^-691 (9.7 × 10^-209) against the random agent and 2.8 × 10^-140 against the pathfinding agent (`pacman.stats.mcnemar_one_sided(691, 0)` and `(570, 21)`). The baseline rows are the architecture's §13.3 figures to the digit. `check --policy` prints the same learned row from the policy's stored summary, so the saved table is the evaluated one, and reports no differing knob.

Two things worth noticing. The learned agent wins slowly: 37.4 steps per win against the pathfinding agent's 13.6, because it detours around the ghost rather than walking the shortest route into it. And it ties 4.2% of games, which the baselines never do: with the ghost adjacent it sometimes prefers to stall out the clock rather than risk a corner, a behaviour the bootstrap tie update rewards relative to a loss.

**PRD §5 "results to distrust":** no agent is at exactly 0% or 100% except the random agent at 0%, which the PRD expects; no two agents share a rate; the validation rate is measured on maps whose layouts are disjoint from the training split (`test_splits.py`); the ghost moves (`test_ghost.py`, and random loses every game); ties are reported separately from wins; and no observation carries the true map (`test_env.py`).

### 5.3 Held-out test (PRD §5 items 2 and 3)

Not run. Waits for §4.

## 6. Requirement traceability

| Requirement | Demonstrated by |
|---|---|
| FR-1 grid, whole-cell walls, blocked moves stay | `test_grid.py` (`move`), `test_env.py` (blocked move costs a step) |
| FR-2 map generation, connectivity, winnability, splits | `test_mapgen.py`, `test_splits.py` |
| FR-3 starts, the start corner visited | `test_env.py` |
| FR-4 the referee's tie on move `max_steps` | `test_referee.py`, `test_episode.py` |
| FR-5 ghost vision | `test_grid.py` (`line_of_sight`), `test_ghost.py` |
| FR-6 chase, pursue, search, move probability | `test_ghost.py` |
| FR-7 turn order, collisions, reward accounting | `test_env.py` |
| FR-8 one agent interface, one loop, no agent branching | `test_contracts.py`, `test_episode.py`, `test_baselines.py`, `test_learned.py` |
| FR-9 training saves table, configuration, seed range | `test_train.py`, `test_policy_io.py`; the run in §5.1 |
| FR-10 same seeds for every agent, Wilson intervals | `test_evaluate.py`, `test_stats.py` |
| FR-11 paired one-sided exact McNemar | `test_stats.py`, `test_evaluate.py` |
| FR-12 learning curve saved and plotted | `test_train.py`, `test_plot.py`; `curve.csv`/`curve.png` in §5.1 |
| FR-13 watch mode as pixel art with a plain fallback | `test_render.py`, `test_cli.py` |
| FR-14 any evaluated game replays | `test_cli.py` (replay from `eval-val.json`), `test_integration.py` |
| FR-15 the difficulty check never trains | `test_evaluate.py`, `test_cli.py` |
| FR-16 memory rebuilt at reset, no unsensed wall leaks | `test_env.py` |
| FR-17 simulation summaries | `test_evaluate.py`, `test_cli.py` |
| NFR-1 repeatability within the runtime | `test_rng.py`, `test_determinism.py`, `test_learned.py` |
| NFR-2 configuration in one place | `test_config.py`; every module reads its `Config` |
| NFR-3 suite under 30 s | §3 |
| NFR-4 default training under 10 min | §5.1 |
| NFR-5 dependencies | `pyproject.toml` |
| NFR-6 no layout in two splits | `test_splits.py` |
| NFR-7 the listed unit tests | the files above |

## 7. Evidence committed

`reports/results/` holds the run of §5.1 copied out of the git-ignored `runs/` (architecture D-6): `policy.npz`, `config.json`, `curve.csv`, `curve.png`, `eval-val.json` and `train-default.log` (the command's output and timing). Any game in `eval-val.json` replays with

```sh
python -m pacman watch --agent learned --seed <seed> --policy reports/results/policy.npz
```

and ends on the recorded outcome at the recorded move. Checked by hand on two records of `eval-val.json`: the learned agent's win on seed 1,000,000 (recorded 49 steps, return 100.55) replays to `Move 49/100 ... Outcome: win`, and its tie on seed 1,000,008 (recorded 100 steps) replays to `Move 100/100 ... Outcome: tie`. `tests/test_integration.py` replays every record of a small evaluation the same way.
