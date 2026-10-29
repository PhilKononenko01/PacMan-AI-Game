# Phase 1 — Acceptance report (issue #18)

**Status:** DRAFT. Training and validation measured; the held-out test evaluation waits for the knob freeze (§4).

## 1. What was built

Issues #4 to #17 implement everything in the Phase 1 PRD (v0.11) and architecture (v0.3): the fixed 5x5 grid and planning primitives, bounded map generation with hashed train/validation/test splits, the ghost, the environment with Pac-Man's partial wall memory, the referee and the one episode loop, the random and pathfinding baselines, the ADR-2 feature encoder, tabular Q-learning, versioned policy files, statistics, paired evaluation, training with learning-curve checkpoints, terminal rendering, and the five commands. The per-ticket entries in `phase-1-review.md` record what each one built and decided.

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

**Sign-off:** pending. Christian and Phillip record the freeze in `phase-1-review.md` before any test-split game is played. No test-split episode has been run while writing this draft.

## 5. Results

### 5.1 Training at the defaults (NFR-4, FR-9, FR-12)

Pending: the default 50,000-episode run is in progress. This section will record its wall time against the 10-minute target, the learning curve, and the final validation summary stored in the policy.

### 5.2 Validation (FR-10, FR-11, FR-15)

Pending: `evaluate --split val` on the trained policy, all three agents on all 1,000 validation seeds. The two baselines on the same seeds were already measured while building #10 and match the architecture's §13.3 figures exactly: random 0 wins, pathfinding 142 wins (14.2% [12.2, 16.5]) with 13.6 steps per win.

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

Pending: the policy, `config.json`, `curve.csv`, `curve.png` and `eval-val.json` of the run in §5.1 are copied from the git-ignored `runs/` into `reports/phase-1/` once the run is accepted (architecture D-6).
