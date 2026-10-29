# Issue #4 — Foundation handoff

**Date:** 2026-10-28. **Branch:** `feature/4-project-foundation`.

The foundation implements configuration, JSON serialisation, component RNGs and shared data types. It does not implement movement, map generation, the ghost, gameplay, the referee, policies, training or CLI commands. `generate`, `Environment.reset` and `Environment.step` explicitly raise `NotImplementedError` until their owning tickets are implemented.

## Public contracts

| Module | Available interface | Owner after foundation merges |
|---|---|---|
| `config.py` | Frozen `Config(game, learner)`, `GameKnobs`, `LearnerSettings`; `Config.to_json()` and `Config.from_json(text)` | Coordinate changes with the setup owner |
| `rng.py` | `derive(seed, component) -> random.Random`; `pick(rng, n) -> int` | Coordinate changes with the setup owner |
| `grid.py` | `ROWS = COLS = 5`, `Cell`, `Direction`, `DELTAS`, `Passable` | #5 |
| `mapgen.py` | Frozen `GameMap(walls)`, `Split`, `MapGenerationError`, `MAX_DRAWS`, `generate` signature | #6 |
| `env.py` | `Outcome`, `KnownCell`, frozen `Observation` and `StepResult`; `Environment` signatures | #8 |
| `episode.py` | Frozen `EpisodeRecord(seed, agent, outcome, steps, episode_return)` | #9 |
| `agents/base.py` | Runtime-checkable `Agent` protocol with `name`, `reset`, `act`, `observe` | Shared; coordinate interface changes |
| `agents/__init__.py` | Package marker only; no factory yet | #17 |

`Direction` ids are `UP=0`, `RIGHT=1`, `DOWN=2`, `LEFT=3`. `DELTAS` uses the same order. `Split` values are `train`, `val`, `test`; `Outcome` values are `win`, `loss`, `tie`. `KnownCell` values are `UNKNOWN=0`, `OPEN=1`, `WALL=2`.

`Observation(pacman, ghost, visited, known)` contains no true map or move count. `known` is a row-major tuple of 25 cells. Wall sets, visited corners and remembered cells are copied into immutable containers so later mutations by a caller cannot change an earlier snapshot.

Configuration has only `game` and `learner` sections. Missing JSON keys take defaults; unknown keys and invalid values fail with `ValueError`. Integer settings reject booleans and fractional values. Numeric settings must be finite. Settings such as `min_loops=17` remain expressible: bounded map generation in #6 owns rejection/failure of infeasible generator configurations.

## Recorded decisions and defaults

- **D-1, confirmed by Phillip:** keep `tie_update="bootstrap"` as the default. `terminal` remains supported as a configuration value. This does not change the referee's tie outcome or permit any action after the final signal.
- **D-2 through D-7:** retain the architecture's existing defaults while implementing issue #4: the `5x5:` split tag; 10,000 candidate draws; training seed `k` for episode `k`; remembered-map rendering; ignored `runs/` with selected final artifacts preserved separately; final validation performed by training. Their functionality belongs to the later tickets, not this foundation. This records retained implementation defaults, not a separate sign-off by Christian.
- **Per-game cache reset, confirmed by Phillip:** the agent resets previous state/action and cached next-state indices for every new game, while retaining learned Q values. #11 implements and tests this behavior; the protocol documents it now.
- **Explicit end signal:** the referee sends the final `Outcome` through `observe`, and the runner asks for no further actions. #9 implements it. A tie's bootstrap update is a learning choice independent of stopping action requests.
- **Reproducibility, confirmed by Phillip:** retain deterministic seed derivation and record the runtime/configuration/policy needed for replay. Exact cross-machine or cross-version training equality is not an acceptance requirement.
- **Final test, confirmed by Phillip:** freeze the policy and settings before the held-out run. Document a missed target without further tuning against the test outcomes.

The win objective, corner accounting, starts, movement rules, collision precedence, rewards and default map filters remain unchanged. The illustrative 2x2 wall idea is not an additional approved constraint.

## Parallel work and validation

After this branch is reviewed and merged, #5 (grid) and #12 (policy persistence) can proceed in separate files. #13 (statistics) can be developed independently. Each feature owner adds tests in its own file and avoids editing shared configuration or interfaces in a competing branch.

Foundation verification uses `python -m pytest` in the editable-installed virtual environment, plus `python -m pip check`. The measured foundation run passed 82 tests in 0.42 seconds using Python 3.11.14, NumPy 2.4.6, Matplotlib 3.11.2 and pytest 9.2.0. Tests include input failures, JSON round trips, immutable snapshots, the documented RNG golden value and subprocess checks with different `PYTHONHASHSEED` values. An import/round-trip smoke check also passed outside the repository, verifying the editable installation. Full-game repeatability and training performance remain #18 work.
