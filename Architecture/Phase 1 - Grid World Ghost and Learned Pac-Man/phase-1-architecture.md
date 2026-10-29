# Phase 1 — Grid World, Ghost and Learned Pac-Man: Architecture

**Status:** DRAFT **v0.3**. Implements PRD v0.11. Issue #4 implements configuration, RNG and shared types; gameplay modules remain planned. D-1 is confirmed as bootstrap. D-2 through D-7 retain their existing implementation defaults; see `phase-1-foundation-handoff.md` for scope and ownership.
**Date:** 2026-10-28
**Design baseline:** `PhilKononenko01/PacMan-AI-Game` commit `90c6323`, which holds PRD v0.10, architecture v0.2 and the review log before issue #4. The foundation handoff identifies implemented configuration, RNG and shared type contracts; all other modules and commands below remain planned. Game/training measurements come from a discarded exploratory probe (Python 3.9.6 with NumPy 2.0.2; see `phase-1-review.md`) and must be checked against the implementation.
**Authority:** the PRD. This document decides only what PRD §7 leaves to it, plus the module, type and command structure needed to build the PRD. Where it seems to disagree with the PRD, the PRD wins and this document is wrong.
**Form.** This document states what is decided and what is to be built. Review findings and version history go in `phase-1-review.md`.

---

## 1. PRD §7 items and where they are settled

| PRD §7 item | Settled in |
|---|---|
| Exact bytes hashed with SHA-256 for the split | §6.3 |
| How each component's generator is derived from the episode seed; generator algorithm and version pin; no dependence on string hashing or set order | §5 |
| Which training seeds the training episodes use | §11.1 |
| Saved-policy file format, including the last validation results | §12 |
| The draw limit after which map generation fails | §6.4 |
| Whether the Q-learning update on a tie step is terminal or bootstraps (bootstrap confirmed in issue #4) | §10.5, §16 |
| Measured sweep of each game knob's effect on both baselines | §15 |
| Command-line entry points | §14 |

## 2. Runtime and dependencies (ADR-1, NFR-5)

- **Python 3.11 or later.** The team laptop that ran the probe now has Python 3.14.10 from python.org, on which NumPy 2.5.4, Matplotlib 3.11.2 and pytest 9.2.0 install, and on which the §5.1 and §6.3 golden values hold. Each team member installs 3.11+ before building. Nothing below uses a feature newer than 3.11.
- **NumPy** for the Q-table only. No NumPy randomness is used anywhere (§5).
- **Matplotlib** for the learning-curve plot only. It is imported inside `pacman/plot.py` and nowhere else, with the `Agg` backend, so the game, the tests and the terminal views never load it.
- **pytest** for tests.
- Everything else is the standard library: `random`, `hashlib`, `json`, `argparse`, `math`, `fractions`, `dataclasses`, `enum`, `collections`, `time`, `os`, `sys`, `shutil`.
- `pyproject.toml` declares `requires-python = ">=3.11"`, the runtime dependencies `numpy` and `matplotlib`, and `pytest` as a development dependency. No other dependency may be added without team agreement.

## 3. File inventory

```
pyproject.toml
.gitignore           # runs/ (§16 D-6)
pacman/
  __init__.py
  __main__.py        # python -m pacman  ->  cli.main()
  cli.py             # argument parsing and the five commands (§14)
  config.py          # Config and its parts, defaults, validation, JSON (§4.3)
  rng.py             # derive(), pick() (§5)
  grid.py            # Cell, Direction, neighbours, move, bfs, first_step, line_of_sight (§9)
  mapgen.py          # GameMap, loop count, split bucket, generate(), MapGenerationError (§6)
  ghost.py           # Ghost (§8)
  env.py             # Environment, Observation, StepResult, Outcome, KnownCell (§7)
  referee.py         # Referee: move count, max_steps, the tie call (§7.4)
  agents/
    __init__.py      # make_agent(name, ...) factory used by the CLI only
    base.py          # Agent protocol (§10.1)
    random_agent.py  # RandomAgent
    pathfinding.py   # PathfindingAgent, plan() shared with features
    features.py      # state index from an observation (§10.4)
    learned.py       # LearnedAgent: Q-table, epsilon-greedy, update (§10.5)
  episode.py         # play_episode(), EpisodeRecord (§10.6)
  train.py           # training loop and learning curve (§11)
  policy_io.py       # save_policy(), load_policy() (§12)
  stats.py           # wilson(), mcnemar_one_sided() (§13.2)
  evaluate.py        # run_split(), summarise(), the evaluation report (§13)
  render.py          # pixel-art and plain-text frames (§14.6)
  plot.py            # learning-curve PNG (only Matplotlib importer)
tests/
  (one file per module, §17)
runs/                # outputs; git-ignored (§16)
```

The environment, the ghost, the map generator and the agents never import the CLI, the renderer, the plot module or each other's internals. `agents/` sees the game only through `Observation`. `env.py` imports `mapgen`, `ghost`, `grid` and `config`, and never imports `agents` or `referee`. Only `episode.py` joins the environment, the referee and an agent (FR-8).

## 4. Core types

All types are immutable unless the row says otherwise. Coordinates are cells, not pixels. Time is in steps unless a unit is given.

### 4.1 Game types

| Type | Definition | Notes |
|---|---|---|
| `Cell` | `tuple[int, int]`, `(row, col)` | Row 0 at the top, col 0 at the left (PRD §2.1). The grid is fixed: `grid.ROWS = grid.COLS = 5` are constants, not configuration (PRD OQ-15). |
| `Direction` | `IntEnum`: `UP=0`, `RIGHT=1`, `DOWN=2`, `LEFT=3` | The integer values are the action ids and the fixed tie-break order (PRD §2.5). Deltas `(-1,0)`, `(0,1)`, `(1,0)`, `(0,-1)`. |
| `Outcome` | `Enum`: `WIN`, `LOSS`, `TIE` | |
| `KnownCell` | `IntEnum`: `UNKNOWN=0`, `OPEN=1`, `WALL=2` | Pac-Man's memory of one cell. |
| `GameMap` | frozen dataclass: `walls: frozenset[Cell]` | The true 5x5 map. Never placed in an observation. |
| `Observation` | frozen dataclass: `pacman: Cell`, `ghost: Cell`, `visited: frozenset[Cell]`, `known: tuple[KnownCell, ...]` | `known` is row-major, length 25. It is a snapshot: later steps never change an observation already handed out. Exactly the fields of PRD §2.8. |
| `StepResult` | frozen dataclass: `observation: Observation`, `reward: float`, `outcome: Outcome \| None` | From the environment, `outcome` is `WIN`, `LOSS` or `None`; the environment never returns `TIE`. The referee completes it (§7.4). |
| `EpisodeRecord` | frozen dataclass: `seed: int`, `agent: str`, `outcome: Outcome`, `steps: int`, `episode_return: float` | One per played episode. `steps` is the number of the step the episode ended on. |

### 4.2 Splits

| Split | `Split` value | Hash buckets | First seed | Seeds |
|---|---|---|---|---|
| Training | `TRAIN` | 0 to 79 | 0 | 1,000,000 |
| Validation | `VAL` | 80 to 89 | 1,000,000 | 1,000 |
| Test | `TEST` | 90 to 99 | 2,000,000 | 1,000 |

`Split` is an `Enum` in `mapgen.py`. These numbers are the PRD §2.7 table and are constants of `mapgen.py`, not configuration: they define what the splits are. `split_of_seed(seed)` returns the split whose seed range holds `seed` and raises `ValueError` for any other seed, so a seed alone names its split everywhere a command takes one (§14).

### 4.3 Configuration (PRD §2.7, NFR-2)

`Config` is a frozen dataclass with two parts, `game` and `learner`. It is the only place in the code where any game knob or learner setting has a value; every other module reads it from the `Config` it is given (NFR-2).

**Game knobs** (`GameKnobs`):

| Field | Type | Default | Valid | Unit |
|---|---|---|---|---|
| `ghost_move_prob` | float | 1.0 | 0.0 to 1.0 | probability per step |
| `wall_density` | float | 0.20 | 0.0 to 1.0 | probability per non-corner cell |
| `max_walls` | int | 6 | 0 or more | walls |
| `min_loops` | int | 4 | 0 or more | independent loops |
| `max_steps` | int | 100 | 1 or more | moves; read only by the referee (§7.4) |

**Learner settings** (`LearnerSettings`):

| Field | Type | Default | Valid | Unit |
|---|---|---|---|---|
| `reward_win` | float | 100.0 | any finite number | reward |
| `reward_loss` | float | -100.0 | any finite number | reward |
| `reward_corner` | float | 1.0 | any finite number | reward per first corner visit |
| `reward_step` | float | -0.05 | any finite number | reward per step |
| `learning_rate` | float | 0.1 | above 0, at most 1 | |
| `discount` | float | 0.99 | 0 to 1 | |
| `epsilon_start` | float | 1.0 | 0 to 1 | |
| `epsilon_end` | float | 0.05 | 0 to `epsilon_start` | |
| `epsilon_decay_fraction` | float | 0.5 | above 0, at most 1 | fraction of training episodes |
| `episodes` | int | 50,000 | 1 to 1,000,000 | episodes |
| `tie_update` | str | `"bootstrap"` | `"bootstrap"` or `"terminal"` | §10.5 |

`tie_update` is a learner setting: it changes how the agent learns, not how hard the game is.

**Serialisation.** `Config.to_json()` writes a JSON object with the keys `game` and `learner`, each holding its fields by name, and `Config.from_json()` reads it back. Missing keys take their defaults, unknown keys are an error, and every field is checked against its valid range when a `Config` is built. A JSON file is the `--config` input of every command, and `--set name=value` overrides one field by its bare name (§14).

## 5. Randomness (ADR-3, NFR-1)

### 5.1 One generator per component, derived from the episode seed

```
derive(seed: int, component: str) -> random.Random
    digest = sha256(f"{seed}:{component}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
```

| Component name | Used by | Draws per step |
|---|---|---|
| `"map"` | `mapgen.generate` | none during play; 21 per candidate map, one per non-corner cell (§6.1) |
| `"ghost"` | `Ghost` | one move draw every step, plus one tie draw when searching with tied candidates (§8.3) |
| `"agent"` | the agent's `reset` (random agent's moves, learned agent's exploration) | random agent: one; learned agent in training: one or two (§10.5); otherwise none |

Each component's stream is independent of every other's, so for one seed the map and the ghost's draws are the same whichever agent plays (the pairing that FR-10 and FR-11 rely on).

### 5.2 Only `random()` is called

Python guarantees that `random.Random(n).random()` produces the same sequence for the same integer seed across Python versions. It does not guarantee this for `randrange`, `choice`, `shuffle` or the other methods. The code therefore calls only `rng.random()`, and turns it into an index with one helper:

```
pick(rng, n) -> int        # uniform index in 0 .. n-1
    return min(int(rng.random() * n), n - 1)
```

Choices among candidates use `pick` over the candidates listed in direction order. Probability tests use `rng.random() < p`. The encoding and RNG construction are fixed and protected by golden-value tests. Reported runs record the supported runtime and dependency versions under NFR-1; exact cross-machine or cross-version training equality is not required. NumPy's random module is not used.

### 5.3 Rules (NFR-1)

- No call to the built-in `hash()`, no `PYTHONHASHSEED` dependence, no global `random` functions, no `numpy.random`, no time or process-id seeding.
- No decision depends on the iteration order of a `set` or `dict` keyed by cells. Every choice iterates cells in row-major order or neighbours in direction order. Sets are used for membership tests only.
- Floating-point work uses Python floats and `float64` NumPy element operations. Verify repeatability within the recorded supported runtime; cross-machine bit-for-bit equality is not an acceptance requirement.

A test plays the same seeds twice in one process and once in a fresh subprocess with a different `PYTHONHASHSEED`, and asserts identical episode records (§17).

## 6. Map generation (PRD §2.3, FR-2)

### 6.1 Drawing a candidate

Non-corner cells are visited in row-major order. For each one, one `rng.random()` is drawn from the `"map"` generator, and the cell is a wall if the draw is below `wall_density`. Every non-corner cell consumes exactly one draw per candidate, whatever its outcome, so the stream position after a candidate never depends on its walls.

### 6.2 Accepting a candidate

The checks run in this order, and the first failure rejects the candidate:

1. **Connected:** a breadth-first search from `(0, 0)` over open cells reaches every open cell.
2. **Wall cap:** the number of walls is at most `max_walls`.
3. **Loops:** the number of edges between orthogonally adjacent open cells, minus the number of open cells, plus one, is at least `min_loops`. Edges are counted once, by looking only right and down from each open cell.
4. **Split:** `split_bucket(map)` lies in the requested split's buckets (§4.2).

A rejected candidate is followed by the next candidate from the same stream. The order changes no result, because a map is accepted only if all four pass, but it fixes how much work a rejection costs.

### 6.3 Split bucket: the hashed bytes

```
layout_bytes(map) = b"5x5:" + bytes of the row-major wall string
                    where each cell is b"1" for a wall and b"0" for open
split_bucket(map) = int.from_bytes(sha256(layout_bytes(map)).digest(), "big") % 100
```

The `5x5:` prefix is a format tag. The grid never changes size (PRD OQ-15), so the prefix carries no information, but it is kept because the splits measured in this document were computed with it; removing it would move every map to a new split (D-2). These values are fixed by tests (§17) and never change:

| Map | `layout_bytes` | SHA-256 | Bucket |
|---|---|---|---|
| Empty 5x5; also the map of test seed 2,000,000 | `5x5:0000000000000000000000000` | `b7bce985c39409cc0504aa9dd356c00f862dd5e048e4c7df57156e90d2fa3ba4` | 96 (test) |
| Walls at `(1, 2)`, `(1, 3)`; the map of validation seed 1,000,000 | `5x5:0000000110000000000000000` | `3d51ab9a4b367db5950aa90bcf713327b097ed2b46493f6f909d69fbbebef908` | 84 (validation) |
| Walls at `(3, 4)`, `(4, 1)`; the map of training seed 0 | `5x5:0000000000000000000101000` | `dafd13479351b71d98d6ae1a373db43ff366d020b157b886ed5c314d1a0eb1bc` | 28 (training) |

The seed-to-map rows also fix §5.1's derivation and §6.1's draw order: a change to either changes these maps.

At the default knobs the empty map is about 1.25% of accepted maps and falls in the test split, the residual that PRD §2.7 and round 1 accept.

### 6.4 Draw limit

`generate(seed, split, config, max_draws=MAX_DRAWS)` tries at most `MAX_DRAWS = 10_000` candidates, then raises `MapGenerationError` with the seed, the split, the generator knobs and the number of draws. `MAX_DRAWS` is a constant of `mapgen.py`, not a knob: it never changes which map a seed produces, only when the generator gives up. The `max_draws` argument exists so a test can reach the error quickly.

Measured at the default knobs over 1,000 seeds of each split:

| Split | Mean draws | Most draws | Distinct layouts in 1,000 maps |
|---|---|---|---|
| Training | 1.81 | 10 | 829 |
| Validation | 15.35 | 167 | 579 |
| Test | 12.96 | 75 | 491 |

The limit is 60 times the worst case seen. A configuration that accepts nothing (`min_loops` 17) fails in 0.84 s. Generating all 1,000 validation maps takes 1.6 s.

## 7. Environment (PRD §2.4, §2.6, FR-3, FR-4, FR-7, FR-16)

`Environment(config)` holds one episode at a time. It is mutable and is the only owner of the true map, the ghost and Pac-Man's memory.

### 7.1 Reset

`reset(seed, game_map=None) -> Observation`:

1. `split = split_of_seed(seed)` and `map = generate(seed, split, config)`. A test may pass a hand-built `game_map` instead, which skips generation; no command passes one.
2. Pac-Man at `(0, 0)`, the ghost at `(4, 4)`, `visited = {(0, 0)}` (FR-3). The environment holds no move count.
3. Memory: every cell `UNKNOWN`.
4. The ghost is built with `derive(seed, "ghost")` (§8).
5. Pac-Man senses (§7.3), and the observation is returned.

### 7.2 Step

`step(action: Direction, t: int) -> StepResult` runs PRD §2.4 steps 1 to 6. `t` is the move number, given by the referee (§7.4); the environment uses it only to stamp the ghost's visit record and never compares it with anything.

| PRD §2.4 | Code |
|---|---|
| — | `reward = reward_step` |
| 1 | `pacman = move(pacman, action, map)` |
| 2 | if `pacman == ghost.cell`: `reward += reward_loss`; end `LOSS` |
| 3 | if `pacman` is a corner not in `visited`: add it, `reward += reward_corner`; if all four are visited: `reward += reward_win`; end `WIN` |
| 4 | sense (§7.3) |
| 5 | `ghost.act(pacman, map, t)` (§8) |
| 6 | if `ghost.cell == pacman`: `reward += reward_loss`; end `LOSS` |
| 7 | not here: the referee (§7.4) |

This is the PRD §2.6 accounting: every step pays `reward_step`, a first corner visit pays `reward_corner`, the ending step pays the win or loss reward, and a loss at row 2 never reaches row 3. Calling `step` after a win or loss raises an error.

### 7.3 Sensing and memory

Sensing sets each in-grid neighbour of Pac-Man's cell to `OPEN` or `WALL` from the true map. Nothing else ever writes memory, and memory is rebuilt at every reset (FR-16). Pac-Man's own start cell stays `UNKNOWN` until a later move senses it. Planning treats unknown cells as open, and corners are never walls, so this changes no decision.

The observation is built from the memory, never from the map: `known` is copied into a new tuple each step.

### 7.4 Referee (PRD §2.4 step 7, OQ-14)

`Referee(max_steps)` is the only holder of the move count and the only reader of `max_steps`. It knows nothing else about the game.

| Field | Type | Initial value |
|---|---|---|
| `max_steps` | int | from `config.game.max_steps` |
| `moves` | int | 0 at each episode start |

- `start()` sets `moves = 0`.
- `next_move() -> int` adds 1 to `moves` and returns it. The episode loop passes it to `Environment.step`.
- `rule(result) -> Outcome | None`: if `result.outcome` is `WIN` or `LOSS`, returns it unchanged; a win or loss on move `max_steps` stays a win or loss, as in PRD §2.4. Otherwise, if `moves == max_steps`, returns `TIE`: the referee stops the episode. Otherwise returns `None`.

The referee's ruling is what the agent receives (§10.1) and what the record stores. After a ruling other than `None`, the episode loop asks the agent for no further move.

## 8. Ghost (PRD §2.5, FR-5, FR-6)

### 8.1 State

| Field | Type | Initial value |
|---|---|---|
| `cell` | `Cell` | `(4, 4)` |
| `last_seen` | `Cell \| None` | `None` |
| `last_visit` | `dict[Cell, int]`, step of the most recent visit | `{start cell: 0}` |
| `rng` | `random.Random` from `derive(seed, "ghost")` | |

### 8.2 One ghost move, `act(pacman, map, t)`

1. **Vision.** If `line_of_sight(cell, pacman, open on the true map)`, set `last_seen = pacman`.
2. **Move draw.** `u = rng.random()`. This draw happens every step. If not `u < ghost_move_prob`, the ghost stays put this step and nothing else changes.
3. **Mode and move:**

| Mode | Condition | Move |
|---|---|---|
| Chase | Step 1 saw Pac-Man (so `last_seen` is Pac-Man's current cell) | `first_step(cell, last_seen)` on the true map |
| Pursue | Step 1 did not see Pac-Man, `last_seen` is set | `first_step(cell, last_seen)` on the true map |
| Search | `last_seen` is `None` | Least recently visited open neighbour (below) |

4. If `cell == last_seen` after the move, set `last_seen = None` (PRD §2.5, "reaching it clears the record").
5. `last_visit[cell] = t`, in every mode.

Chase and pursue use the same code path; they differ only in whether `last_seen` was refreshed this step. The mode is not stored.

**Search.** The candidates are the in-grid open neighbours, in direction order. Each candidate's key is `last_visit.get(cell, -1)`, so a never-visited cell is less recent than any visited one. The candidates with the smallest key are the ties. One tie is taken directly. Two or more are chosen by `pick(rng, number of ties)` over the ties in direction order. An accepted map is connected and has at least two open cells, so a candidate always exists.

### 8.3 Ghost draw order per step

1. One move draw, every step.
2. One tie draw, only on a step when the ghost moves in search mode and two or more candidates tie.

No other draw is taken from the ghost's generator.

## 9. Planning primitives (`grid.py`)

All take a `passable(cell) -> bool` predicate, so the ghost (true map) and Pac-Man's side (memory, unknown counts as open) share one implementation.

- `neighbours(cell)`: the in-grid neighbours in direction order.
- `move(cell, direction, passable)`: the neighbour, or `cell` itself if it is off the grid or not passable (FR-1).
- `bfs(source, passable)`: distances from `source` to every reachable cell.
- `first_step(source, target, passable) -> Direction`: with `d = bfs(target, passable)`, the first direction in `UP, RIGHT, DOWN, LEFT` whose neighbour `n` is passable and has `d[n] == d[source] - 1`. This is the PRD's "first step of a shortest path" with ties broken by direction order. It is only called with `source != target`.
- `line_of_sight(a, b, passable)`: true when `a` and `b` share a row or a column and every cell strictly between them is passable. Adjacent cells have nothing between them, so they always see each other.

**Pac-Man's passable predicate** is `known[cell] != WALL`. **The ghost's** is `cell not in map.walls`.

**No "no path" case.** On the true map every open cell is reachable (§6.2). On Pac-Man's memory, every known wall is a real wall and every unknown cell counts as open, so the true path is always available. `first_step` and `bfs` may therefore assume a path exists, and a test asserts it on every map of a sample.

## 10. Agents (PRD §2.8, ADR-2, FR-8)

### 10.1 Interface

```
class Agent(Protocol):
    name: str
    def reset(self, rng: random.Random) -> None
    def act(self, observation: Observation) -> Direction
    def observe(self, reward: float, observation: Observation, outcome: Outcome | None) -> None
```

`outcome` is the referee's ruling (§7.4): `None` while the episode goes on, otherwise `WIN`, `LOSS` or `TIE` (PRD §2.8, OQ-14). The episode loop (§10.6) is the only caller. The environment never sees an agent.

`reset` clears previous state/action and cached next-state indices on every new game, while retaining learned Q values. The final `observe` delivers an explicit end signal; the runner never asks for another action after any final outcome. The configured tie update independently controls whether that last learning update bootstraps.

### 10.2 Random agent (`"random"`)

`act` returns `Direction(pick(rng, 4))`. `observe` does nothing.

### 10.3 Pathfinding agent (`"pathfinding"`)

`plan(observation)`, shared with the features:

1. `dist = bfs(pacman, Pac-Man's passable)`.
2. Unvisited corners, in the fixed order `(0, 4)`, `(4, 0)`, `(4, 4)`. The target is the first with the smallest `dist`.
3. Returns the target, `first_step(pacman, target, Pac-Man's passable)`, `dist`, and the number of unvisited corners.

`act` returns the first step. The plan is rebuilt from the latest memory every step, which is the PRD's replanning on a newly sensed wall. `observe` does nothing. It never looks at the ghost.

### 10.4 Learned agent's state index (ADR-2)

From one observation, with `plan` as above:

| Feature | Value | Range |
|---|---|---|
| `step` | `int(first step to the nearest unvisited corner)` | 0 to 3 |
| `open_mask` | bit `i` set when the neighbour in direction `i` is in the grid and `known == OPEN` | 0 to 15 |
| `offset` | `k = (gr - pr + 4) * 9 + (gc - pc + 4)`, then `k - 1` if `k` is above the centre, 40 | 0 to 79 |
| `distance` | `min(dist[ghost], 4) - 1` | 0 to 3 |
| `seen` | `1` if `line_of_sight(ghost, pacman, Pac-Man's passable)` else `0` | 0 to 1 |
| `remaining` | number of unvisited corners minus 1 | 0 to 2 |

`index = ((((step * 16 + open_mask) * 80 + offset) * 4 + distance) * 2 + seen) * 3 + remaining`, so the table has 122,880 rows (PRD ADR-2). The ghost's distance is at least 1, because an observation with Pac-Man on the ghost's cell ends the episode and is never indexed.

### 10.5 Learned agent (`"learned"`)

- **Table.** A `float64` NumPy array of shape `(122880, 4)`, column `i` for direction `i`, initialised to 0. About 3.9 MB.
- **Modes.** `training: bool`, set by its owner. `epsilon: float`, set by the training loop before each episode.
- **`act`.** Computes the state index (or reuses the one cached by the last `observe`). In training, draws `u = rng.random()`; if `u < epsilon`, the action is `pick(rng, 4)`. Otherwise, and always outside training, the action is `argmax` of the row; `numpy.argmax` returns the first maximum, which is the direction-order tie-break (PRD §2.8).
- **`observe`.** Does nothing outside training. In training, with the previous state `s` and action `a`:
  - The outcome comes from the referee's ruling; the agent never infers it.
  - Target: `reward` alone at `WIN` or `LOSS`. At `None`, `reward + discount * max(Q[s'])` with `s'` the new observation's index. At `TIE`, the setting `tie_update` decides: `"bootstrap"` (default) uses the same target as at `None`; `"terminal"` uses `reward` alone.
  - Update: `Q[s, a] += learning_rate * (target - Q[s, a])`.
  - Cache `s'` only while the episode continues. At a final outcome clear the transition cache after the update; never index a WIN/LOSS observation. At a bootstrap TIE, compute the target from the final observation, then clear the cache because the next action belongs to a new game.

**Why bootstrap at a tie (confirmed default, §16).** Phillip chose to estimate continuation value when the referee stops on the clock, while still ending the episode and all action requests. The state has no move count, so bootstrapping avoids assigning zero continuation solely because a particular visit fell on step 100. This is a learning choice, not a change to the game's tie outcome; terminal mode remains available. The exploratory results for the two settings are similar (§11.3), without establishing equivalence across training runs.

### 10.6 Episode loop (`episode.py`)

```
play_episode(environment, referee, agent, seed) -> EpisodeRecord
    observation = environment.reset(seed)
    referee.start()
    agent.reset(derive(seed, "agent"))
    loop:
        t = referee.next_move()
        result = environment.step(agent.act(observation), t)
        outcome = referee.rule(result)
        agent.observe(result.reward, result.observation, outcome)
        add result.reward to the episode return
        if outcome is not None: return EpisodeRecord(seed, agent.name, outcome, t, return)
        observation = result.observation
```

The same function serves training, evaluation, the difficulty check, simulation and watch mode (watch passes a frame callback, §14.6). Nothing in it branches on the agent.

## 11. Training (FR-9, FR-12, NFR-4)

### 11.1 Seeds and schedule

- Episode `k`, for `k = 0 .. episodes - 1`, plays training seed `k`. Seeds `0 .. episodes - 1` all lie in the training range, because `episodes` is at most 1,000,000 (§4.3).
- Before episode `k`: `epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * k / (episodes * epsilon_decay_fraction))`. At the defaults it falls linearly from 1.0 at episode 0 to 0.05 at episode 25,000 and stays there.
- **Evaluation sizes** are the PRD §2.8 numbers, held as constants: `train.CURVE_EVERY = 2_500`, `train.CURVE_GAMES = 200` and `evaluate.FULL_GAMES = 1_000`. They are not knobs or learner settings: they set how much is measured, not what is played or learned. Tests lower them with `monkeypatch` to stay inside NFR-3 (§17).
- **Learning curve (FR-12).** After every 2,500th episode (episodes 2,500, 5,000, ..., 50,000: 20 points at the defaults), the table is played greedily (`training = False`) on validation seeds 1,000,000 to 1,000,199. The point is `(episodes completed, win rate)`. The table is not changed by these games.
- Restore training mode after each checkpoint, including when validation raises an error; a separate evaluation agent sharing the table without writes is also acceptable. Reset episode caches for each validation and subsequent training game. Tests must show that training updates resume after validation.
- **Final validation.** After the last episode, the table is played greedily on all 1,000 validation seeds. Its summary is stored in the policy file (§12) and shown by the difficulty check (FR-15). The table saved is the final one; no checkpoint is chosen by its score (PRD §2.8).

### 11.2 Outputs

`train --out runs/<name>` writes into that directory:

| File | Content |
|---|---|
| `policy.npz` | §12 |
| `curve.csv` | Header `episodes,win_rate`, one row per learning-curve point |
| `curve.png` | The curve, plotted by `plot.py` |
| `config.json` | The configuration used, as in §4.3 |

### 11.3 Measured

Two exploratory probe training runs at the default configuration, one per `tie_update` setting. Both ran alongside two knob sweeps on Python 3.9.6. These are feasibility measurements, not an upper bound or verification of the production implementation on another runtime. Issue #18 measures the implemented default run on the supported runtime.

| | `bootstrap` (default) | `terminal` |
|---|---|---|
| Training, 50,000 episodes with 20 learning-curve checks | 330 s | 331 s |
| Including the final 1,000-seed validation run | 340 s | 340 s |
| Table rows ever updated | 5,239 of 122,880 | 5,140 of 122,880 |
| Learning curve at 25,000 / 50,000 episodes | 59.0% / 72.5% | 61.0% / 67.5% |
| Validation wins, 1,000 seeds, 95% interval | **69.1%** (66.2–71.9) | 68.0% (65.0–70.8) |
| Validation losses / ties | 26.7% / 4.2% | 26.6% / 5.4% |
| Mean steps to a win | 37.4 | 34.9 |
| vs random: `b`, `c`, one-sided McNemar `p` | 691, 0, below 1e-200 | 680, 0, below 1e-200 |
| vs pathfinding: `b`, `c`, `p` | 570, 21, below 1e-130 | 561, 23, below 1e-130 |

On the same validation seeds the random agent won 0.0% and the pathfinding agent 14.2%.

What this settles:

- **NFR-4 appears feasible:** the probe took about 5.7 minutes against the 10-minute target. The default remains 50,000 episodes, with acceptance pending measurement of the actual implementation in issue #18.
- **The two tie settings agree within their intervals.** D-1 (§16) is a choice on principle, not on results.
- **The curve flattens from about 32,500 episodes**, after epsilon reaches its floor, between about 65% and 73%.
- **ADR-2's state count is not a problem.** Training reaches about 4% of the table's rows; most rows cannot occur (PRD ADR-2).
- **First difficulty reading.** At the defaults the learned agent is inside the PRD's 50% to 90% band on validation maps, and beats both baselines by a wide margin. This is a validation reading, not a result: the test maps were not played, and they are played once, after the team freezes the knobs (PRD §2.7).
- **Replays are exact.** Playing 50 validation seeds twice with the trained table gave identical records.

## 12. Saved-policy file (FR-9, FR-14, FR-15)

One NumPy `.npz` file, written with `numpy.savez` and read with `numpy.load(..., allow_pickle=False)`. It holds two arrays:

| Key | dtype, shape | Content |
|---|---|---|
| `q` | `float64`, `(122880, 4)` | The Q-table |
| `meta` | `uint8`, `(n,)` | UTF-8 bytes of the JSON object below |

```json
{
  "format_version": 1,
  "config": { "game": {...}, "learner": {...} },
  "training_seeds": [0, 49999],
  "learning_curve": [[2500, 0.365], [5000, 0.41]],
  "validation": {
    "seeds": [1000000, 1000999],
    "games": 1000, "wins": 0, "losses": 0, "ties": 0,
    "mean_steps_to_win": 0.0
  }
}
```

`training_seeds` is the first and last seed, inclusive. `mean_steps_to_win` is `null` when there are no wins. The values in the example are placeholders.

**Loading checks.** `load_policy` refuses a file whose `format_version` is not one it knows, whose `q` shape is not `(122880, 4)`, or that lacks any key above. There is one format version in phase 1. A later change to the file adds a version and keeps the loader for version 1, so policy files from earlier runs stay readable.

## 13. Evaluation (FR-10, FR-11)

### 13.1 Running a split

`run_split(config, agent, split, games)` plays the first `games` seeds of the split, in seed order, and returns their `EpisodeRecord`s. Every agent sees the same seeds, so the same maps and the same ghost generator streams (FR-10). A learned agent is evaluated with `training = False`.

### 13.2 Statistics (`stats.py`)

- **Wilson interval** at 95%, with `z = 1.959963984540054`, for wins, losses and ties out of the games played (FR-10).
- **Mean steps to a win:** the mean of `steps` over won episodes; shown as `n/a` when there are none.
- **One-sided exact McNemar (FR-11).** Pair the learned agent's and a baseline's records by seed, reduce each to win or not-win. `b` = seeds the learned agent won and the baseline did not; `c` = the reverse. With `n = b + c`, `p = sum(comb(n, k) for k in b..n) / 2**n`, computed exactly with `fractions.Fraction` and converted to float at the end. `p = 1.0` when `n = 0`. The report shows `b`, `c` and `p` for each baseline.

### 13.3 Evaluation report

`evaluate --split val|test --policy runs/<name>/policy.npz` plays all three agents on all 1,000 seeds of the split under the policy's configuration and prints:

Example layout, with the measured validation numbers for the two baselines at the defaults:

```
split val, 1000 games per agent, knobs: ghost_move_prob=1.0 wall_density=0.2 max_walls=6 min_loops=4 max_steps=100
agent         wins                  losses                ties                  steps/win
random          0   0.0% [ 0.0,  0.4]  1000 100.0% [99.6,100.0]    0   0.0% [ 0.0,  0.4]   n/a
pathfinding   142  14.2% [12.2, 16.5]   858  85.8% [83.5, 87.8]    0   0.0% [ 0.0,  0.4]   13.6
learned       ...
learned vs random:       b=... c=... one-sided exact McNemar p=...
learned vs pathfinding:  b=... c=... one-sided exact McNemar p=...
```

It also writes `runs/<name>/eval-<split>.json`: the configuration, and every `EpisodeRecord` (seed, agent, outcome, steps, return), so any evaluated game can be found and replayed (FR-14).

## 14. Command-line entry points

`python -m pacman <command>`. Every command accepts `--config FILE` and repeatable `--set name=value` (§4.3), and exits non-zero with a message on any error, including `MapGenerationError`.

### 14.1 `train`

`train --out DIR [--config FILE] [--set ...]` — trains (§11) and writes §11.2. It runs in simulation mode (PRD §2.9): it prints each learning-curve point as it is made, and ends with the final validation summary row in the §13.3 format. Refuses an existing `DIR` that already holds a `policy.npz`.

### 14.2 `evaluate`

`evaluate --split {val,test} --policy FILE` — §13.3. The configuration comes from the policy file. `--config` or `--set` with a game knob different from the policy's is an error, because the policy would be evaluated on a game it was not trained for.

### 14.3 `check` (FR-15)

`check [--config FILE] [--set ...] [--policy FILE]` — plays the random and pathfinding agents on all 1,000 validation seeds under the given knobs and prints their summary rows. With `--policy`, it prints beside them the stored `validation` summary and the game knobs from the policy file, marking each knob that differs from the current ones. It never trains. Measured: the two baselines take about 6 s together on 1,000 validation seeds.

### 14.4 `simulate` (FR-17)

`simulate --agent NAME [--agent NAME ...] --split {train,val,test} --games N [--policy FILE] [--config FILE] [--set ...]` — plays the first `N` seeds of the split for each named agent with no graphics, then prints the per-agent summary of §13.3 (counts, percentages, 95% intervals, mean steps to a win). `N` is at most the split's seed count. A learned agent needs `--policy`, and its configuration rules are those of `evaluate`.

### 14.5 `watch` (FR-13, FR-14)

`watch --agent NAME --seed SEED [--policy FILE] [--config FILE] [--set ...] [--delay SECONDS] [--plain]` — plays one game and draws it. The split comes from the seed (§4.2). A seed, an agent and a configuration, plus the policy file for the learned agent, are all it needs, and they are exactly what `eval-<split>.json` records (FR-14). `--delay` defaults to 0.25 s between frames.

### 14.6 Rendering (`render.py`, PRD §2.9)

- **Frames.** One frame at reset and one after each step, through a callback from the episode loop. Each frame redraws the grid, then a status line: the referee's move count `t` of `max_steps`, corners visited out of 4, and whether the ghost can see Pac-Man (line of sight on the true map at the end of the step). The last frame adds the referee's ruling.
- **Pixel art.** Each cell is a 4-by-4-pixel sprite drawn with the upper half block `▀` (U+2580): every character shows two pixels, its foreground the upper and its background the lower, so a cell is 4 characters wide and 2 lines tall. Colours are 256-colour ANSI escapes (`ESC[38;5;Nm`, `ESC[48;5;Nm`). Sprites, each a constant 4x4 grid of colour indices: Pac-Man, ghost, wall, unknown cell, unvisited corner, visited corner, open floor. Pac-Man and the ghost are drawn over their cell's sprite.
- **What is drawn.** Walls and unknown cells are drawn from Pac-Man's memory, so the viewer sees what he knows: a cell he has not sensed shows the unknown sprite, whether it is open or a wall. Pac-Man, the ghost and the corners are always drawn where they are (§16).
- **Plain-text fallback.** One character per cell: `P` Pac-Man, `G` ghost, `#` known wall, `?` unknown, `o` unvisited corner, `*` visited corner, `.` known open. It is used when `--plain` is given, when standard output is not a terminal, when `NO_COLOR` is set, or when `TERM` is `dumb`, and it is what tests check. The renderer returns each frame as a string; only the CLI writes it and sleeps.

## 15. Game-knob sweep (measured)

Each row changes one knob from the defaults and plays the random and pathfinding agents on all 1,000 validation seeds. The random agent won at most 0.2% in every row.

| Knob | Value | Pathfinding wins | 95% interval |
|---|---|---|---|
| (defaults) | | 14.2% | 12.2–16.5 |
| `ghost_move_prob` | 0 | 0.0% | 0.0–0.4 |
| | 0.25 | 10.2% | 8.5–12.2 |
| | 0.5 | 16.7% | 14.5–19.1 |
| | 0.75 | 20.0% | 17.6–22.6 |
| `wall_density` | 0.1 | 19.2% | 16.9–21.8 |
| | 0.3 | 9.6% | 7.9–11.6 |
| | 0.4 | 8.2% | 6.7–10.1 |
| `max_walls` | 3 | 17.1% | 14.9–19.6 |
| | 9 | 14.2% | 12.2–16.5 |
| `min_loops` | 2 | 14.1% | 12.1–16.4 |
| | 8 | 14.5% | 12.5–16.8 |
| `max_steps` | 50 | 14.2% | 12.2–16.5 |
| | 200 | 14.2% | 12.2–16.5 |

What this settles for tuning (PRD §1 step 6, §2.7):

- `ghost_move_prob` peaks near 0.75 for the pathfinding agent and falls on both sides. Its floor is the region below about 0.25, where a mostly still ghost guards `(4, 4)`. Tuning starts in 0.5 to 1.0.
- `wall_density` and `max_walls` move difficulty in one direction: fewer walls, easier game. They also change which maps exist (PRD §2.7).
- `min_loops` and `max_steps` have no measurable effect on the pathfinding agent. It never ties, so `max_steps` cannot matter to it; its effect on the learned agent, which can wait, is unmeasured.
- These are baseline readings. The learned agent's response to each knob is measured only when it is retrained (PRD §2.7, OQ-11).

## 16. Implementation defaults

Each is a choice this document made where the PRD leaves room. Phillip confirmed D-1 as bootstrap while implementing issue #4. D-2 through D-7 retain the architecture v0.2 defaults under the instruction to build the foundation; later tickets implement their behavior. This does not record a separate sign-off by Christian. Changes to these defaults need a recorded decision before dependent work changes its contracts.

| # | Default | Where | Alternative |
|---|---|---|---|
| D-1 | When the referee calls a tie, the Q update bootstraps (`tie_update = "bootstrap"`). | §10.5 | `"terminal"`: a tie ends the update like a win or loss. Measured within noise of the default (§11.3). |
| D-2 | Split hashing keeps the `"5x5:"` tag before the row-major wall string, though the grid is fixed, so the measured splits stand. | §6.3 | The wall string alone. This moves every map to a new split and voids every measurement in this document, so it is only worth doing before code exists. |
| D-3 | `MAX_DRAWS = 10_000` candidates per map. | §6.4 | Any limit well above the measured worst case of 167. |
| D-4 | Training episode `k` uses training seed `k`. | §11.1 | Any fixed list of training seeds. |
| D-5 | Watch mode draws walls and unknown cells from Pac-Man's memory, not the true map. | §14.6 | Draw the true map, marking unsensed cells with a tint. |
| D-6 | `runs/` is git-ignored. Policy files and reports the final report cites are copied out and committed on purpose. | §3 | Commit `runs/` outright. |
| D-7 | `train` runs the final 1,000-seed validation evaluation itself and stores it in the policy file. | §11.1, §12 | A separate command writes it. |

## 17. Tests (NFR-3, NFR-6, NFR-7)

One test file per module. Every test is headless, needs no Matplotlib except `test_cli`'s `train` run, which writes `curve.png` with the `Agg` backend, and the whole suite runs in under 30 seconds (NFR-3). Measured costs to budget against: about 1.6 s to generate 1,000 validation maps, about 3 s for 1,000 pathfinding games, about 6 ms per training episode. Every test uses the 5x5 grid (PRD OQ-15). Tests that need a particular layout build a `GameMap` by hand and pass it to `Environment.reset(seed, game_map=...)`.

| File | Covers | PRD |
|---|---|---|
| `test_config.py` | Defaults equal the PRD's; JSON round trip; unknown key and out-of-range value rejected | §2.7, NFR-2 |
| `test_rng.py` | `derive` golden value (`derive(0, "map").random() == 0.8442537128977315`); components independent; `pick` range | ADR-3 |
| `test_contracts.py` | Fixed grid/action ids; immutable snapshots copy caller-owned containers; 25-cell observation memory; explicit outcome types and Agent protocol | FR-8, NFR-7 |
| `test_mapgen.py` | Corners never walls; connectivity; wall cap (a connected 7-wall map rejected at `max_walls` 6); loop count (empty 5x5 has 16, the ring round a 3x3 block has 1 and is rejected); split membership for many seeds of each split; `layout_bytes`, SHA-256 and bucket golden values (§6.3); golden maps for seeds 0, 1,000,000 and 2,000,000 (§6.3); draw-limit error with `max_draws` small; same seed and split give the same map; the generator's result does not change when a non-generator knob changes | FR-2, NFR-6, NFR-7 |
| `test_splits.py` | No layout shared between a 1,000-seed sample of each split (NFR-6) | NFR-6 |
| `test_grid.py` | `move` into walls and off the grid; `first_step` direction-order ties; line of sight in all four directions, blocked and clear, adjacent | FR-1, FR-5, NFR-7 |
| `test_env.py` | Sensing at reset and after each move; memory resets; no observation's `known` ever shows a wall Pac-Man has not sensed (checked against the true map over many games); every row of the §7.2 table, including Pac-Man stepping onto the ghost's cell, a win before the ghost moves and a loss on the ghost's move; the environment never returns `TIE`; reward totals for each outcome | FR-3, FR-4, FR-7, FR-16, NFR-7 |
| `test_referee.py` | Counts from 1 after `start`; calls `TIE` exactly on move `max_steps` and not before; a win or loss on move `max_steps` stays a win or loss; the agent receives the ruling, and is asked for no move after one; the observation carries no move count | FR-4, NFR-7 |
| `test_ghost.py` | Chase, pursue and search, and every transition between them; clearing `last_seen` on arrival; search least-recent rule and never-visited counts as least recent; random tie draw; stays put on a failed move draw with no visit recorded; draw order (§8.3) | FR-6, NFR-7 |
| `test_agents.py` | Interface conformance of all three; pathfinding corner-order and path ties; a path always exists on memory; feature ranges and the 122,880-row index on 5x5; greedy tie-break; Q update arithmetic on hand-built steps, including win, loss and both `tie_update` settings | FR-8, ADR-2, NFR-7 |
| `test_stats.py` | Wilson against known values; McNemar exact values for small `n`, and `n = 0` | FR-10, FR-11 |
| `test_train.py` | Epsilon schedule values at the start, the decay end and after it; with `CURVE_EVERY` lowered, curve points at the right episode counts, on the right seeds, with the table unchanged by them; the final validation summary stored | FR-9, FR-12 |
| `test_policy_io.py` | Round trip; refusal of a wrong version, wrong shape or missing key | FR-9 |
| `test_determinism.py` | Records identical across two runs in-process and in a subprocess with a different `PYTHONHASHSEED`; a short training run (200 episodes) gives a byte-identical table twice | NFR-1 |
| `test_render.py` | Plain-text frames for a hand-built state; a pixel-art frame contains only expected escape codes and the half-block character | FR-13 |
| `test_cli.py` | Each command runs end to end with `--set episodes=5` and the §11.1 size constants lowered by `monkeypatch`, and writes its outputs; `evaluate` refuses a game knob that differs from the policy's; `watch --plain` replays a seed taken from `eval-<split>.json` | FR-9, FR-14, FR-15, FR-17 |

## 18. Traceability

| Requirement | Built in | Verified by |
|---|---|---|
| FR-1 | `grid.move` | `test_grid` |
| FR-2 | `mapgen.generate` | `test_mapgen` |
| FR-3 | `Environment.reset` | `test_env` |
| FR-4 | `Environment.step`, `Referee` | `test_env`, `test_referee` |
| FR-5 | `grid.line_of_sight`, `Ghost.act` | `test_grid`, `test_ghost` |
| FR-6 | `Ghost` | `test_ghost` |
| FR-7 | `Environment.step`, `Config` | `test_env` |
| FR-8 | `agents/`, `episode.play_episode` | `test_agents`, `test_referee` |
| FR-9 | `train`, `policy_io` | `test_train`, `test_policy_io`, `test_cli` |
| FR-10 | `evaluate.run_split`, `stats.wilson` | `test_stats`, recorded run |
| FR-11 | `stats.mcnemar_one_sided`, `evaluate` | `test_stats`, recorded run |
| FR-12 | `train`, `plot` | `test_train`, `test_cli`, recorded run |
| FR-13 | `render`, `cli watch` | `test_render`, recorded run |
| FR-14 | `cli watch`, `eval-<split>.json` | `test_cli` |
| FR-15 | `cli check` | `test_cli` |
| FR-16 | `Environment` sensing and memory | `test_env` |
| FR-17 | `cli simulate` | `test_cli` |
| NFR-1 | §5 rules | `test_determinism` |
| NFR-2 | `config` | `test_config`; review of every change |
| NFR-3 | test design | the suite's own run time |
| NFR-4 | §11 | recorded training run (§11.3) |
| NFR-5 | `pyproject.toml` | review of every change |
| NFR-6 | `mapgen` split check | `test_splits`, `test_mapgen` |
| NFR-7 | — | the files in §17 |

## 19. SDLC state

Architecture v0.3, against PRD v0.11. Issue #4 implements the shared foundation; see `phase-1-foundation-handoff.md`. The GitHub tracker (#19) provides the remaining implementation tickets and dependencies. Next: review and merge the foundation, then the grid and policy-persistence work can proceed in parallel; finish with the PRD §5 gate. Full-game measurements remain pending implementation.
