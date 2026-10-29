# PacMan-AI-Game

A Pac-Man-style pathfinding game inspired by Wumpus World, used to test whether a reinforcement-learning agent can learn to beat a scripted ghost. This is our final project.

> **Status: implemented.** The game, the ghost, all three agents, training, evaluation and the five commands are built and tested (issues #4 to #17). Issue #18 records the final acceptance run; see [`reports/results.md`](reports/results.md).

## The game

- A **5x5 grid** with randomly generated walls. Every map is guaranteed to be fully connected, and corners are never walls.
- **Pac-Man** starts in the top-left corner and must **visit all four corners** to win.
- A single **ghost** starts in the bottom-right corner. Catching Pac-Man ends the game in a loss.
- If **100 turns** pass with neither, the game ends in a **tie**.

**What each side knows:**

| | Pac-Man | Ghost |
|---|---|---|
| The other's position | Always knows where the ghost is | Sees Pac-Man only along a shared row or column with no wall in between |
| The walls | Senses only the four squares next to him, and remembers what he has seen | Knows the whole map |
| Behaviour | Controlled by one of three agents (below) | Chases when it sees Pac-Man, heads to where it last saw him, then searches the squares it has checked least recently |

Both move one square per turn. Pac-Man moves first.

## The agents

| Agent | How it plays |
|---|---|
| **Random** | Picks a random direction each turn. The floor. |
| **Pathfinding** | Takes the shortest known route to the nearest unvisited corner and ignores the ghost. |
| **Learned** | Tabular Q-learning over a compact summary of the situation: where the nearest corner is, which moves are open, where the ghost is and whether it can see Pac-Man. |

**Rewards used in training:** +100 for a win, -100 for being caught, +1 for each new corner, and -0.05 per step to keep Pac-Man moving.

## What we measure

Every agent plays the same 1,000 test maps, which are never seen during training or tuning. We report wins, losses and ties with 95% confidence intervals, and use a paired McNemar test to check whether the learned agent beats each baseline.

**Targets:**
- The learned agent wins roughly **50% to 90%** of games. This is a rough guide to whether the game is balanced, not a hard pass/fail line.
- The learned agent beats **both** baselines, each with p < 0.05.

## Commands

Everything runs as `python -m pacman <command>`. Every command takes `--config FILE` (a JSON file in the shape `config.json` below) and repeatable `--set name=value` overrides of one game knob or learner setting by its bare name, for example `--set max_steps=80`.

| Command | What it does |
|---|---|
| `train --out DIR` | Trains the learned agent on the training maps, prints a learning-curve point every 2,500 episodes, and writes `policy.npz`, `curve.csv`, `curve.png` and `config.json` into `DIR`. Refuses a `DIR` that already holds a policy. |
| `evaluate --split val\|test --policy FILE` | Plays all three agents on every seed of the split under the policy's configuration, prints the table with 95% intervals and the two McNemar tests, and writes `eval-<split>.json` beside the policy so any game can be replayed. |
| `check [--policy FILE]` | Plays the two baselines on the validation maps under the given knobs, and with a policy shows its stored validation result and marks any knob that differs. Never trains. |
| `simulate --agent NAME ... --split S --games N` | Plays the first `N` seeds of a split for each named agent with no graphics and prints the summary. |
| `watch --agent NAME --seed SEED [--policy FILE] [--delay S] [--plain]` | Plays one game and animates it in the terminal as pixel art, one frame per step, with the move count, corners visited, whether the ghost can see Pac-Man and the outcome. `--plain` (or a non-terminal, `NO_COLOR` or `TERM=dumb`) draws plain text. |

A game is identified by its seed: seeds 0 to 999,999 are training maps, 1,000,000 to 1,000,999 validation and 2,000,000 to 2,000,999 test. To replay an evaluated game, take its `seed` and `agent` from `eval-<split>.json` and pass the same `--policy`.

```sh
python -m pacman train --out runs/default
python -m pacman evaluate --split val --policy runs/default/policy.npz
python -m pacman watch --agent learned --seed 1000007 --policy runs/default/policy.npz
python -m pacman simulate --agent random --agent pathfinding --split val --games 200
```

## Tech

Python 3.11+, NumPy and pytest, with Matplotlib for plots. Runs use explicit component seeds and recorded configuration for replay within a supported runtime. Exact results across machines or library versions are not required.

## Development setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Any Python 3.11+ interpreter can create the virtual environment. The suite is one test file per module plus `test_determinism.py` (same seeds, same games, in and across processes) and `test_integration.py` (train, save, load, evaluate, record, replay in one process); it runs headless in well under the 30-second budget. `PACMAN_DEBUG=1` makes a failing command raise with a traceback instead of printing `error: ...`.

See the [foundation handoff](<Architecture/foundation-handoff.md>) for module ownership and interfaces, the [review log](<Architecture/review.md>) for what each ticket built and decided, and the [implementation tracker](https://github.com/PhilKononenko01/PacMan-AI-Game/issues/19) for dependencies.

## Repository layout

| Path | What it is |
|---|---|
| [`Design/project-proposal.md`](Design/project-proposal.md) | The team's original proposal, the design authority. |
| [`Architecture/prd.md`](<Architecture/prd.md>) | Requirements (PRD): rules, agents, evaluation and acceptance criteria. |
| [`Architecture/review.md`](<Architecture/review.md>) | Review findings and team decisions behind the PRD. |
| [`Architecture/architecture.md`](<Architecture/architecture.md>) | Module contracts, algorithms, configuration and commands. |
| `pacman/` | The package: `grid`, `mapgen`, `ghost`, `env`, `referee`, `episode`, `agents/` (random, pathfinding, features, learned), `stats`, `evaluate`, `train`, `policy_io`, `render`, `plot`, `cli`. |
| `tests/`, `pyproject.toml` | One test file per module, the determinism and integration suites, and package metadata. |
| [`reports/results.md`](reports/results.md), `reports/results/` | The acceptance report and the policy, configuration and evaluation files it cites. |
| [`AGENTS.md`](AGENTS.md) | Git rules for AI coding agents working in this repo. |

## Team

- Christian Gore
- Phillip Kononenko ([@PhilKononenko01](https://github.com/PhilKononenko01))

## Roadmap

1. ~~Project proposal~~
2. ~~Requirements (PRD), reviewed and all questions answered~~
3. ~~Architecture doc and implementation tickets~~
4. ~~Build: environment, ghost, baselines, learned agent, training, evaluation and the commands~~
5. Final acceptance (#18): train at the defaults, freeze the knobs, run the held-out evaluation, write the report
