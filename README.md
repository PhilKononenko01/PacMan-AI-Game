# PacMan-AI-Game

A Pac-Man-style pathfinding game inspired by Wumpus World, used to test whether a reinforcement-learning agent can learn to beat a scripted ghost. This is our final project.

> **Status: the ghost changed after acceptance.** On 2026-11-13 the team kept only the pathfinding ghost, which always knows where Pac-Man is. The accepted results, a frozen policy winning 73.6% of 1,000 held-out games against pathfinding's 14.9% and random's 0.0%, were measured against the original ghost, which saw Pac-Man only along rows and columns. Those results, `reports/results/` and the submitted report describe that earlier game; see [`Architecture/review.md`](Architecture/review.md).

## The game

- A **5x5 grid** with randomly generated walls. Every map is guaranteed to be fully connected, and corners are never walls.
- **Pac-Man** starts in the top-left corner and must **visit all four corners** to win.
- A single **ghost** starts in the bottom-right corner. Catching Pac-Man ends the game in a loss.
- If **100 turns** pass with neither, the game ends in a **tie**.

**What each side knows:**

| | Pac-Man | Ghost |
|---|---|---|
| The other's position | Always knows where the ghost is | Always knows where Pac-Man is |
| The walls | Senses only the four squares next to him, and remembers what he has seen | Knows the whole map |
| Behaviour | Controlled by one of three agents (below) | Takes one step along a shortest path to Pac-Man every turn |

Both move one square per turn. Pac-Man moves first.

## The agents

| Agent | How it plays |
|---|---|
| **Random** | Picks a random direction each turn. The floor. |
| **Pathfinding** | Takes the shortest known route to the nearest unvisited corner and ignores the ghost. |
| **Learned** | Tabular Q-learning over a compact summary of the situation: where the nearest corner is, which moves are open, where the ghost is and whether the two are in a clear line. |

**Rewards used in training:** +100 for a win, -100 for being caught, +1 for each new corner, and -0.05 per step to keep Pac-Man moving.

**Ties (optional).** By default a tie carries no reward: the turn limit is treated as a cut-off, so running out the clock costs only the step penalties and the learner can come to prefer it. `--set tie_update=terminal --set reward_tie=-100 --set discount=1.0` makes a tie a final outcome worth exactly as much as being caught. Without `discount=1.0`, a tie 100 turns away is discounted to about a third of its value, and the learner still leans towards it. The tie reward is a training signal only; recorded episode returns do not include it. Every reported result uses the default.

## What we measure

Every agent plays the same seeds. Only games that can be won count: since the ghost is predictable, an exact search over Pac-Man's moves, with the whole map known, decides for each seed whether any route wins, and the other seeds are neither played nor counted. Of the first 1,000 validation seeds, 483 can be won. The generator keeps test layouts separate from training and validation; layouts can repeat within a split. Training still uses every training map. We report wins, losses and ties with 95% confidence intervals, and use a paired McNemar test to check whether the learned agent beats each baseline.

**Targets:**
- The learned agent wins roughly **50% to 90%** of games. This is a rough guide to whether the game is balanced, not a hard pass/fail line.
- The learned agent beats **both** baselines, each with p < 0.05.

## Commands

Everything runs as `python -m pacman <command>`. Every command takes `--config FILE` (a JSON file in the shape `config.json` below) and repeatable `--set name=value` overrides of one game knob or learner setting by its bare name, for example `--set max_steps=80`.

| Command | What it does |
|---|---|
| `train --out DIR` | Trains the learned agent on the training maps, prints a learning-curve point every 2,500 episodes, and writes `policy.npz`, `curve.csv`, `curve.png` and `config.json` into `DIR`. Refuses a `DIR` that already holds a policy. |
| `evaluate --split val\|test --policy FILE` | Plays all three agents on every winnable seed of the split under the policy's configuration, prints the table with 95% intervals and the two McNemar tests, and writes `eval-<split>.json` beside the policy so any game can be replayed. |
| `check [--policy FILE]` | Plays the two baselines on the winnable validation maps under the given knobs, and with a policy shows its stored validation result and marks any knob that differs. Never trains. |
| `simulate --agent NAME ... --split S --games N` | Plays the winnable seeds among the first `N` of a split for each named agent with no graphics and prints the summary. |
| `watch --agent NAME --seed SEED [--policy FILE] [--delay S] [--plain]` | Plays one game and animates it in the terminal as pixel art, one frame per step, with the move count, corners visited and the outcome. `--plain` (or a non-terminal, `NO_COLOR` or `TERM=dumb`) draws plain text. |

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

## Team

- Christian Gore
- Phillip Kononenko ([@PhilKononenko01](https://github.com/PhilKononenko01))

## Roadmap

1. ~~Project proposal~~
2. ~~Requirements (PRD), reviewed and all questions answered~~
3. ~~Architecture doc and implementation tickets~~
4. ~~Build: environment, ghost, baselines, learned agent, training, evaluation and the commands~~
5. ~~Final acceptance (#18): training, freeze, held-out evaluation, report and team table approval~~
