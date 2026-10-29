# PacMan-AI-Game

A Pac-Man-style pathfinding game inspired by Wumpus World, used to test whether a reinforcement-learning agent can learn to beat a scripted ghost. This is our final project.

> **Status: foundation implemented.** Issue #4 provides package setup, validated configuration, RNG helpers and shared interfaces. Gameplay, agents and commands are planned in the remaining tickets; the game is not playable yet.

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

## Planned modes

- **Watch mode:** plays one game and animates it as pixel art in the terminal, step by step. Any evaluated game can be replayed.
- **Simulation mode:** plays many games with no graphics and prints a summary of wins, losses and ties at the end. Used for training and evaluation.

## Tech

Python 3.11+, NumPy and pytest, with Matplotlib for plots. Runs use explicit component seeds and recorded configuration for replay within a supported runtime. Exact results across machines or library versions are not required.

## Development setup

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Any Python 3.11+ interpreter can create the virtual environment. The foundation tests cover configuration validation, RNG repeatability and shared snapshot contracts. Gameplay and CLI tests arrive with their implementation tickets.

See the [foundation handoff](<Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-foundation-handoff.md>) for module ownership and interfaces, and the [implementation tracker](https://github.com/PhilKononenko01/PacMan-AI-Game/issues/19) for dependencies. Claim a ticket before editing its files.

## Repository layout

| Path | What it is |
|---|---|
| [`Design/project-proposal.md`](Design/project-proposal.md) | The team's original proposal, the design authority. |
| [`Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-grid-world-prd.md`](<Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-grid-world-prd.md>) | Phase 1 requirements (PRD): rules, agents, evaluation and acceptance criteria. |
| [`Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-review.md`](<Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-review.md>) | Review findings and team decisions behind the PRD. |
| [`Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-architecture.md`](<Architecture/Phase 1 - Grid World Ghost and Learned Pac-Man/phase-1-architecture.md>) | Module contracts, algorithms, configuration and commands. |
| `pacman/`, `tests/`, `pyproject.toml` | Foundation implementation, tests and package metadata. |
| [`AGENTS.md`](AGENTS.md) | Git rules for AI coding agents working in this repo. |

## Team

- Christian Gore
- Phillip Kononenko ([@PhilKononenko01](https://github.com/PhilKononenko01))

## Roadmap

1. ~~Project proposal~~
2. ~~Phase 1 requirements (PRD), reviewed and all questions answered~~
3. ~~Architecture doc and implementation tickets~~
4. Build: foundation implemented; environment, ghost, baselines, learned agent and evaluation next
5. Tune difficulty, run the final evaluation, write the report
