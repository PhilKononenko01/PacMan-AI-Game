# PacMan-AI-Game

A small Pac-Man game with a Q-learning player and a ghost that combines BFS pursuit with learned search. The final experiment uses two randomly placed pellets. The same package still holds the earlier four-corner game that the intermediate report describes.

## Status

Snapshot taken 2026-11-20 on `main` at `29b9a2f`, level with `origin/main`, working tree clean.

- **Final submission is complete.** [`submission/CSCI_3202_Final_Report.ipynb`](submission/CSCI_3202_Final_Report.ipynb) reports the two-pellet game: Pac-Man won 6,691 of 10,000 frozen test games, 66.91%, across five independent training runs. The [experiment report](reports/hybrid-learning.md) and the [evidence directory](reports/hybrid-55/) back it.
- **Test suite is green.** `python -m pytest` gives 567 passed in 22.74 s on Python 3.14.10, NumPy 2.5.4, Matplotlib 3.11.2 and pytest 9.2.0. See [Tests](#tests) for the breakdown.
- **Open work, not part of the submission.** [PR #25](https://github.com/PhilKononenko01/PacMan-AI-Game/pull/25) adds three opt-in options to the corner game and is mergeable, awaiting review. Branch `feature/stronger-learner` holds learner refinements for the corner game with no PR and overlaps PR #25.
- **Held-out maps have been used twice.** Once under the original corner game on 2026-10-29 and once under the frozen two-pellet rules on 2026-11-13. Do not rerun or tune against the test split without a team decision.

Two games live in this package:

| | Final two-pellet game | Legacy four-corner game |
|---|---|---|
| Entry point | `./play`, `python -m pacman.hybrid` | `python -m pacman` |
| Ghost | Sees Pac-Man only along clear rows and columns, remembers sightings, BFS pursuit, learned search choices | Always knows where Pac-Man is and takes one shortest-path step per turn |
| Win | Collect two random pellets within 100 turns | Visit all four corners within 100 turns |
| Games counted | Every scheduled game | Only seeds an exact solver proves winnable |
| Learned Pac-Man | 66.91% of 10,000 frozen test games | 18.2% of 483 winnable validation maps at the defaults |
| Evidence | [`reports/hybrid-learning.md`](reports/hybrid-learning.md), `reports/hybrid-55/` | [`Architecture/review.md`](Architecture/review.md), entries dated 2026-11-13 |

## Run the final game

From the repository root, run:

```sh
./play
```

Each run starts a fresh game with the saved Q-learning Pac-Man and learned
search ghost from `reports/hybrid-55/replicate-0/`. The launcher randomly
chooses a validation map seed and episode key. Layouts can repeat, but the key
also changes pellet placement and movement randomness. The final JSON prints
both identifiers so a game can be repeated. The players use their saved
policies without training during the game.

```sh
./play --delay 0 --plain
./play --seed 1000000 --key 42
./play --seed 1000007 --key 43
./play --pac-baseline bfs
./play --pac-baseline random --fixed-ghost
./play --fixed-ghost
./play --evaluate --games 100 --repeats 2
./play --evaluate --pac-baseline bfs --fixed-ghost --games 100 --repeats 2
./play --help
```

`--seed` selects the wall layout input. `--key` selects pellet placement and
random decisions. Specify both to replay exactly. BFS and random Pac-Man are
comparison players. `--fixed-ghost` keeps the ghost's BFS and memory but removes
learned search choices. `--evaluate` runs a matched validation schedule without
animation and prints wins, losses, ties, and win rate. `--games 100 --repeats 2`
runs 200 games. These demonstrations do not replace the recorded final test.

The launcher uses `.venv/bin/python` when available and otherwise `python3`.
Set up the project dependencies below before first use.

The [final submission notebook](submission/CSCI_3202_Final_Report.ipynb) contains
the report, visible source, saved evidence, live examples, and cited AI appendix.
The [submission guide](submission/final_README.md) explains how to run it.
The cited extra-credit explanation and its two-page PDF copy are embedded in the notebook.

## Two-pellet hybrid experiment

The final experiment trains Q-learning Pac-Man against a ghost with line-of-sight sensing, sighting memory, BFS pursuit, and learned search choices. Pac-Man knows pellet and ghost positions while discovering maze walls. It counts every scheduled evaluation game. The legacy commands and the submitted corner-game results below use a different ruleset.

```bash
python -m pacman.hybrid train --out runs/hybrid --replicate 0
python -m pacman.hybrid evaluate --pac-policy runs/hybrid/pacman.npz --ghost-policy runs/hybrid/ghost.npz --split val --games 1000 --repeats 2
python -m pacman.hybrid watch --pac-policy runs/hybrid/pacman.npz --ghost-policy runs/hybrid/ghost.npz --seed 1000000 --key 42 --plain
python scripts/verify_hybrid_run.py reports/hybrid-55/replicate-0 --all
```

`train` runs three stages with fixed budgets: 30,000 Pac-Man games against the fixed ghost, 30,000 ghost games against frozen Pac-Man checkpoints, then 50,000 Pac-Man games against a mixed fixed and learned ghost schedule. `evaluate` plays a matched schedule and writes a full JSON report with `--out`. `watch` animates one game. The verifier replays every saved game in a run directory and checks its totals, hashes and policies.

See [the experiment rules and training protocol](Architecture/hybrid-experiment.md). The 55% win-rate target requires measured results; training does not guarantee it.

The [completed experiment report](reports/hybrid-learning.md) records a 66.91% Pac-Man win rate across five training runs and 10,000 frozen test games against learned ghosts. Each run exceeded 55%, with win rates from 66.30% to 67.55%. All 125,000 saved evaluation games replayed exactly before archival, and an [independent review](reports/hybrid-55/final-review.md) reproduced the totals.

## Legacy four-corner game

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

> **Where the corner game stands.** On 2026-11-13 the team kept only the pathfinding ghost (PRD OQ-16) and ruled that only winnable games count (OQ-17). Against this ghost the learned agent at the default settings wins 88 of the 483 winnable validation maps, 18.2%, with 70 losses and 325 ties; 100,000 training games raise it to 26.1%. It still beats both baselines, pathfinding 8 wins and random 0, but sits below the 50% guideline. The accepted 73.6% in `reports/results/` and the intermediate report was measured against the original ghost, which saw Pac-Man only along rows and columns; that policy wins 2.4% of validation games against the current ghost. No held-out test game has been played under the current corner rules. See [`Architecture/review.md`](Architecture/review.md).

## The agents

| Agent | How it plays |
|---|---|
| **Random** | Picks a random direction each turn. The floor. |
| **Pathfinding** | Takes the shortest known route to the nearest unvisited corner and ignores the ghost. |
| **Learned** | Tabular Q-learning over a compact summary of the situation: where the nearest corner is, which moves are open, where the ghost is and whether the two are in a clear line. |

**Rewards used in training:** +100 for a win, -100 for being caught, +1 for each new corner, and -0.05 per step to keep Pac-Man moving.

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

## Tests

Run `python -m pytest` from the repository root. The suite is headless. The corner-game tests play no held-out test-seed games; map-generation and split tests inspect test layouts without playing them. One launcher test plays four two-pellet games on test seeds with the frozen final policies to check that `--split` is forwarded; it tunes nothing.

Most recent full run, 2026-11-20 on `main` at `29b9a2f`:

```
567 passed in 22.74s
```

| Area | Files | Tests |
|---|---|---:|
| Shared primitives and the corner game | `test_grid`, `test_mapgen`, `test_rng`, `test_config`, `test_contracts`, `test_splits`, `test_ghost`, `test_env`, `test_referee`, `test_episode`, `test_winnable`, `test_baselines`, `test_features`, `test_learned`, `test_stats`, `test_evaluate`, `test_train`, `test_policy_io`, `test_render`, `test_plot`, `test_cli`, `test_determinism`, `test_integration` | 473 |
| Two-pellet game | `test_hybrid_game`, `test_hybrid_artifacts`, `test_hybrid_experiment`, `test_hybrid_verifier` | 72 |
| `./play` launcher | `test_play` | 22 |
| **Total** | 28 files | **567** |

`test_determinism.py` checks that the same seeds give the same games in and across processes, and `test_integration.py` trains, saves, loads, evaluates, records and replays in one process. `test_hybrid_verifier.py` includes a positive round trip and eight negative controls for altered or missing evidence. Earlier recorded counts were 476 at the corner-game acceptance on 2026-10-29 (`reports/results/tests-final.log`) and 545 when the two-pellet evidence was archived on 2026-11-13 (`reports/hybrid-55/final-tests.log`). `PACMAN_DEBUG=1` makes a failing command raise with a traceback instead of printing `error: ...`.

## Tech

Python 3.11+, NumPy and pytest, with Matplotlib for plots. Runs use explicit component seeds and recorded configuration for replay within a supported runtime. Exact results across machines or library versions are not required.

## Development setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Any Python 3.11+ interpreter can create the virtual environment. Training output goes to `runs/`, which git ignores; the committed evidence lives under `reports/`.

See the [foundation handoff](<Architecture/foundation-handoff.md>) for module ownership and interfaces, the [review log](<Architecture/review.md>) for what each ticket built and decided, and the [implementation tracker](https://github.com/PhilKononenko01/PacMan-AI-Game/issues/19) for dependencies.

## Repository layout

| Path | What it is |
|---|---|
| [`Design/project-proposal.md`](Design/project-proposal.md) | The team's original proposal, the design authority. |
| [`Architecture/prd.md`](<Architecture/prd.md>) | Requirements (PRD) for the corner game: rules, agents, evaluation and acceptance criteria. |
| [`Architecture/architecture.md`](<Architecture/architecture.md>) | Module contracts, algorithms, configuration and commands for the corner game. |
| [`Architecture/hybrid-experiment.md`](Architecture/hybrid-experiment.md) | Rules, observation boundaries, rewards and training protocol for the two-pellet game. |
| [`Architecture/review.md`](<Architecture/review.md>) | Review findings and team decisions, newest first. |
| [`Architecture/foundation-handoff.md`](<Architecture/foundation-handoff.md>) | Module ownership and shared interfaces from the build phase. |
| `pacman/` | The corner game: `grid`, `mapgen`, `ghost`, `env`, `referee`, `episode`, `winnable`, `agents/` (random, pathfinding, features, learned), `stats`, `evaluate`, `train`, `policy_io`, `render`, `plot`, `cli`. |
| `pacman/hybrid/` | The two-pellet game: `game`, `ghost`, `learning`, `models`, `artifacts`, `experiment`, and its command line. |
| `pacman/play.py`, `play` | The `./play` launcher that wraps `pacman.hybrid watch` and `evaluate` with the saved final policies. |
| `scripts/` | Evidence tooling for the two-pellet runs: freeze and test evaluation, archive and summary, verification by full replay, and the results plot. |
| `tests/`, `pyproject.toml` | The 28 test files and package metadata. |
| [`reports/hybrid-learning.md`](reports/hybrid-learning.md), `reports/hybrid-55/` | The two-pellet experiment report and its evidence: five replicate directories with policies and compressed game records, `aggregate.json`, hashes, the test freeze, replay examples and the independent review. |
| [`reports/results.md`](reports/results.md), `reports/results/` | The corner-game acceptance report against the original ghost, with the frozen policy, configuration and evaluation files it cites. |
| `submission/` | The final report notebook and guide, the preserved intermediate notebook, write-up and builder, and the random-versus-random experiment. |
| `AGENTS.md` | Git permissions for AI coding agents working in this checkout. |
| `runs/` | Local training output, ignored by git. |

## Team

- Christian Gore
- Phillip Kononenko ([@PhilKononenko01](https://github.com/PhilKononenko01))

## Roadmap

1. ~~Project proposal~~
2. ~~Requirements (PRD), reviewed and all questions answered~~
3. ~~Architecture doc and implementation tickets~~
4. ~~Build: environment, ghost, baselines, learned agent, training, evaluation and the commands~~
5. ~~Intermediate acceptance (#18): training, freeze, held-out evaluation, report and team table approval~~
6. ~~Corner game revised: pathfinding ghost only, winnable games only (2026-11-13)~~
7. ~~Two-pellet hybrid experiment: five training runs, frozen test, 66.91%, independent review~~
8. ~~Final report notebook executed and committed~~
9. Open: decide whether PR #25 and `feature/stronger-learner` merge, and whether the corner game gets a new acceptance run
