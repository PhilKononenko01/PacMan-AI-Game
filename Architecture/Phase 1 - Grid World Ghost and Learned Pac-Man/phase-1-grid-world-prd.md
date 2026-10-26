# Phase 1 — Grid World, Ghost and Learned Pac-Man: PRD

**Status:** DRAFT **v0.1**. Owed: the team's answers to OQ-1 through OQ-8. FR-2 through FR-6 are written against the proposed answers and change if the team picks something else.
**Date:** 2026-10-26
**Describes the repository at:** `PhilKononenko01/PacMan-AI-Game` commit `de01224` ("Initial commit"). That commit holds one file, `README.md`, whose whole content is the heading `# PacMan-AI-Game` (shown by a clone on 2026-10-26). No code exists yet, so every statement below about behaviour is about what will be built, not about what the project does.
**Design authority:** `Design/project-proposal.md`, the team's proposal recorded verbatim. It has four paragraphs, cited here as P1 (game), P2 (start positions and vision), P3 (learning and baselines) and P4 (targets).
**Form.** This document states what is decided and what is to be built. Review findings, provenance and version history go in `phase-1-review.md`. The open-question state lives in §6 and nowhere else.

---

## Capabilities Delivered

**Live on ship:**
- A deterministic 5x5 grid game: random walls, one Pac-Man, one ghost, a four-corner objective, win, loss and timeout outcomes. Same seed, same game, on any machine.
- A ghost that chases with shortest-path search when it can see Pac-Man and explores when it cannot (P3).
- Three Pac-Man agents behind one interface: a random agent, a pathfinding baseline, and a reinforcement-learning agent trained on generated maps (P3).
- An evaluation harness that plays every agent on the same held-out maps and reports win rate with confidence intervals and a significance test against each baseline (P3, P4).
- A text renderer and episode replay, so any reported game can be watched step by step.

**Unblocks:**
- The two claims the final report has to make (P4): that the game's difficulty is balanced, meaning the learned agent wins between 50% and 90% of held-out games, and that the learned agent beats both baselines.

## 1. Scope

Build the game, the ghost, three agents, training, and evaluation, in that order. Each step is usable before the next one starts.

1. **Environment.** Grid, map generator, movement, turn order, outcomes, rewards, renderer.
2. **Ghost.** Line of sight, chase, explore.
3. **Baselines.** Random agent and pathfinding agent. Running these first gives the first difficulty reading before any learning exists.
4. **Learned agent.** Tabular Q-learning (ADR-2), training loop, saved policy.
5. **Evaluation.** Held-out maps, paired comparison, report tables and the learning curve.
6. **Difficulty tuning.** The team adjusts the environment knobs in §2.7 until the learned agent's win rate lands in the target band. This is a team call, not something to automate (§2.7).

### Non-goals

- No pellets, power pellets, fruit, lives or score other than the reward. The only objective is the four corners (P1).
- No second ghost (P1 says "a single ghost").
- No learning ghost. The ghost is scripted (P3).
- No grid larger than 5x5 in any reported result. Grid size is a parameter so tests can use small grids, but only 5x5 is evaluated.
- No graphical game client unless OQ-8 is answered yes. The text renderer is the only view this phase.
- No deep reinforcement learning unless the tabular agent misses the gate (ADR-2 states when that fallback opens).
- No human-playable mode unless OQ-8 is answered yes.

## 2. Design

### 2.1 Grid and coordinates

- The grid is 5 rows by 5 columns. A cell is `(row, col)`, with row 0 at the top and col 0 at the left.
- A cell is either open or a wall. Walls occupy whole cells, as in Wumpus World (OQ-3 asks whether the team meant walls between cells instead).
- Moves are up, down, left and right. A move into a wall or off the grid leaves the mover in place.

### 2.2 Start positions and corners

- Pac-Man starts at `(0, 0)` and the ghost starts at `(4, 4)`. These are the opposite corners P2 asks for (OQ-1).
- The four corners are `(0, 0)`, `(0, 4)`, `(4, 0)` and `(4, 4)`. Pac-Man's start corner counts as visited at step 0 (OQ-1).
- The ghost's start corner is one of the four Pac-Man must reach. Pac-Man therefore has to enter ground the ghost starts on, which is a large part of the difficulty.

### 2.3 Map generation

- Every non-corner cell becomes a wall independently with probability `wall_density` (default 0.20, a §2.7 knob).
- Corners are never walls.
- A map is accepted only if every open cell is reachable from every other open cell. A rejected map is redrawn from the same seeded random stream, so a seed still maps to exactly one map.
- Generation takes an integer seed and nothing else random. The seed is recorded with every episode result.

### 2.4 Turn order and outcomes

Each step runs in this order:

1. Pac-Man chooses and makes a move.
2. If Pac-Man is on the ghost's cell, the episode ends in a **loss**.
3. If Pac-Man is on a corner, that corner is marked visited. If all four are visited, the episode ends in a **win**, before the ghost moves.
4. The ghost makes its move (§2.5).
5. If the ghost is on Pac-Man's cell, the episode ends in a **loss**.
6. If the step count has reached `max_steps` (default 100, a §2.7 knob), the episode ends in a **timeout**.

Because moves are sequential, the two can never pass through each other unseen: a swap of cells is caught at step 2. A timeout is not a win in any reported win rate (OQ-4).

### 2.5 Ghost

- **Vision (P2).** The ghost sees Pac-Man when they share a row or a column and every cell strictly between them is open. Adjacent cells always see each other. Vision is checked at the start of each ghost move.
- **Chase.** When it sees Pac-Man, the ghost takes the first step of a shortest path to Pac-Man's current cell, found by breadth-first search. It also records that cell as Pac-Man's last known position.
- **Pursue last sighting.** When it cannot see Pac-Man but has a last known position it has not yet reached, it takes the first step of a shortest path to that cell. Reaching it clears the record. This is the proposed reading of "explore" right after losing sight (OQ-5).
- **Explore.** Otherwise it moves to the adjacent open cell it has visited least recently, breaking ties with a seeded random draw. It never stands still while a move exists.
- **Speed.** The ghost moves one cell per step, the same as Pac-Man (OQ-6).
- **Shortest-path ties** are broken by a fixed direction order (up, right, down, left), so the ghost is deterministic given the seed.

### 2.6 Pac-Man's knowledge and rewards

- **Knowledge.** Pac-Man always knows the ghost's cell (P2). Pac-Man also knows the whole map (OQ-2). Without the map, the pathfinding baseline P3 asks for could not exist.
- **Rewards** (defaults, all §2.7 knobs):

| Event | Reward |
|---|---|
| Win | +100 |
| Loss | -100 |
| Each step | -1 |
| First visit to a corner | 0 by default. OQ-7 asks whether to enable it. |
| Timeout | No reward beyond the step costs already paid. |

The win and loss rewards are what P1 specifies. The step cost is an addition. Without it, a policy that hides until timeout scores the same as one that loses slowly, and Q-learning has no pressure to finish.

### 2.7 Difficulty knobs

One configuration object holds every value that changes how hard the game is: `wall_density`, `max_steps`, the ghost's speed and explore rule (OQ-5, OQ-6), and the reward table. No difficulty value is written anywhere else in the code.

**The team decides difficulty.** Tooling makes the decision cheap. A single command runs both baselines and the current learned policy on the validation maps and prints their win rates side by side, so one knob change can be judged in minutes.

**Tuning uses validation maps only.** If knobs are tuned against the maps that produce the final numbers, the final win rate is fitted to those maps and no longer measures anything. Map seeds are split three ways and the ranges never overlap:

| Split | Seeds | Used for |
|---|---|---|
| Training | 0 to 999,999 | Q-learning episodes |
| Validation | 1,000,000 to 1,000,999 | Knob tuning and the learning curve |
| Test | 2,000,000 to 2,000,999 | The final reported numbers, run once knobs are frozen |

### 2.8 Agents

All agents implement one interface: given an observation, return an action. The environment never knows which agent is playing.

- **Random agent.** Picks uniformly among the four moves. It is the floor.
- **Pathfinding agent.** Takes the first step of a shortest path to the nearest unvisited corner, by breadth-first search. It ignores the ghost entirely. This is the "basic pathfinding agent" of P4. If it already wins most games, the ghost is too weak, and that is useful to know before training anything.
- **Learned agent.** Tabular Q-learning over the features in ADR-2, with epsilon-greedy exploration during training and a greedy policy at evaluation. The trained table is saved to disk with the configuration and seed range that produced it.

## 3. Decisions (ADRs)

**ADR-1: Python 3.11+, NumPy, pytest; Matplotlib for plots only.** Every course reinforcement-learning resource assumes Python, a 5x5 grid needs no performance work, and NumPy covers the Q-table. Adding any other dependency needs team agreement first. The architecture doc may reopen this only for a measured reason.

**ADR-2: Tabular Q-learning over map-independent features, not over raw coordinates.**

- *Why not raw coordinates.* A state of Pac-Man's cell, the ghost's cell and the visited corners has 25 x 25 x 16 = 10,000 rows, but a row holds one value per action whatever the map. Walls differ on every map, so the same row would have to mean "go right" on one map and "right is a wall" on the next. The agent would learn an average over maps rather than a policy for the map in front of it.
- *What instead.* The state is built from features Pac-Man can compute from what it knows (§2.6):

| Feature | Values |
|---|---|
| First step of the shortest path to the nearest unvisited corner | 4 |
| Which of the four moves are open | 16 |
| Shortest-path distance to the ghost, bucketed: 1, 2, 3, 4 or more | 4 |
| First step of the ghost's shortest path toward Pac-Man | 4 |
| Whether the ghost can currently see Pac-Man | 2 |
| Corners still unvisited: 1, 2 or 3 | 3 |

  That is 4 x 16 x 4 x 4 x 2 x 3 = 6,144 states, small enough to visit thoroughly in a short training run. The same state means the same situation on every map, which is what lets the policy carry over to held-out maps.
- *The cost.* The features use breadth-first search, which the pathfinding baseline also uses. The comparison is therefore "does learning when to deviate from the shortest path beat always taking it", which is the question P4 asks. If the course requires the agent to learn from raw input, OQ-2 is where that is said, and the fallback below applies.
- *Fallback.* If the tabular agent misses the gate (§5) on validation maps after tuning, a small deep Q-network over the full grid becomes in scope. That needs a dependency, so ADR-1's agreement rule applies.

**ADR-3: Determinism is an interface rule.** Every function that uses randomness takes a seeded generator as an argument. Nothing reads a global random state. Map generation, the ghost's tie-breaks, the random agent and training exploration each get their own generator, derived from the episode seed. This is what makes a reported game reproducible and the agent comparison paired.

## 4. Requirements

### Functional

| ID | Requirement |
|---|---|
| FR-1 | The grid is 5x5 with cell walls (§2.1). Moves into walls or off the grid leave the mover in place. |
| FR-2 | Map generation follows §2.3. Corners are never walls, every accepted map is fully connected, and a seed always produces the same map. |
| FR-3 | Pac-Man starts at `(0, 0)` and the ghost at `(4, 4)`. Pac-Man's start corner is visited at step 0 (§2.2). |
| FR-4 | Each step follows the order in §2.4 and ends in win, loss or timeout exactly as stated there. |
| FR-5 | Ghost vision follows §2.5. The ghost sees Pac-Man only along a shared row or column with no wall between them. |
| FR-6 | The ghost chases, pursues the last sighting, and explores exactly as in §2.5, and is deterministic for a given seed. |
| FR-7 | The reward table in §2.6 is applied each step. Every value is read from the configuration (§2.7). |
| FR-8 | The random, pathfinding and learned agents implement one interface (§2.8). The environment does not branch on agent type. |
| FR-9 | Training runs Q-learning on training-split maps (§2.7) and saves the table with its configuration and seed range. |
| FR-10 | Evaluation plays every agent on the same maps with the same ghost seeds. It reports, per agent, the win, loss and timeout rates, each with a 95% Wilson confidence interval, and the mean steps to a win. |
| FR-11 | Evaluation reports a McNemar test of the learned agent against each baseline on the paired outcomes. A paired test is required because every agent plays the identical games. |
| FR-12 | Training records a learning curve: validation win rate every fixed number of episodes, saved and plotted. |
| FR-13 | A text renderer draws the grid, walls, Pac-Man, the ghost, visited corners, and whether the ghost can see Pac-Man. |
| FR-14 | Any evaluated episode can be replayed from its seed and agent, step by step, through the renderer. |
| FR-15 | One command runs the §2.7 difficulty check: both baselines and the current policy on validation maps, with win rates side by side. |

### Non-functional

| ID | Requirement |
|---|---|
| NFR-1 | The same seed and configuration produce identical maps, ghost moves, training results and evaluation numbers on any machine (ADR-3). |
| NFR-2 | No difficulty value is written outside the configuration object (§2.7). |
| NFR-3 | The full test suite runs headless in under 30 seconds. A slower test is a bug. |
| NFR-4 | A full training run finishes in under 10 minutes on a team member's laptop. |
| NFR-5 | Dependencies are limited to those in ADR-1 unless the team agrees otherwise. |
| NFR-6 | Training, validation and test seed ranges never overlap, and a test asserts it (§2.7). |
| NFR-7 | Unit tests cover map connectivity and the corner rule, line of sight in every direction including blocked cases, each step of the turn order including a swap of cells, each ghost mode and its transitions, and reward accounting for every outcome. |

## 5. Acceptance gate

1. Every FR is demonstrated by a test or by a recorded run, and the NFR-7 suite is green.
2. **Difficulty is in band.** On the 1,000 test maps, with knobs frozen beforehand, the learned agent's win rate is between 50% and 90% (P4).
3. **Learning beats both baselines.** The learned agent's win rate is higher than the random agent's and the pathfinding agent's, each with a McNemar p-value below 0.05 (P4).
4. **The team signs off** the final knob values and the report tables. Difficulty is a judgement, and the numbers above are its evidence, not a replacement for it.

**Results to distrust.** A win rate of exactly 0% or 100%, identical rates for two different agents, or a test rate well above the validation rate each point to a broken measurement before they point to a result. Check for overlapping seed ranges, a ghost that never moves, timeouts counted as wins, or an agent evaluated on its training maps.

## 6. Open questions

Each question carries a proposal, and the FRs are written against it. The team answers, and the answer is written back here and into the section the question names.

- **OQ-1: Start positions and the start corner.** P2 says "fixed positions opposite each other" but does not name them. *Proposal:* Pac-Man at `(0, 0)`, ghost at `(4, 4)`, and Pac-Man's start corner counts as visited. The alternative, starting both off the corners, makes the game longer and changes the step limit.
- **OQ-2: How much does Pac-Man know, and must it learn from raw input?** P2 gives Pac-Man the ghost's position. P1's Wumpus World reference could mean Pac-Man only senses nearby walls. *Proposal:* full map knowledge, which the pathfinding baseline needs, and the feature state of ADR-2. If the course requires learning from raw input, ADR-2's fallback becomes the plan.
- **OQ-3: Cell walls or edge walls?** "Randomly generated walls" fits either. *Proposal:* whole-cell walls, as in Wumpus World. Edge walls would change line of sight and map generation.
- **OQ-4: What does a timeout count as?** P1 names only win and loss. *Proposal:* a timeout is its own outcome, reported separately, and is not a win. Counting it as a loss is the stricter alternative.
- **OQ-5: What does "explore" mean for the ghost?** P3 says the ghost explores when Pac-Man is out of sight. *Proposal:* go to the last place it saw Pac-Man, then visit the least recently visited neighbours. A purely random walk is simpler and makes the ghost weaker.
- **OQ-6: Does the ghost move at Pac-Man's speed?** Not stated. *Proposal:* the same speed, one cell per step. On a 5x5 grid, an equal-speed ghost can usually be outrun, so if the game is too easy the first knob to try is letting the ghost move twice every few steps.
- **OQ-7: Corner rewards?** P1 rewards only the finished objective. *Proposal:* keep that as the default, and enable a reward for each new corner only if training stalls. Its size must then stay well below the win reward so that it cannot outweigh finishing.
- **OQ-8: A playable or graphical version?** P1 calls this a game, but the deliverable P3 and P4 describe is an experiment. *Proposal:* text renderer and replay only this phase. A Pygame window or keyboard play would follow if the team wants a demo.

## 7. SDLC state

PRD drafted. Next, per the workflow this project follows: PRD doc review, logged in `phase-1-review.md`; then the architecture doc (`phase-1-architecture.md`); then epics, the coding plan, implementation and the §5 gate.
