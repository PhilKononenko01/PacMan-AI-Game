# Grid World, Ghost and Learned Pac-Man: PRD

**Status:** DRAFT **v0.11**. OQ-1 to OQ-15 are answered. Issue #4 implements the package foundation and records the confirmed tie-update, cache-reset, reproducibility and final-test decisions in `foundation-handoff.md`.
**Date:** 2026-10-28
**Design baseline:** `PhilKononenko01/PacMan-AI-Game` commit `90c6323`, which holds PRD v0.10 and architecture v0.2 before issue #4. The foundation handoff identifies implemented configuration, RNG and type contracts; gameplay and evaluation behavior below remains a requirement for subsequent tickets.
**Design authority:** `Design/project-proposal.md`, the team's proposal recorded verbatim. It has four paragraphs, cited here as P1 (game), P2 (start positions and vision), P3 (learning and baselines) and P4 (targets).
**Form.** This document states what is decided and what is to be built. Review findings, provenance and version history go in `review.md`. The question state lives in §6 and nowhere else.

---

## Capabilities Delivered

**Live on ship:**
- A repeatable 5x5 grid game: random walls, one Pac-Man, one ghost, a four-corner objective, and win, loss and tie outcomes. Record the seed, configuration, agent/policy and supported runtime for replay.
- A Pac-Man who always knows where the ghost is but discovers walls only as he moves next to them (P2, OQ-2).
- A ghost that chases with shortest-path search when it can see Pac-Man and searches when it cannot (P3).
- Three Pac-Man agents behind one interface: a random agent, a pathfinding baseline, and a reinforcement-learning agent trained on generated maps (P3).
- An evaluation harness that plays every agent on the same held-out maps and reports wins, losses and ties with confidence intervals, and a significance test against each baseline (P3, P4).
- A watch mode that animates a game as pixel art in the terminal, and a simulation mode that plays many games with no graphics and prints a summary at the end (OQ-8).

**Unblocks:**
- The two claims the final report has to make (P4): that the game's difficulty is roughly balanced, with the learned agent winning somewhere around 50% to 90% of held-out games, and that the learned agent beats both baselines.

## 1. Scope

Build the game, the ghost, three agents, training, and evaluation, in that order. Each step is usable before the next one starts.

1. **Environment.** Grid, map generator, Pac-Man's wall sensing and memory, movement, turn order, outcomes, rewards, the referee, watch and simulation modes.
2. **Ghost.** Line of sight, chase, search.
3. **Baselines.** Random agent and pathfinding agent. Running these first gives the first difficulty reading before any learning exists.
4. **Learned agent.** Tabular Q-learning (ADR-2), training loop, saved policy.
5. **Evaluation.** Held-out maps, paired comparison, report tables and the learning curve.
6. **Difficulty tuning.** The team adjusts the game knobs in §2.7 until the learned agent's win rate sits in or near the target band. This is a team call, not something to automate (§2.7).

### Non-goals

- No pellets, power pellets, fruit, lives or score other than the reward. The only objective is the four corners (P1).
- No second ghost (P1 says "a single ghost").
- No learning ghost. The ghost is scripted (P3).
- No grid other than 5x5. The size is fixed, not a parameter, and tests use 5x5 maps too (OQ-15).
- No graphical window and no human-playable mode (OQ-8). The terminal is the only view.
- No deep reinforcement learning unless the tabular agent fails to beat at least one baseline (ADR-2 states when that fallback opens).

## 2. Design

### 2.1 Grid and coordinates

- The grid is 5 rows by 5 columns. A cell is `(row, col)`, with row 0 at the top and col 0 at the left.
- A cell is either open or a wall. Walls occupy whole cells, as in Wumpus World (OQ-3).
- Moves are up, down, left and right. A move into a wall or off the grid leaves the mover in place.

### 2.2 Start positions and corners

- Pac-Man starts at `(0, 0)` and the ghost starts at `(4, 4)`. These are the opposite corners P2 asks for (OQ-1).
- The four corners are `(0, 0)`, `(0, 4)`, `(4, 0)` and `(4, 4)`. Pac-Man's start corner counts as visited at step 0, so three remain to reach (OQ-1).
- The ghost's start corner is one of the four Pac-Man must reach.

### 2.3 Map generation

- Every non-corner cell becomes a wall independently with probability `wall_density` (default 0.20, a §2.7 game knob).
- Corners are never walls.
- A map is accepted only if every open cell is reachable from every other open cell, the map passes the winnability rules below, and it belongs to the requested split (§2.7). A rejected map is redrawn from the same seeded random stream, so a seed and split still map to exactly one map for a fixed generator configuration (`wall_density`, `max_walls`, `min_loops`).
- **Winnability (OQ-13)** is a property of the map alone. The generator never simulates Pac-Man or the ghost, and no knob outside the generator changes which maps exist. A map is winnable when:
  - it has at most `max_walls` walls (default 6, a §2.7 game knob); and
  - its open cells contain at least `min_loops` independent loops (default 4, a §2.7 game knob). For a connected map this is the number of edges between adjacent open cells, minus the number of open cells, plus one. An empty grid has 16. A 3x3 block of walls in the middle, which leaves one path around the edge, has 1, so it is rejected; with 9 walls it also breaks the wall cap.

  The loop rule is what guarantees a choice of routes: more loops means more places Pac-Man can turn away from the ghost and still reach a corner. Winnability does not guarantee that any agent can win a given game.
- At the defaults, about 70% of all draws are accepted before the split check (about 1.4 draws per accepted map), and accepted maps average about 3.4 walls (simulated, 20,000 draws). Every accepted map has at least 4 loops, so none is a single corridor.
- Generation takes an integer seed and a split, and nothing else random. The seed is recorded with every episode result.
- **Bounded generation.** Some generator configurations accept no map in a split: `min_loops` above 16 accepts nothing, and `max_walls` 0 accepts only the empty map, which belongs to one split, so the other two accept nothing. The generator therefore stops after a fixed number of draws and fails with an error naming the configuration and split. It never retries forever.

### 2.4 Turn order and outcomes

At reset (step 0), Pac-Man senses the cells next to his start cell (§2.6). Steps are numbered from 1. Each step runs in this order:

1. Pac-Man chooses and makes a move.
2. If Pac-Man is on the ghost's cell, the episode ends in a **loss**.
3. If Pac-Man is on a corner, that corner is marked visited. If all four are visited, the episode ends in a **win**, before the ghost moves.
4. Pac-Man senses the cells next to his new position (§2.6).
5. The ghost makes its move (§2.5).
6. If the ghost is on Pac-Man's cell, the episode ends in a **loss**.
7. The **referee** counts the move. If the count equals `max_steps` (default 100, a §2.7 game knob), the referee stops the episode and declares a **tie** (OQ-4, OQ-14).

**The referee (OQ-14).** The move count belongs to a referee that stands outside the game. It numbers the steps, and it alone knows `max_steps` and calls a tie. Pac-Man, the ghost and the map never see the count, and no agent's observation carries it. When the referee calls a tie it tells the agent that the episode has stopped and that the outcome is a tie, so the agent never has to infer it. Wins and losses are still decided by steps 2, 3 and 6, and the referee reports them to the agent in the same way.

Because moves are sequential, the two can never pass through each other unseen: if Pac-Man moves onto the ghost's cell, step 2 catches it before the ghost moves. A tie is reported as its own outcome and is never counted as a win.

### 2.5 Ghost

- **Vision (P2).** The ghost sees Pac-Man when they share a row or a column and every cell strictly between them is open. Vision is checked at the start of each ghost move. The ghost knows the whole map.
- **Chase.** When it sees Pac-Man, the ghost takes the first step of a shortest path to Pac-Man's current cell, found by breadth-first search. It also records that cell as Pac-Man's last known position.
- **Pursue last sighting.** When it cannot see Pac-Man but has a last known position it has not yet reached, it takes the first step of a shortest path to that cell. Reaching it clears the record (OQ-5).
- **Search.** Otherwise it moves to the adjacent open cell it has visited least recently. A cell it has never visited counts as less recent than any cell it has. Its start cell counts as visited at step 0, and every cell the ghost moves onto counts as visited at that step, whichever mode moved it there. Ties are broken by a seeded random draw. On a step when it moves, it never stands still while a move exists (OQ-5).
- **Speed.** The ghost moves at most one cell per step, the same as Pac-Man (OQ-6).
- **Move probability (OQ-10).** Each step, after checking vision and updating its last known position, the ghost draws from its seeded generator and moves with probability `ghost_move_prob` (default 1.0, a §2.7 game knob). Otherwise it stays where it is for that step. The draw is made every step, including at 1.0. A step on which the ghost stays put does not change its visit record.
- **Shortest-path ties** are broken by a fixed direction order (up, right, down, left), so the ghost is deterministic given the seed.

### 2.6 Pac-Man's knowledge and rewards

- **The ghost.** Pac-Man always knows the ghost's cell (P2).
- **Walls (OQ-2).** Pac-Man does not start with the map. He knows the grid is 5x5, and at the start and after every move he senses whether each of the four cells next to him is open or a wall. He remembers every cell he has sensed for the rest of the episode. A cell he has not sensed is **unknown**. Memory resets at the start of each episode.
- **Planning on a partial map.** Wherever Pac-Man's side plans a route (the pathfinding agent and the ADR-2 features), unknown cells are treated as open. When a planned route runs into a newly sensed wall, the next step plans again from the updated memory.
- **Who keeps the memory.** The environment keeps Pac-Man's remembered map and passes it in every observation, so every agent sees exactly the same knowledge and no agent can read the true map.
- **Rewards** (defaults, all §2.7 learner settings):

| Event | Reward |
|---|---|
| Win | +100 |
| Loss | -100 |
| First visit to each corner (the start corner gives nothing) | +1 (OQ-7) |
| Each step | -0.05 |
| Tie | No reward beyond what was already earned. |

**Per-step accounting.** Every step, including the last, earns the step cost. A step also earns +1 if its step 3 (§2.4) marked a corner for the first time, and the win or loss reward if the episode ended on it. A step that ends in a loss at §2.4 step 2 never reaches step 3, so it earns no corner reward.

The win and loss rewards are what P1 specifies. The corner reward gives a little credit for progress and is far below the win reward, so it cannot outweigh finishing. The step cost exists only to keep Pac-Man moving: standing still or wandering is never free. It is small enough not to change how the outcomes rank. A tie has reached at most two of the three remaining corners, since the third would have been a win, so it scores at most 2 - 0.05 x 100 = -3, far below any win. A tie scores at least -5, far above any loss.

### 2.7 Game knobs, learner settings and map splits

**Game knobs** change how hard the game is: `ghost_move_prob`, `wall_density`, `max_walls`, `min_loops` and `max_steps` (OQ-10, OQ-13). The three generator knobs, `wall_density`, `max_walls` and `min_loops`, also change which maps exist, so a change to any of them changes every split's maps. `ghost_move_prob` is the primary knob, and it can be set finely. Its effect is not monotonic. Lowering it from 1.0 makes the game easier at first, but a ghost that rarely moves rarely leaves `(4, 4)`, a corner Pac-Man must reach, so below some value the game gets harder again, and at 0 it cannot be won. The pathfinding agent's win rate is about 0% at 0, 10% at 0.25, 20% at 0.75 and 14% at 1.0 (simulated, 1,000 validation seeds at the other defaults; architecture §15). Tuning searches the range above that floor. The ghost's speed and search rule are fixed (§2.5), not knobs. **Learner settings** change how well the agent learns, not how hard the game is: the reward table, the learning rate, the discount factor, the exploration schedule and the number of training episodes (§2.8). Both live in one configuration object, and no value of either is written anywhere else in the code. Only game knobs are turned to adjust difficulty. Turning learner settings to move the win rate into the band would measure the agent's tuning, not the game's balance.

**The team decides difficulty.** Tooling makes the decision cheap. A single command runs both baselines on the validation maps under the current knob values and prints their results side by side, next to the stored learned policy's last validation results and the knob values it was trained under. One knob change can therefore be judged in minutes without retraining. The learned agent is retrained, and run on the validation maps again, when the team considers a knob setting final, and always before the test maps are played (OQ-11).

**Tuning uses validation maps only.** If knobs are tuned against the maps that produce the final numbers, the final win rate is fitted to those maps and no longer measures anything.

**Splits are by map, not by seed.** A 5x5 grid at the default wall density has few likely layouts, so different seeds produce the same map often. In a simulation of the v0.1 generator, 932 of 1,000 test maps also appeared among the first 200,000 training seeds. Disjoint seed ranges therefore do not hold maps out. Instead, every map has a split fixed by its layout: the SHA-256 hash of its wall layout, modulo 100, puts it in training (0 to 79), validation (80 to 89) or test (90 to 99). The generator accepts a map only if it belongs to the requested split, so no map ever appears in two splits. Seeds are still drawn from separate ranges so each split's games are reproducible on their own:

| Split | Map hash bucket | Seeds | Used for |
|---|---|---|---|
| Training | 0 to 79 | 0 to 999,999 | Q-learning episodes |
| Validation | 80 to 89 | 1,000,000 to 1,000,999 | Knob tuning and the learning curve |
| Test | 90 to 99 | 2,000,000 to 2,000,999 | The final reported numbers, run once knobs are frozen |

A map can repeat inside one split. That reflects how often the generator produces it and is not an error.

### 2.8 Agents

All agents implement one interface with three operations:

- **Reset** at the start of each episode, taking that episode's seeded generator for the agent (ADR-3). Agents that use no randomness ignore it.
- **Act:** given an observation, return an action.
- **Observe:** after each step, receive the step's reward, the next observation and the outcome: none while the episode continues, otherwise win, loss or the referee's tie (§2.4, OQ-14). Agents that do not learn ignore it, and the learned agent updates its table only in training.

The observation holds Pac-Man's cell, the ghost's cell, the visited corners and Pac-Man's remembered map (§2.6). The environment never knows which agent is playing.

- **Random agent.** Picks uniformly among the four moves. It is the floor.
- **Pathfinding agent.** Takes the first step of a shortest path to the nearest unvisited corner on Pac-Man's remembered map, treating unknown cells as open, and plans again every step. It ignores the ghost entirely. This is the "basic pathfinding agent" of P4. If it already wins most games, the ghost is too weak, and that is useful to know before training anything.
- **Nearest-corner ties.** When two unvisited corners are equally near, the first in the fixed order `(0, 4)`, `(4, 0)`, `(4, 4)` is chosen. Path ties use the ghost's direction order (§2.5). The ADR-2 features use the same rules.
- **Learned agent.** Tabular Q-learning over the features in ADR-2, with epsilon-greedy exploration during training and a greedy policy at evaluation. Greedy ties are broken by the direction order. The trained table is saved to disk with the configuration and seed range that produced it.
- **Learner settings** (defaults, §2.7): learning rate 0.1, discount factor 0.99, epsilon decaying linearly from 1.0 to 0.05 over the first half of training then held, and 50,000 training episodes. The learning curve (FR-12) is the evidence for changing any of them.
- **Evaluation sizes (OQ-12).** Validation and test evaluations each play all 1,000 seeds of their split. The learning curve checks the current table, greedily, every 2,500 training episodes on the first 200 validation seeds. The table saved at the end of training is the one evaluated; no checkpoint is picked by its validation score.

### 2.9 Watch mode and simulation mode

- **Watch mode** plays one game and animates it in the terminal as pixel art: each cell is drawn as a small coloured block sprite using ANSI colour codes and Unicode block characters, with distinct sprites for Pac-Man, the ghost, walls, unknown cells, and visited and unvisited corners. One frame is drawn per step, with a short delay between frames. Each frame also shows the step number, the corners visited, whether the ghost can see Pac-Man, and at the end the outcome. Any evaluated game can be replayed this way (FR-14).
- **Simulation mode** plays many games with no graphics at all, then prints a summary per agent: wins, losses and ties as counts and percentages, with their 95% intervals, and the mean steps to a win. Training and evaluation run in this mode.
- Both modes use the standard library only (ADR-1). Watch mode has a plain-text fallback for terminals without colour and for tests.

## 3. Decisions (ADRs)

**ADR-1: Python 3.11+, NumPy, pytest; Matplotlib for plots only.** Every course reinforcement-learning resource assumes Python, a 5x5 grid needs no performance work, and NumPy covers the Q-table. Terminal pixel art needs no library. Adding any other dependency needs team agreement first. The architecture doc may reopen this only for a measured reason.

**ADR-2: Tabular Q-learning over map-independent features, not over raw coordinates.**

- *Why not raw coordinates.* A state of Pac-Man's cell, the ghost's cell and the visited corners has 25 x 25 x 8 = 5,000 rows (the start corner is always visited), but a row holds one value per action whatever the map. Walls differ on every map, so the same row would have to mean "go right" on one map and "right is a wall" on the next. The agent would learn an average over maps rather than a policy for the map in front of it.
- *What instead.* The state is built from features Pac-Man can compute from what he knows (§2.6). Features that need walls use his remembered map, with unknown cells treated as open:

| Feature | Values |
|---|---|
| First step of the shortest path to the nearest unvisited corner | 4 |
| Which of the four moves are open (always known, since adjacent cells are sensed) | 16 |
| Where the ghost is, as its row offset and column offset from Pac-Man, each -4 to +4, never both 0 (OQ-9) | 80 |
| Shortest-path distance to the ghost, bucketed: 1, 2, 3, 4 or more | 4 |
| Whether the ghost could see Pac-Man, judged on the remembered map | 2 |
| Corners still unvisited: 1, 2 or 3 | 3 |

  That is 4 x 16 x 80 x 4 x 2 x 3 = 122,880 table rows, about 4 MB as a NumPy array of four 64-bit values per row. Most combinations cannot occur, because the offset limits the path distance and the open moves; the states actually reached are a small fraction, and the learning curve (FR-12) shows whether training visits them often enough. The offset tells Pac-Man where the ghost is; the path distance tells him how far that is once known walls are counted. Because unknown cells count as open, the distance and the line-of-sight feature can be optimistic: the ghost may be farther than it looks, and a wall Pac-Man has not sensed may in fact block the ghost's view. A state describes the situation relative to Pac-Man rather than in map coordinates, which is what lets the policy carry over to held-out maps. It is a summary, so situations that differ in detail can share a state, and the table learns the best action on average across them. In particular, the state does not say whether the ghost is chasing, pursuing or searching, or where it last saw Pac-Man, so two situations that call for different moves can look the same. On a game this small tabular Q-learning usually copes with this; the learning curve (FR-12) is where a failure would show.
- *The cost.* The features use breadth-first search on the remembered map, which the pathfinding baseline also uses. The comparison is therefore "does learning when to deviate from the shortest known path beat always taking it", which is the question P4 asks. The agent sees the direction to the nearest unvisited corner only, so it cannot plan a different corner order; it can only learn when to step away from the shortest route.
- *Fallback.* If the tabular agent fails to beat at least one baseline (§5 item 3) on validation maps after tuning, a small deep Q-network over the full grid becomes in scope. That needs a dependency, so ADR-1's agreement rule applies.

**ADR-3: Determinism is an interface rule.** Every function that uses randomness takes a seeded generator as an argument. Nothing reads a global random state. Map generation, the ghost's tie-breaks, the random agent and training exploration each get their own generator, derived from the episode seed. This is what makes a reported game reproducible and the agent comparison paired.

## 4. Requirements

### Functional

| ID | Requirement |
|---|---|
| FR-1 | The grid is 5x5 with whole-cell walls (§2.1). Moves into walls or off the grid leave the mover in place. |
| FR-2 | Map generation follows §2.3. Corners are never walls, and every accepted map is fully connected, has at most `max_walls` walls, has at least `min_loops` loops and belongs to its requested split. Other maps are rejected and redrawn. The generator does not depend on Pac-Man, the ghost or any non-generator knob. For a fixed generator configuration, a seed and split always produce the same map. If no map is accepted within the draw limit, generation fails with an error (§2.3). |
| FR-3 | Pac-Man starts at `(0, 0)` and the ghost at `(4, 4)`. Pac-Man's start corner is visited at step 0 (§2.2). |
| FR-4 | Each step follows the order in §2.4 and ends in win, loss or tie exactly as stated there. Only the referee counts moves and calls a tie, and it tells the agent the outcome (OQ-14). |
| FR-5 | Ghost vision follows §2.5. The ghost sees Pac-Man only along a shared row or column with no wall between them. |
| FR-6 | The ghost chases, pursues the last sighting, and searches exactly as in §2.5, moves with probability `ghost_move_prob`, and is deterministic for a given seed. |
| FR-7 | The reward table in §2.6 is applied each step. Every value is read from the configuration (§2.7). |
| FR-8 | The random, pathfinding and learned agents implement one interface with reset, act and observe operations (§2.8). The environment does not branch on agent type. |
| FR-9 | Training runs Q-learning on training-split maps (§2.7) and saves the table with its configuration and seed range. |
| FR-10 | Evaluation plays every agent from the same starting conditions: the same maps and the same episode seeds, all 1,000 seeds of the split (§2.8). It reports, per agent, the win, loss and tie rates, each with a 95% Wilson confidence interval, and the mean steps to a win. |
| FR-11 | Evaluation reports a one-sided exact McNemar test of the learned agent against each baseline. Each episode's outcome is reduced to win or not-win (loss and tie both count as not-win), and outcomes are paired by episode seed. A paired test is required because every agent starts each game from the same conditions; the games then differ only through the agents' choices. |
| FR-12 | Training records a learning curve: the greedy validation win rate every 2,500 episodes on the first 200 validation seeds (§2.8), saved and plotted. |
| FR-13 | Watch mode animates a game as terminal pixel art, one frame per step, as in §2.9, with a plain-text fallback. |
| FR-14 | Any evaluated episode can be replayed in watch mode from its seed, agent and configuration, plus the saved policy file for the learned agent. |
| FR-15 | One command runs the §2.7 difficulty check: both baselines on validation maps under the current knobs, shown beside the stored policy's last validation results and the knob values it was trained under. It does not retrain. |
| FR-16 | Pac-Man senses and remembers walls as in §2.6. Observations carry the remembered map, never the true map, and memory resets each episode. |
| FR-17 | Simulation mode plays any number of games without graphics and ends with the per-agent summary in §2.9: wins, losses and ties with their 95% intervals, and the mean steps to a win. |

### Non-functional

| ID | Requirement |
|---|---|
| NFR-1 | The same seed, configuration, learner settings and agent/policy produce repeatable results within the recorded supported runtime (ADR-3). Record Python and dependency versions with reported runs. Exact equality across machines or library versions is not required. |
| NFR-2 | No game knob or learner setting is written outside the configuration object (§2.7). |
| NFR-3 | The full test suite runs headless in under 30 seconds. A slower test is a bug. |
| NFR-4 | A full training run at the default episode count (§2.8), including its learning-curve checks, finishes in under 10 minutes on a team member's laptop. The architecture doc measures this before the episode count is fixed. |
| NFR-5 | Dependencies are limited to those in ADR-1 unless the team agrees otherwise. |
| NFR-6 | No map appears in more than one split, and a test asserts it: the generator never returns a map outside its requested split, and a sample of each split shares no layout with the others (§2.7). |
| NFR-7 | Unit tests cover map connectivity, the corner rule and split membership, the winnability rules (including rejection of a map over the wall cap, and of a connected map with fewer than `min_loops` loops such as the ring around a 3x3 wall block), the error when no map is accepted within the draw limit, wall sensing and memory (including that no observation leaks an unsensed wall), line of sight in every direction including blocked cases, each step of the turn order including Pac-Man moving onto the ghost's cell, the referee's tie on move `max_steps` (and that a win or loss on that move is not turned into a tie), each ghost mode and its transitions, the ghost staying put when its move draw fails, all tie-break rules, and reward accounting for every outcome. |

## 5. Acceptance gate

1. Every FR is demonstrated by a test or by a recorded run, and the NFR-7 suite is green.
2. **Difficulty is roughly balanced (guideline).** On the test maps, with knobs frozen beforehand, the learned agent's win rate is reported with its 95% interval and compared with the 50% to 90% target from P4. The band is a rough guideline, not a pass/fail line: a result near or just outside it is discussed in the report, not treated as a failure.
3. **Learning beats both baselines.** The learned agent's win rate is higher than the random agent's and the pathfinding agent's, each with a one-sided McNemar p-value below 0.05 (FR-11, P4). Both tests must pass, so no correction for running two tests is needed.
4. **The team signs off** the final knob values and the report tables. Difficulty is a judgement, and the numbers above are its evidence, not a replacement for it.

Freeze the policy and all settings before the final held-out evaluation. If a target is missed, document the outcome and limitations without further tuning against those test results.

**Results to distrust.** A win rate of exactly 0% for the pathfinding or learned agent, a win rate of exactly 100% for any agent, identical rates for two different agents, or a test rate well above the validation rate each point to a broken measurement before they point to a result. The random agent winning about 0% is expected (simulated: 0 of 1,000 validation games at the defaults). Check for a map appearing in two splits, a ghost that never moves, ties counted as wins, an observation that leaks the true map, or an agent evaluated on its training maps.

## 6. Questions

No question is open. OQ-1 to OQ-9 were answered on 2026-10-26, OQ-10 to OQ-12 on 2026-10-27, OQ-13 on 2026-10-27 and revised on 2026-10-28, and OQ-14 and OQ-15 on 2026-10-28. Each answer is written into the section it names. A new question goes here first, with a proposal, before any section changes.

- **OQ-1: Start positions.** *Answered:* Pac-Man at `(0, 0)`, ghost at `(4, 4)`, and Pac-Man's start corner counts as visited (§2.2).
- **OQ-2: How much does Pac-Man know?** *Answered:* he always knows where the ghost is, but senses only the walls next to him. He remembers walls he has sensed and plans as if unknown cells are open (§2.6). The agent learns from computed features, not from the raw board (ADR-2).
- **OQ-3: Cell walls or edge walls?** *Answered:* whole-cell walls (§2.1).
- **OQ-4: What does hitting the turn limit count as?** *Answered:* a tie, reported alongside wins and losses and never counted as a win (§2.4). An earlier idea, ending the game when Pac-Man's score goes negative, was dropped: the step cost is only there to keep him moving.
- **OQ-5: What does "explore" mean for the ghost?** *Answered:* go to where it last saw Pac-Man, then visit the cells it has visited least recently (§2.5).
- **OQ-6: Does the ghost move at Pac-Man's speed?** *Answered:* yes, at most one cell per step. A faster ghost was considered as a difficulty knob and replaced by `ghost_move_prob` (OQ-10).
- **OQ-7: Corner rewards?** *Answered:* +1 for the first visit to each corner (§2.6).
- **OQ-8: A playable or graphical version?** *Answered:* watched games are animated as pixel art in the terminal; simulations have no graphics and print a summary of wins, losses and ties at the end. No graphical window and no human play (§2.9).
- **OQ-9: How does the learned agent sense where the ghost is?** *Answered:* its state carries the ghost's position as a row and column offset from Pac-Man (ADR-2).
- **OQ-10: Can the game knobs move difficulty in both directions?** *Answered:* add `ghost_move_prob` (default 1.0), the chance the ghost moves on a given step. It is the primary knob. Ghost speed and search rule are no longer knobs (§2.5, §2.7).
- **OQ-11: Is the learned policy retrained after each knob change?** *Answered:* no. The quick check runs the baselines only and shows the stored policy's last results with the knob values it was trained under. The learned agent is retrained when a knob setting is considered final, and before the test maps are played (§2.7, FR-15).
- **OQ-12: How many games does each evaluation play, and which policy is evaluated?** *Answered:* all 1,000 seeds per split; a learning-curve check every 2,500 episodes on 200 validation seeds, counted in NFR-4; the final table is evaluated (§2.8, FR-10, FR-12).
- **OQ-13: Can a connected but unwinnable map be accepted?** *Answered:* no, and winnability belongs to the map generator alone, with nothing to do with Pac-Man or the ghost. A map is winnable if it has at most `max_walls` walls and at least `min_loops` independent loops, so the routes branch rather than forming one path around a central block. Other maps are rejected and redrawn (§2.3, §2.7, FR-2, NFR-7).
- **OQ-14: Who keeps the move count?** *Answered:* a referee outside the game. It counts moves, stops the episode at `max_steps`, and tells the agent it was a tie; nothing in the game or the observations sees the count (§2.4, §2.8, FR-4, NFR-7).
- **OQ-15: Can the grid size change?** *Answered:* no. The grid is always 5x5; it is not a parameter, even for tests (non-goals, §2.1).

## 7. SDLC state

PRD v0.11. Review rounds 1 to 3 and subsequent decisions are logged in `review.md`. Architecture v0.3 and the issue #4 foundation handoff define the shared contracts. The GitHub implementation tracker (#19) records the remaining module owners and dependencies. Next: review and merge the foundation, implement the remaining tickets, then run the §5 gate.

The architecture doc settles these, which this PRD leaves to it:

- The exact bytes hashed with SHA-256 to put a map in a split (§2.7). A different encoding sends maps to different splits, so it is fixed once and tested.
- How each component's generator is derived from the episode seed (ADR-3), which generator algorithm is used and the version pin that keeps its stream identical across machines, and a rule that nothing depends on Python's per-process string hashing or set iteration order (NFR-1).
- Which training seeds the training episodes use (§2.7, §2.8).
- The saved-policy file format: table, configuration, seed range, and the last validation results that the difficulty check shows (FR-9, FR-14, FR-15).
- The draw limit after which map generation fails (§2.3).
- The Q-learning update on a tie step supports both terminal and bootstrap modes. Phillip confirmed bootstrap as the default during issue #4. The referee still explicitly ends the episode and tells the agent the outcome.
- A measured sweep of each game knob's effect on both baselines' validation win rates, including the floor of `ghost_move_prob` (§2.7), so tuning starts from known directions.
- The command-line entry points for training, evaluation, the difficulty check, watch mode and simulation mode.
