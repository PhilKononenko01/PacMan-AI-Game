# Two-pellet hybrid learning experiment

This experiment responds to the professor's suggestion to investigate learning and pathfinding for the ghost. Its target is at least 55% observed Pac-Man wins against a frozen learned ghost. It is a separate ruleset from the historical corner game. The original submitted results do not measure this experiment.

Run the experiment through `python -m pacman.hybrid`. The historical `python -m pacman` commands retain their existing behavior. Both games share grid movement, BFS, map generation, and deterministic random-stream primitives.

## Game rules

The board is a connected 5 by 5 grid generated with the existing wall cap, cycle requirement, and layout-hash partition. Pac-Man starts at `(0, 0)` and the ghost starts at `(4, 4)`. Each game places exactly two distinct pellets on open cells, excluding both starting cells. Both pellets must be collected to win.

Pac-Man receives the ghost's current position and all remaining pellet positions every turn. He senses his own cell and neighboring cells, and remembers discovered walls and corridors. Unexplored walls remain hidden. Route features treat unknown cells as potentially open, so their distances and sight estimates can be wrong.

Pac-Man moves first. Moving onto the ghost loses immediately. Otherwise a pellet is collected once. Collecting the second pellet wins before the ghost can move. In a continuing game the ghost observes, moves, and can capture Pac-Man. If neither actor ends the game, turn 100 produces a terminal timeout. Every scheduled evaluation game counts, including captures and timeouts. No perfect-play solver filters this experiment's denominator.

## Ghost knowledge and control

The ghost knows the static maze for navigation. Its observation contains its own cell, turns remaining, and Pac-Man's cell only when both share an unobstructed row or column. The controller receives neither hidden Pac-Man positions nor hidden movement commands.

Visible sightings refresh memory. Direction is inferred only when two consecutive visible positions differ by one cardinal move. One sighting, a stationary sighting, or reacquisition after lost sight gives an unknown direction. Losing sight retains a previously inferred direction but breaks the consecutive-observation pair.

While Pac-Man is visible the ghost uses BFS toward the visible position. After sight is lost it uses BFS to the last-seen cell. It then investigates targets up to three open cells along a cardinal direction. Each investigation lasts at most four ghost moves, and each sighting permits at most three investigation choices. Memory expires after 12 hidden ghost observations. Expiration, exhausted choices, or exhausted candidate targets returns the ghost to random legal wandering. A new sighting interrupts an investigation immediately.

The fixed and learned ghosts share these visibility, memory, navigation, candidate, and budget rules. The fixed ghost prefers the last observed heading when available and otherwise chooses randomly. The learned ghost uses a Q-table to choose investigation directions. Unseen states with equal values use the fixed rule during evaluation. Learning changes search choice, not access to information or movement speed.

## What learns

Pac-Man learns scores for the four movement directions. Exploration, greedy selection, and bootstrap values use the same legal-move mask. His compact state encodes the planned first step toward a pellet, local openings, ghost offset, remembered-map ghost distance and sight, pellet count, pellet distance, and time remaining. The representation merges some different maze situations. It is an approximation under partial observation, not a claim of a fully observed Markov state.

Pac-Man receives `+1` per pellet, `+100` for winning, `-100` for capture or timeout, and `-0.05` per turn. His learning rate is `0.1` and discount is `1.0`. Timeout is terminal and never bootstraps. These rewards encourage winning, but also include route length and intermediate collection. They do not mathematically optimize win probability alone.

The ghost receives `+1` for either capture timing, `-1` when Pac-Man wins, and `0` for timeout. It receives no reward for distance to hidden Pac-Man, suppressing Pac-Man wins, or remaining alive. Its learning rate is `0.1` and discount is `0.97`.

A ghost search choice may last several game turns. Its pending transition accumulates discounted rewards and elapsed turns until another search choice or termination. The update uses the elapsed-time discount for the next search state's value. Reacquisition does not discard a pending transition. The terminal game turn notifies the ghost even when Pac-Man wins or walks into it before its next action. This assigns capture credit without inventing hidden-state rewards.

## Training protocol

Each replicate uses a distinct block of training layout seeds and independent labeled episode streams. The registered starting budgets are 30,000 baseline Pac-Man games, 30,000 ghost games, and 50,000 adaptation games. Exploration decreases from `1.0` to `0.05` during the first 70% of each stage.

1. Train Pac-Man against the fixed ghost. Save immutable checkpoints at one third, two thirds, and the final episode.
2. Freeze those Pac-Man policies. Train the ghost against a sampled checkpoint pool.
3. Freeze the learned ghost. Continue Pac-Man's final baseline table against an equally mixed fixed and learned ghost schedule.
4. Evaluate both Pac-Man policies against both ghosts. Also evaluate random and BFS Pac-Man baselines under the same rules.

The actors continue to react during frozen-opponent games. Frozen means the Q-table cannot update, not that a route is prerecorded. The ghost still has seeded random wandering and its own episode memory. Training outcomes under exploration are not the headline greedy evaluation metric.

## Evidence and limitations

Artifacts have a separate namespace, role, rules version, feature version, full rule settings, finite numeric tables, and training metadata. Historical policy files are rejected. Frozen tables use detached immutable rows. Opponent hashes are checked before and after training stages and evaluation. New artifacts refuse overwrite.

Validation uses the first requested split seed slots and declared repeat keys. Every scheduled outcome is saved with layout hash, pellet placement, replay key, turns, returns, and policy hashes. Repeated keys and repeated wall layouts are correlated. Ordinary Wilson intervals describe game counts and are labeled as unclustered; independent training-replicate results must also be reported.

The 55% target is an observed all-game win fraction against the learned ghost. Report whether it is met, the uncertainty, and the losses and timeouts. A stronger ghost may lower Pac-Man's win rate. A win-rate gain caused by changing to two pellets or reducing ghost information is a game-rule change, not evidence of better learning. Learning comparisons must use this same ruleset on both sides.

The existing test layouts were evaluated for the historical project. A new evaluation on that partition can assess this ruleset, but must not be described as a never-before-seen dataset. Validation guides development. Any subsequent test run occurs only after policies and the experiment method are frozen, with no test-driven tuning.
