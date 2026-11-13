# Hybrid 55 design

## Recommendation

Build this as an opt-in, versioned experiment alongside the existing game. Reuse `grid.py` (directions, movement, BFS, line of sight), `mapgen.py` (same deterministic layout partitions), RNG derivation, renderer primitives, and the episode/reporting patterns. Do not migrate the current core in place: its `Observation` exposes the ghost cell and its environment owns corner collection, while existing policies, reports, and serialized v1 policies encode those rules. A full migration would invalidate historical results and force broad CLI, policy, and test changes. Instead add an explicit ruleset path, e.g. `pacman/hybrid55/`, with a separate observation/environment/episode runner and dispatch from `train`, `evaluate`, and `watch`. Leave legacy commands and artifacts readable and behaviorally fixed.

## Data model and boundaries

`PacObservation` is immutable: Pac-Man cell, ghost cell, two pellet cells and remaining-pellet bitmask, visited cells, remembered wall/open grid. It is Pac-Man's complete view. `GhostObservation` is immutable and contains ghost cell, optional currently visible Pac-Man cell, last-seen cell, inferred direction only when two consecutive visible observations establish it, remembered map, and turns since sighting. It never contains Pac-Man's hidden cell, hidden-motion direction, pellet location, or hidden-derived rewards. `EpisodeState` privately owns true map, pellet set, both positions, both memories, step count, and terminal outcome. `StepResult` returns each role's observation, its reward, and terminal outcome. `Policy` is an immutable metadata plus Q-table artifact, tagged with rules version, feature schema, role, learner settings, seeds, and opponent pool identifiers.

Separate observation constructors by role at the environment boundary; agent interfaces receive only their role's observation. Every move uses the same sequential order: Pac-Man moves, capture is checked, pellet is collected (win on second), sensing updates, then ghost acts and capture is checked; referee ends at 100 turns as a tie. Notify both learners of the same transition boundary, but only their own public reward.

## Agents and learning

Use one ghost interface for both baseline and learned ghosts: local BFS navigation on the true map toward last-seen position, falling back to bounded investigation of last-seen cell plus inferred direction; when memory expires, wander among legal moves using its own RNG. The learned ghost chooses only the bounded search/investigate/wander decision; BFS still executes navigation. Baseline uses the same visibility, memory expiry, legal actions, and navigation with a fixed simple search rule. This isolates the learned decision and gives fair comparisons. The ghost receives terminal capture reward only when its move catches Pac-Man; Pac-Man gets capture loss, pellet reward once per pellet, second-pellet win, per-turn cost, and timeout tie with no success bonus. Do not reward a hidden position, near miss, or predicted capture. Apply the final transition update exactly once; win/loss are terminal and timeout uses configured tie handling.

Pac-Man's feature vector should encode legal/remembered moves, pellet-relative navigation targets, ghost-relative offset and remembered-map distance/visibility, and remaining pellet count. Ghost features encode only `GhostObservation`, turn-to-expiry, candidate investigation direction, and BFS distance from ghost to last-seen target. Version each encoding independently. A compact tabular or linear Q learner is sufficient; do not add replay/framework machinery. Ensure legal action selection and deterministic tie order.

## Training, evaluation, watch

Keep explicit `train --rules hybrid55`, `evaluate --rules hybrid55 --policy ...`, and `watch --rules hybrid55 --agent ...`; reject a policy with a different rules or feature version before play. Save checkpoints and manifests atomically to a new run directory; never overwrite or relabel legacy results. Evaluation loads frozen policies and verifies their hashes/metadata. `watch` renders true positions for the human spectator, but never passes its frame to an agent.

Train Pac-Man first against the fixed visibility-limited BFS ghost, with two uniquely sampled open pellets excluding both spawn cells. Then freeze a diverse Pac-Man checkpoint pool and train the ghost against that pool; freeze the resulting ghost pool and train Pac-Man against a fixed mixture of baseline and learned ghosts. Keep opponent policies immutable during each learner's run. Derive independent deterministic RNG streams from replicate seed for layout, pellet placement, ghost movement, Pac-Man exploration, and ghost decisions. Record replicate seed ranges and training curves; do not tune rules or select evaluation seeds to reach 55%.

Evaluate all scheduled games on the existing test partition, five independent training seeds, and report wins/losses/ties (all count in denominator), per-map outcomes, aggregate win rate with Wilson interval, and opponent breakdown. Report target met or missed plainly. Show all scheduled games in the headline, including unwinnable maps; no perfect-play filter.

## Required behavioral checks

Cover hidden-position invariance (identical ghost observations/actions for states differing only in unseen Pac-Man position), visibility and remembered-direction transitions, memory expiry, two unique pellets and reward-once, both collision timings and second-pellet termination, timeout with no success reward, frozen-opponent immutability, reproducible independent RNG streams, policy schema/version rejection, and renderer isolation. These are the gates before interpreting any win-rate result.
