# Recommended boundary

Implement the approved two-pellet game as `pacman.hybrid`, exposed by `python -m pacman.hybrid train|evaluate|watch`. Share `grid`, `mapgen`, `rng`, and scalar statistics. Keep legacy commands, policies, recorded results, and report unchanged. This is a distinct ruleset, not a replacement interpretation of the old experiments.

Full core migration could reuse the existing episode runner, renderer, and CLI, but all three embody corner collection and the existing evaluator filters through `is_winnable`. Existing `Policy` metadata has no rules identifier, and `LearnedAgent(training=False)` still owns a mutable array. Migrating requires compatibility routing across environment, features, artifacts, evaluation, rendering, and tests. That cost buys little for one new experiment. An isolated package keeps one authoritative implementation of the new rules and small explicit imports of genuinely shared primitives. Avoid a universal environment framework or duplicated map generation.

# Data shapes first

Use frozen dataclasses with tuple/frozenset fields.

```python
PacObservation(pacman, ghost, pellets, known, turns_left)
Seen(position, direction, age)
GhostObservation(cell, visible_pacman, memory, turns_left)
EpisodeSpec(layout_seed, pellet_seed, ghost_seed, agent_seed)
TurnResult(pac_observation, pac_reward, ghost_reward, outcome)
EpisodeRecord(spec, pac_policy_id, ghost_policy_id, outcome, turns, returns)
```

`pellets` contains remaining cells. Exactly two unique open cells are sampled from sorted open cells excluding both spawns. Pac-Man globally knows ghost and pellets, senses adjacent walls, and retains its 25-cell knowledge tuple. `visible_pacman` is optional. `memory` is either absent or a bounded last-visible record. The ghost knows the static map for BFS navigation, but neither observations nor callbacks contain a hidden current Pac-Man cell, hidden trajectory, or hidden direction.

The environment alone owns true positions, pellet removal, collision checks, and horizon. Preserve established order. Pac-Man moves, collision is checked, pellet is consumed and a second pellet wins immediately, then the ghost observes and moves, then collision is checked. Turn 100 ends in a tie if no earlier result. Sensing and returned observations reflect the completed turn. An ended environment rejects another step.

# Ghost controller and learning boundary

Both baseline and learned ghost use one controller. Visible Pac-Man resets memory and triggers BFS pursuit. Direction is inferred only from two consecutive visible observations at the same observation point in successive turns, and only for an adjacent displacement. Lost sight breaks that pair immediately. A stationary observation provides no directional movement. No environment-supplied movement direction enters the controller.

Without sight, BFS reaches the stored cell. There the controller chooses a bounded investigation target from a small fixed action set, such as remembered direction, left, right, and reverse. Unknown direction uses fixed world directions. Legal reachable targets are derived only from memory and the static map. Each choice commits at most a fixed number of ghost turns and is interrupted by renewed sight or memory expiration. A shared legal-action mask excludes ineffective targets. Memory expires after a fixed count of invisible observations, after which shared seeded random wandering begins. Baseline chooses uniformly from the same choices. Learned chooses with tabular epsilon-greedy Q values. Navigation, visibility, aging, movement probability, option limits, and rewards remain identical.

```python
controller.reset(rng, frozen_policy_or_trainer)
controller.act(observation, game_map) -> Direction | None
controller.observe_turn(ghost_reward, outcome) -> None
controller.finish(outcome) -> None
```

`observe_turn` runs exactly once for every completed game turn, including a Pac-Man terminal action that prevents a ghost move. Its reward is capture-only positive reward, with zero for timeout and no hidden-distance shaping. The controller owns memory and a pending search decision. The trainer owns the mutable table.

Treat search decisions as semi-Markov transitions. At the next search decision, close the previous pending decision with its accumulated discounted rewards and `gamma**duration * max(Q(next_search_state))`, then choose a new action. Count every elapsed game turn, including missed movement draws and intervening pursuit, rather than one discount per option. On terminal, close any pending decision with no bootstrap. A terminal before any search decision creates no update. `finish` checks the episode lifecycle, clears pending state, and performs no second reward delivery. Thus a capture during pursuit can train an earlier search decision without inventing rewards for search timeouts. Pac-Man uses ordinary per-turn Q updates with terminal ties and exactly one final update.

# Compact modules and immutable policies

Use `game.py` for rules/observations/controller, `learning.py` for encoders and two learners, `artifacts.py` for validated snapshots, `experiment.py` for episode/training/evaluation, and `__main__.py` for commands and a pellet-aware text renderer. Keep feature encodings explicit and versioned; begin with compact tabular features, not a neural stack.

```python
trainer.freeze() -> FrozenPolicy
FrozenPolicy.choose(state, legal_actions) -> int
run_episode(spec, rules, pac_actor, ghost_actor, on_frame=None) -> EpisodeRecord
train_stage(plan, frozen_opponents, train_seed) -> StageResult
```

`FrozenPolicy` copies tables into immutable bytes and exposes only read-only views, or stores an immutable sparse tuple representation. A frozen dataclass around an owning writable NumPy array is insufficient. It has no update method or training flag. Actors keep episode RNG and memory separately from the snapshot. Evaluation accepts frozen snapshots only. Hash snapshots before and after stages as an additional assertion.

Artifacts carry format version, rules ID/config hash, role, feature version, table encoding/shape, finite values, training replicate, seed schedule, stage lineage, opponent IDs/hashes, selection protocol, and validation counts. Reject incompatible legacy artifacts and rule overrides. Write atomically and never overwrite legacy results.

# Training and evidence

Derive replicate-specific streams with independent layout, pellet, ghost, agent, exploration, and opponent-selection keys. Preserve `mapgen`'s SHA256 split contract. Sample layouts only from the relevant split. A replicate changes action/pellet randomness independently of map partitions.

Train Pac-Man against fixed baseline, saving checkpoints. Freeze the selected checkpoint pool. Train ghost against that fixed pool, then freeze it. Continue Pac-Man against a declared baseline/learned mixture. Select checkpoints on validation only. Evaluate final frozen Pac-Man against the frozen learned ghost over every scheduled test game, including unwinnable instances. Record wins, losses, ties, denominator, per-map records, Wilson intervals, and training-replicate range. Run five replicate cross-play if runtime permits; disclose fewer replicates. Do not change rules, weaken ghost, or select test seeds to reach 55%. A miss remains a measured miss.

Required behavioral checks cover two-pellet placement/consumption, pre-ghost wins, terminal timeout, one terminal update, variable-duration discounting, lost-sight direction reset, hidden-position invariance for observations/actions/rewards, snapshot alias/mutation rejection, schema rejection, and identical baseline/learned non-search behavior. Model the Domain shaped the explicit ghost lifecycle and pending-decision record, replacing scattered training flags and implicit terminal transitions.
