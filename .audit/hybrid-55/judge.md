# Cross-judge verdict

Use **Astra as the base**, retaining its separate `game.py`, `ghost.py`, `learning.py`, `artifacts.py`, and `runner.py`. Graft Sol's precise turn notification and frozen-policy contract, and Terra's compact feature/coverage and replay evidence. The top three are close; Astra wins on concrete bounded-search constants and clean ownership, rather than a fundamentally different architecture. All correctly prefer an opt-in subsystem over migrating historical rules.

Scores are design completeness, 1–5; they are not empirical performance predictions.

| Candidate | Visibility/no hidden leak | Pending/terminal learning | Preserve main/history | Small interfaces/time to evidence | Unfiltered/reproducible eval | Total |
|---|---:|---:|---:|---:|---:|---:|
| Astra | 5 | 4 | 5 | 5 | 5 | 24 |
| Sol | 5 | 5 | 5 | 4 | 5 | 24 |
| Terra | 5 | 5 | 5 | 4 | 5 | 24 |
| Luna | 4 | 2 | 5 | 3 | 4 | 18 |

## Required corrections before implementation

1. **Make notification timing executable, not implied.** Replace Astra's ambiguous `reward(value, terminal)` contract with one per-completed-game-turn notification, including terminal turns on which the ghost never acts. A pending search decision stores elapsed game turns and discounted return. At the next decision, close the old transition before opening the new one. At terminal, close once with no bootstrap and clear it. Define the first reward after choosing as exponent zero. Use Sol's lifecycle clarity but avoid both `observe_turn` and `finish` performing rewards/updates. Capture on Pac-Man's move must earn +1 for an existing pending search decision too. Luna explicitly misses this case.

2. **Keep last observed direction distinct from consecutive-visibility tracking.** An invisible observation invalidates the preceding-visible sample for new direction inference, but need not erase a direction already established in the retained sighting. Reacquisition refreshes the sighting with unknown direction until two consecutive visible positions permit inference. Terra's wording about clearing direction on lost sight risks deleting useful legitimate memory. Never derive direction from hidden moves. The controller receives only static map, sanitized observation, its RNG, and scalar outcome rewards.

3. **Fix search geometry and expiration in rules.** Adopt Astra's world-direction targets, length 3, memory age 12, at most 3 choices and 4 moves per target, with precise counting documented before training. Expire on elapsed hidden ghost observations regardless of movement. Exclude zero-length, duplicate and exhausted targets. Empty candidate sets must finish search without an arbitrary illegal action. Define whether budget exhaustion clears memory immediately or waits for age expiry; prefer immediate clearing into shared random wandering. Baseline and learned ghosts differ only in candidate selection. Do not learn wandering as Luna suggests.

4. **Make snapshots actually immutable.** Use Sol's copied immutable storage/read-only views or immutable sparse records, not merely a frozen dataclass around a mutable dictionary or owning NumPy array. Keep controller memory and RNG separate. Evaluation takes frozen actors only; reject rules/feature/role mismatches and assert content hashes around opponent training/evaluation.

5. **Bound feature cardinality before large runs.** Graft Terra's explicit compressed feature buckets and visit/zero-visit metrics. Sparse tuple keys alone do not solve poor coverage. The Pac-Man encoder must receive no true map; any path feature uses its remembered map under a fixed unknown-cell convention. Ghost feature cardinality and legal-action masks should be equally explicit. Start with a short end-to-end training/replay run and measure throughput before choosing full episode budgets. No candidate can guarantee 55% from design alone.

6. **Use one unambiguous entry point and authoritative loop.** Prefer `python -m pacman.hybrid train|evaluate|watch` to minimize legacy CLI edits, or Astra's single top-level `hybrid` dispatch if integration is desired. Do not overload legacy commands with scattered `--rules` branches. Training, evaluation and watch must share episode logic and horizon resolution. No historical artifact/report edits.

7. **Be precise about evaluation units.** Store the full schedule, seeds, pellet placements or replay keys, outcomes, turns and hashes. Count every scheduled game, including unwinnable cases, with ties in the denominator. Standard Wilson intervals describe binomial game counts; if multiple stochastic games share each map, also disclose clustering and per-map results rather than calling an ordinary game-level Wilson interval a map-level interval. Select using validation, freeze the result, then evaluate test once per declared protocol. Cross-play and five seeds are useful extensions, not reasons to defer first valid evidence. Report actual completed replicate counts and any target miss.

## Gates

Before full training require behavioral tests for hidden-position invariance (observations, actions and nonterminal reward), consecutive-visibility direction, memory expiry/empty candidates, two unique pellets/collection once, both capture timings, pre-ghost win, terminal turn 100, exact-once finalization, variable-duration discounting, pending transitions surviving reacquisition, frozen-opponent integrity, schema rejection, deterministic replay, and baseline/learned parity outside search selection. Run existing tests unchanged. Smoke all three commands against newly written artifacts.

Astra's separation plus these specific grafts is the smallest clear route to an honest measured result. Luna should not be the implementation base because capture credit and variable-duration transitions are materially underspecified or wrong.
