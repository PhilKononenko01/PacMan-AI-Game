# Phase 1 — Review Log

Findings, rulings and version history for `phase-1-grid-world-prd.md`. The PRD states what is decided; this file records how it got there. Newest round first.

---

## Team ruling — PRD v0.6 → v0.7

**Date:** 2026-10-28
**Revised:** OQ-13. Winnability is scoped to the map generator. It has nothing to do with Pac-Man or the ghost, and the generator never simulates either.

**Ruling:** a map is winnable if it has at most `max_walls` walls (default 6) and at least `min_loops` independent loops in its open cells (default 4), so it offers a choice of routes rather than one path around a central block. Both are game knobs. Other maps are rejected and redrawn.

**Why the v0.6 rule was replaced.** Certifying a win against the ghost made the map sets depend on ghost knobs and `max_steps`, so tuning difficulty would also have changed which maps exist. It left "winnable" open (full or sensed map knowledge, every ghost sequence or likely ones), and it needed a deep search over the ghost's state for every candidate map, at a cost that counts against NFR-4. A structural rule has none of these problems.

**Changes:** §2.3 states the wall cap and loop rule; §2.7 adds `max_walls` and `min_loops` as game knobs and notes that generator knobs change the maps; FR-2 and NFR-7 test the cap and the loop rule, including rejection of the ring around a 3x3 wall block; OQ-13 rewritten; the winnability item is removed from the §7 handoff list; status line updated.

**Defaults checked (simulated, 20,000 draws at `wall_density` 0.20):** about 70% of all draws are connected and pass both rules, and accepted maps average about 3.4 walls, with the empty map about 1.3% of them. The loop floor is the rule that binds most: with at least 4 loops a map can still have up to 8 walls, so the cap of 6 removes the densest of them. The round 1 figure of 1.37 draws per accepted map becomes about 1.44 before the split check.

**Not chosen:** requiring every corridor to have an alternative route (no bridges, so no dead ends) accepted only about 12% of all draws and averaged 2 walls per map, which leaves the maps nearly empty.

---

## Team ruling — PRD v0.5 → v0.6

**Date:** 2026-10-27
**Answered:** OQ-13. The team requires randomly generated maps to be winnable, not merely connected, and approved adding the requirement to the PRD.

**Ruling:** every accepted map must allow Pac-Man to visit all four corners without capture within `max_steps`, accounting for the configured ghost behaviour and turn order. Unwinnable maps are rejected and regenerated. An agent can still lose through its choices.

**Changes:** §2.3 and FR-2 add the winnability requirement; NFR-7 requires coverage of the check, including a connected single-corridor map that cannot be won. OQ-13 records the decision. §7 assigns the verification method, treatment of ghost randomness and bounded generation failure to the architecture doc. Connectivity or loops alone cannot certify winnability, and failure of a baseline agent does not prove a map unwinnable.

**Architecture follow-up:** acceptance must not rely on an unusually lucky ghost sequence. A configuration with `ghost_move_prob=0` cannot produce a winnable map because the ghost permanently occupies a required corner; generation must fail clearly rather than retry forever in configurations that cannot be certified.

---

## Team rulings — PRD v0.4 → v0.5

**Date:** 2026-10-27
**Answered:** OQ-10 to OQ-12 (PRD §6), raised in round 2. Each was answered as proposed. No question is open after this round.

| Question | Ruling | What changed in v0.5 |
|---|---|---|
| OQ-10 Difficulty knobs | Add `ghost_move_prob` (default 1.0): each step the ghost moves with this probability, drawn from its seeded generator. It is the primary knob. Ghost speed and search rule are no longer knobs. | §2.5 move-probability rule (vision and last sighting still update on a step the ghost stays put; its visit record does not); §2.7 knob list; FR-6; NFR-7 tests the stay-put case; OQ-6 note updated. |
| OQ-11 Retraining | The quick check runs the baselines only, beside the stored policy's last validation results and the knobs it was trained under. Retrain when a knob setting is considered final, and before the test maps. | §2.7 tuning loop; FR-15. |
| OQ-12 Evaluation sizes | All 1,000 seeds per split; learning-curve check every 2,500 episodes on the first 200 validation seeds, counted in NFR-4; the final table is evaluated. | §2.8 evaluation sizes; FR-10, FR-12, NFR-4. |

**Consequences checked:**
- *The turn order is unchanged.* A ghost that stays put still triggers §2.4 step 6, which cannot fire because Pac-Man was not on its cell after step 2.
- *Determinism holds.* The move draw comes from the ghost's own generator (ADR-3) and is made every step, including at 1.0.
- *Learning-curve cost is small.* 20 checks of 200 greedy games is 4,000 games against 50,000 training episodes.

---

## Round 2 — PRD v0.3 → v0.4

**Date:** 2026-10-27
**Reviewed:** `phase-1-grid-world-prd.md` v0.3 at commit `39df5a3`, against `Design/project-proposal.md` (P1–P4) and round 1 of this log.
**Method:** a full read for architecture readiness: could the architecture doc be written against this PRD without reopening it? Arithmetic re-checked by hand. The remote branches `agents` and `edits` were checked and hold nothing that is not already on `main`.
**Outcome:** 11 findings. Three need the team and are recorded as open questions OQ-10 to OQ-12, each with a proposal; the rest are fixed in v0.4. Five items are handed to the architecture doc.

### Serious

**R2-01: The game knobs cannot reliably move difficulty, and nothing makes the game easier.** §2.7 lists `wall_density`, `max_steps`, ghost speed and the ghost's search rule. Wall density has no clear direction, `max_steps` only sets when a game becomes a tie, a speed above 1 is undefined and a large jump, and only one search rule exists. Scope step 6 depends on knobs that move difficulty predictably.
*Fix:* recorded as OQ-10, with a proposal to add `ghost_move_prob`. §2.7 points to it.

**R2-02: The difficulty check does not say whether the policy is retrained.** FR-15 runs "the current learned policy", which after a knob change was trained on a different game; a retrain costs up to 10 minutes, which contradicts "judged in minutes".
*Fix:* recorded as OQ-11, with a proposal: the quick check runs baselines only, and a full retrain is done when a knob setting is considered final.

**R2-03: Evaluation sizes and the evaluated policy were unstated.** FR-10 gives no episode count, FR-12 no checkpoint interval or checkpoint size, and nothing says whether the final table or the best checkpoint is evaluated. Learning-curve time also falls under NFR-4.
*Fix:* recorded as OQ-12, with a proposal: 1,000 games per split, a checkpoint every 2,500 episodes on 200 validation seeds, the final table evaluated.

**R2-04: The agent interface could not support training or seeded agents.** "Given an observation, return an action" gave the random agent no seeded generator, had no episode reset, and gave the learner no reward or end-of-episode signal.
*Fix:* §2.8 defines reset, act and observe; FR-8 names them.

### Moderate

**R2-05: The tie bound was wrong.** §2.6 (and the v0.3 rulings below, "Consequences checked") gave the best tie as 3 - 0.05 x 100 = -2. A tie can have reached at most two of the three remaining corners, because the third would have been a win, so the best tie is -3. The ordering win > tie > loss still holds.
*Fix:* §2.6 corrected. The v0.3 entry below is left as written.

**R2-06: Per-step reward accounting was ambiguous.** Whether the winning or losing step pays the step cost, and whether a step ending in a loss can earn a corner reward, were unstated, and NFR-7 tests reward accounting. The round 1 win formula `100 - 0.05k` also left out corner rewards.
*Fix:* §2.6 states the accounting: every step pays the step cost, plus +1 for a first corner visit marked in §2.4 step 3, plus the terminal reward.

**R2-07: The ghost's visit record was underspecified.** §2.5 did not say whether moves made while chasing or pursuing count as visits for the search rule. That changes ghost behaviour, so it affects FR-6 and NFR-1.
*Fix:* §2.5 states that every cell the ghost moves onto counts as visited, in any mode. This is the plain reading of "visited", so it is recorded as a clarification rather than a question.

### Minor

**R2-08: Sensing at step 0 was missing from the turn order.** §2.6 says Pac-Man senses "at the start", but §2.4 listed sensing only after a move. *Fix:* §2.4 adds sensing at reset.

**R2-09: A learned-agent replay needs more than a seed and an agent.** *Fix:* FR-14 adds the configuration and the saved policy file.

**R2-10: The status line was stale.** It cited `178b216` and v0.2. *Fix:* updated to `58b9f9e` and v0.3.

**R2-11: ADR-2 did not name the hidden ghost mode.** The state does not show whether the ghost is chasing, pursuing or searching, or where it last saw Pac-Man, so states that look the same can need different moves. *Fix:* ADR-2 names it as a risk; the learning curve is where a failure would show.

### Handed to the architecture doc

Listed in PRD §7: the exact bytes hashed for the split; how generators are derived from the episode seed, with no dependence on Python string hashing or set order; which training seeds are used; the saved-policy format; the command-line entry points.

### Checked and sound

- Table size: 4 x 16 x 80 x 4 x 2 x 3 = 122,880 rows, about 3.9 MB at four 64-bit values per row.
- Every unvisited corner is always reachable on the remembered map: known walls are real walls and unknown cells count as open, so the true map's path is always a candidate. No "no path" case exists for the pathfinding agent or the ADR-2 features.
- The turn order still prevents the two passing through each other unseen.
- The split-by-layout scheme and the paired, one-sided McNemar test are sound.
- NFR-4 at 50,000 episodes looks comfortable (on the order of 1.5M steps, each with a few small breadth-first searches), subject to OQ-12's learning-curve cost and the architecture doc's measurement.

---

## Team rulings — PRD v0.2 → v0.3

**Date:** 2026-10-26
**Answered:** OQ-1 to OQ-8 (PRD §6). No question is open after this round.

| Question | Ruling | What changed in v0.3 |
|---|---|---|
| OQ-1 Start positions | As proposed: Pac-Man `(0, 0)`, ghost `(4, 4)`, start corner visited. | Nothing; marked answered. |
| OQ-2 Pac-Man's knowledge | Changed from the proposal: Pac-Man senses only the four cells next to him, remembers what he has sensed, and plans as if unknown cells are open. He still always knows where the ghost is. | §2.6 sensing, memory and optimistic planning; §2.4 adds a sensing step; §2.8 observation carries the remembered map and the pathfinding agent re-plans each step; ADR-2 features use the remembered map; new FR-16; NFR-7 tests that no observation leaks an unsensed wall. |
| OQ-3 Wall type | As proposed: whole-cell walls. | Nothing; marked answered. |
| OQ-4 Turn limit | Hitting `max_steps` is a **tie**, reported alongside wins and losses. An idea to end the game when Pac-Man's score goes negative was raised and dropped: the step cost is only meant to keep him moving. | "Timeout" renamed "tie" throughout; FR-10, FR-11 and §5 report and test ties. |
| OQ-5 Ghost explore | As proposed: pursue the last sighting, then visit least recently visited cells. | Renamed "search" for clarity. |
| OQ-6 Ghost speed | As proposed: same speed. | Faster-ghost note moved to §2.7. |
| OQ-7 Corner reward | +1 for the first visit to each corner (the start corner gives nothing). | §2.6 reward table and rationale. |
| OQ-8 Visuals | Watched games are animated as pixel art in the terminal; simulations have no graphics and end with a summary of wins, losses and ties. No window, no human play. | New §2.9; FR-13 and FR-14 rewritten for watch mode; new FR-17 for the simulation summary; non-goals updated. |

**Consequences checked:**
- *Reward ordering still holds.* The best possible tie scores 3 - 0.05 x 100 = -2, so any win scores far above any tie, and any tie scores far above any loss.
- *Partial knowledge makes two features optimistic.* With unknown cells counted as open, the ghost's path distance can look shorter than it is, and the line-of-sight feature can say "seen" when an unsensed wall actually blocks the ghost. ADR-2 states this. It is accepted: it errs toward caution.
- *The pathfinding baseline still exists.* It plans on the remembered map and re-plans when it senses a wall, so P4's comparison still holds.
- *No new dependency.* Terminal pixel art uses ANSI colour codes and Unicode block characters from the standard library (ADR-1).

---

## Round 1 — PRD v0.1 → v0.2

**Date:** 2026-10-26
**Reviewed:** `phase-1-grid-world-prd.md` v0.1 at commit `be5efa8`, against `Design/project-proposal.md` (P1–P4).
**Method:** a full read for logical errors, internal contradictions and unsupported claims. Finding R1-01 was checked by simulating the v0.1 map generator (200,000 training seeds, 1,000 test seeds, `wall_density` 0.20).
**Outcome:** 13 findings. All fixed in v0.2. Team rulings on four of them are recorded under each finding.

### Serious

**R1-01: Test maps were not held out.** v0.1 split *seeds* into disjoint ranges and called the test maps held out (§2.7, NFR-6). A 5x5 grid at 20% walls has few likely layouts, so different seeds often make the same map. The simulation found 932 of 1,000 test maps among the first 200,000 training seeds, and only 868 distinct maps in the test set. The NFR-6 seed test would have passed while most test maps were training maps.
*Ruling (team, 2026-10-26):* separate the maps. *Fix:* §2.7 assigns each map to a split by the SHA-256 hash of its layout, modulo 100 (training 0–79, validation 80–89, test 90–99); the generator accepts only maps in the requested split (§2.3, FR-2); NFR-6 and NFR-7 test split membership. Removing seen maps from the test set was rejected, because it would leave a test set skewed toward rare, wall-heavy maps.
*Residual:* the most common layouts (the empty map is about 1% of draws) fall in one split only, so the splits differ slightly at the very top of the distribution. This is accepted.

**R1-02: The step-cost reason was wrong, and a timeout scored like a loss.** v0.1 said that without a step cost "a policy that hides until timeout scores the same as one that loses slowly". It does not: hiding scores 0 and losing scores -100. With the v0.1 step cost of -1 and `max_steps` 100, a timeout totalled -100, almost the same as a loss, so the agent had almost no reason to prefer surviving to being caught.
*Ruling (team, 2026-10-26):* staying alive costs -0.05 per step, so Pac-Man has a reason to keep moving. *Fix:* §2.6 reward table and rationale; outcomes are now ordered win (`100 - 0.05k`) > timeout (-5) > loss (`-100 - 0.05k`). The discount factor is set at 0.99 in §2.8.

**R1-03: The learned agent could not tell where the ghost was.** ADR-2 gave it the ghost's path distance and the ghost's next step, but not which of Pac-Man's own moves lead toward the ghost. It could not tell whether its route to a corner ran into the ghost, which is the decision it is meant to learn. v0.1 also claimed "the same state means the same situation on every map", which a feature summary cannot promise.
*Ruling (team, 2026-10-26):* Pac-Man knows where the ghost is on the board at all times (P2). Recorded as OQ-9, answered. *Fix:* ADR-2 replaces the ghost's-next-step feature with the ghost's row and column offset from Pac-Man (80 values). The table grows from 6,144 to 122,880 rows (about 4 MB), most of which cannot occur. The overclaim was replaced with a statement that the state is a summary. ADR-2 now also says that the agent only sees the nearest unvisited corner, so it cannot plan a different corner order.
*Watch:* with the larger table, the learning curve (FR-12) has to show that training visits the reachable states often enough. If it does not, the episode count (§2.8) is the first lever.

### Moderate

**R1-04: No training episode count, so NFR-4 could not be checked.** *Fix:* §2.8 adds learner settings with defaults (learning rate 0.1, discount 0.99, linear epsilon decay from 1.0 to 0.05 over the first half, 50,000 episodes). NFR-4 is tied to the default count, and the architecture doc measures the run time before the count is fixed. A rough estimate put 1M episodes in pure Python at 30–60 minutes, over the NFR-4 limit.

**R1-05: Reward tuning was counted as difficulty tuning.** The reward table was a "difficulty knob", but rewards change how well the agent learns, not how hard the game is. Tuning them to put the win rate in the band would partly measure the tuning. *Fix:* §2.7 splits game knobs (the only ones turned for difficulty) from learner settings. NFR-2 covers both.

**R1-06: "Identical games" was overstated.** The ghost reacts to Pac-Man, so agents play different games from the same start. *Fix:* FR-10 and FR-11 say "same starting conditions". The paired test is still valid.

**R1-07: The McNemar test had no binary outcome or direction.** *Fix:* FR-11 specifies an exact, one-sided test on win versus not-win (loss and timeout count as not-win), paired by episode seed. §5 item 3 notes that both tests must pass, so no correction for two tests is needed.

**R1-08: The status line understated what depends on open questions.** It named only FR-2 through FR-6. *Fix:* the v0.2 status line names each dependency: FR-1 and FR-2 (OQ-3), FR-3 (OQ-1), FR-5 and FR-6 (OQ-5, OQ-6), FR-7 (OQ-7), FR-10 (OQ-4), ADR-2 (OQ-2) and the non-goals (OQ-8).

### Minor

**R1-09: Several ties had no tie-break rule, which broke NFR-1.** *Fix:* nearest-corner ties use the fixed order `(0, 4)`, `(4, 0)`, `(4, 4)`, and path and greedy-action ties use the direction order (§2.8). Never-visited cells count as least recent for the ghost's explore rule, and its start cell counts as visited (§2.5). The timeout is checked on the step whose number equals `max_steps` (§2.4). NFR-7 covers all tie-break rules.

**R1-10: "Adjacent cells always see each other" added nothing,** because the row/column rule already covers it, and it invited a diagonal reading. *Fix:* removed from §2.5.

**R1-11: OQ-6's claim that an equal-speed ghost "can usually be outrun" was untested,** and its suggested fix, a double ghost move, breaks the turn order in §2.4. *Fix:* OQ-6 now says the baselines will tell, and that a faster ghost needs a collision check after each sub-move. The unsupported claim in §2.2 that entering the ghost's start corner is "a large part of the difficulty" was also removed.

**R1-12: The gate did not say what had to fall in the 50–90% band.** *Ruling (team, 2026-10-26):* the band is a rough guideline, not a hard line. *Fix:* §5 item 2 reports the win rate with its 95% interval against the target and discusses a result outside it rather than failing it. The DQN fallback (ADR-2, non-goals) now opens on failing to beat both baselines (§5 item 3), since the band no longer fails anything.

**R1-13: Learner settings were unpinned, so NFR-1 could not hold for training results.** *Fix:* folded into R1-04; NFR-1 now names learner settings.

### Checked and sound

- The 6,144-state count in v0.1 was correct arithmetic. The v0.1 raw-coordinate count was 25 x 25 x 16, but the start corner is always visited, so it should be 25 x 25 x 8 = 5,000; corrected in ADR-2.
- The sequential turn order does prevent the two passing through each other unseen.
- Separate seeded generators per component (ADR-3) give reproducible, paired games.
- The map rejection rule is cheap: about 1.37 draws per accepted map at the default density (simulated).
- Requiring both McNemar tests to pass needs no correction for running two tests.

### Still open

OQ-1 through OQ-8 (PRD §6) awaited the team at the end of this round. They were answered the same day; see Team rulings above.

---

## Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-10-26 | First draft (`be5efa8`). |
| v0.2 | 2026-10-26 | Review round 1 fixes (R1-01 to R1-13); OQ-9 added and answered (`29928fa`). |
| v0.3 | 2026-10-26 | OQ-1 to OQ-8 answered; partial wall knowledge, ties, corner reward, watch and simulation modes (`39df5a3`). |
| v0.4 | 2026-10-27 | Review round 2 fixes (R2-04 to R2-11); OQ-10 to OQ-12 opened with proposals; architecture-doc handoff list in §7 (`80fcb4f`). |
| v0.5 | 2026-10-27 | OQ-10 to OQ-12 answered; `ghost_move_prob`, no retrain in the quick check, evaluation sizes. |
| v0.6 | 2026-10-27 | OQ-13 answered; random maps must be winnable under the configured ghost rules and turn limit; verification, ghost randomness and bounded failure handed to architecture (`835ba30`). |
| v0.7 | 2026-10-28 | OQ-13 revised: winnability is a map-generator rule (`max_walls`, `min_loops`), independent of Pac-Man and the ghost. |
