# Phase 1 — Review Log

Findings, rulings and version history for `phase-1-grid-world-prd.md`. The PRD states what is decided; this file records how it got there. Newest round first.

---

## Round 1 — PRD v0.1 → v0.2

**Date:** 2026-10-26
**Reviewed:** `phase-1-grid-world-prd.md` v0.1 at commit `8a1c699`, against `Design/project-proposal.md` (P1–P4).
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

OQ-1 through OQ-8 (PRD §6) await the team.

---

## Version history

| Version | Date | Change |
|---|---|---|
| v0.1 | 2026-10-26 | First draft (`8a1c699`). |
| v0.2 | 2026-10-26 | Review round 1 fixes (R1-01 to R1-13); OQ-9 added and answered. |
