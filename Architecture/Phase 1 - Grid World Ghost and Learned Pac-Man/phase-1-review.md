# Phase 1 — Review Log

Findings, rulings and version history for `phase-1-grid-world-prd.md` and `phase-1-architecture.md`. The phase documents state what is decided; this file records how it got there. Newest round first.

---

## Issue #18 — Final acceptance: measurements, knob freeze, held-out evaluation

**Date:** 2026-10-29
**Documents:** PRD v0.11 §5, NFR-3, NFR-4; architecture v0.3 §17 to §19. Branch `feature/18-acceptance`, stacked on `feature/17-cli`. Report: `reports/phase-1-results.md`.

**R-09, R-10 (reviewer findings, fixed):** pixel-art watch mode printed frames after a blank line, so a terminal scrolled instead of animating; it now homes and clears the screen before each frame, and plain mode keeps scrolling. `PACMAN_DEBUG=1` makes `main` re-raise instead of printing `error: ...`.

**Added:** `tests/test_determinism.py` (identical records twice in-process, in subprocesses under two `PYTHONHASHSEED` values, and a byte-identical table from two 200-episode trainings) and `tests/test_integration.py` (train, save, load, evaluate all agents, write and read the JSON, replay every record to the same ruling, step and return; the learned agent never asked to act after a ruling and starting every game with an empty cache). README rewritten for the implemented commands. Suite: 472 tests, about 12 s idle.

**Measured:** the default 50,000-episode training run took 234.6 s wall (NFR-4 target 600 s) on an Intel i5-8259U with Python 3.14.10, NumPy 2.5.4, Matplotlib 3.11.2. Final validation on all 1,000 seeds: learned 691 wins (69.1% [66.2, 71.9]), 267 losses, 42 ties, 37.4 steps per win; pathfinding 142 wins, random 0; McNemar b=691/c=0 and b=570/c=21, both p far below 0.05. The learning curve passes 50% at 22,500 episodes and ends at 72.5% on the 200-seed checkpoint set.

**Knob freeze (PRD §5), awaiting sign-off:** the proposal is the unchanged PRD defaults for every game knob and learner setting; nothing was tuned on validation. The policy is `reports/phase-1/policy.npz`. No test-split game has been played. Christian and Phillip record their approval here, after which `evaluate --split test` runs once and the report's §5.3 is written from it without further changes to the policy or settings.

---

## Issue #17 — The five CLI commands, agent factory and replay flow

**Date:** 2026-10-29
**Documents:** architecture v0.3 §4.3, §14. Branch `feature/17-cli`, stacked on `feature/15-train`.

**Implemented:** `agents/__init__.py` with `AGENT_NAMES` and `make_agent(name, config, policy=None)` (the learned agent needs a policy and is always greedy from the factory; imports are lazy so `agents.base` stays light). `__main__.py` for `python -m pacman`. `cli.py` with `parse_args`, `load_config(path, sets)` and `apply_sets` (a `--set NAME=VALUE` finds the field by bare name in either section, parses by the field's type so `max_steps=5.0` is refused, and rebuilds the frozen `Config` so range checks run), `policy_config` (a policy's configuration with overrides; any differing game knob is an error naming it), and the five commands exactly as §14.1 to §14.5: `train` refuses an existing run before training, prints each curve point as it is made and ends with the §13.3 row for the final validation; `evaluate` plays all three agents on every seed of the split, prints the report with both McNemar lines and writes `eval-<split>.json` beside the policy; `check` prints the baselines and, with a policy, its stored validation row and the knobs with `*` on each that differs; `simulate` prints the requested summaries; `watch` plays one game through `play_episode`'s frame callback with `use_plain`'s fallback and sleeps `--delay` (default 0.25 s) between frames, and nothing else. `main` turns any exception into `error: ...` on standard error and exit 1.

**Tests:** `test_cli.py` (22), all with the sizes lowered and `episodes=5`: `python -m pacman --help` lists the commands; the factory; `--set` on top of a file, rejected names and values; the default delay; `train` output and the four files; refusal of an existing run; `evaluate`'s report and JSON with every agent on the same seeds, a differing game knob refused and a learner setting allowed; `check` with and without a policy, knobs flagged, the policy file untouched; `simulate` rows, split bounds, the policy requirement; a `MapGenerationError` as a clean error; `watch` frames one per step with the ruling last, plain text when output is not a terminal, an unknown seed refused, and replaying an evaluated game from `eval-val.json` to the same outcome and step count.

**Timing:** 465 tests in about 12 s.

---

## Issue #15 — Training, validation checkpoints and learning-curve outputs

**Date:** 2026-10-29
**Documents:** architecture v0.3 §11, §12, §14.1. Branch `feature/15-train`, stacked on `feature/14-evaluate`.

**Implemented:** `train.py` with `CURVE_EVERY = 2_500`, `CURVE_GAMES = 200`, `epsilon_for(episode, settings)` (the §11.1 linear schedule), `train(config, on_progress=None) -> TrainingResult(policy, curve, validation)`, `prepare_output_dir` and `write_outputs`. Episode `k` plays training seed `k` with its epsilon set first. A second `LearnedAgent` in evaluation mode shares the learner's table and plays every curve checkpoint and the final validation greedily, so the table is never written by an evaluation game and there is no training mode to restore, the arrangement §11.1 explicitly allows. The curve point is `(episodes completed, win rate)`; `FULL_GAMES` is read from `evaluate` at call time so tests can lower it. The policy carries the final table, the configuration, `(0, episodes - 1)`, the curve and the final validation summary over the seeds actually played. `write_outputs` refuses a directory that already holds `policy.npz` before writing anything, then writes `policy.npz`, `curve.csv` (`episodes,win_rate`), `curve.png` and `config.json`. `plot.py` is the only Matplotlib importer, selects Agg before `pyplot`, closes its figure, and is imported lazily by `write_outputs` so `import pacman.train` loads no Matplotlib.

**R-08 (reviewer finding, fixed on `feature/14-evaluate`, commit `04585ff`):** `run_split` took the agent as given, so a learned agent left in training mode would have learned from the seeds it was evaluating, on the test split silently. It now raises `ValueError` for an agent whose `training` is true; one test on the test split proves nothing is learned and that the same agent runs once `training` is off.

**Tests:** `test_train.py` (18): the constants; epsilon at the start, midway, the decay end and after it, and under other settings; episode `k` on seed `k` with its epsilon, in training mode; checkpoints at the right episodes on validation seeds, greedy, table unchanged; training continuing after a checkpoint and the last table saved; a checkpoint evaluation that raises propagates while the learner stays in training mode and still learns; the final validation stored with its seed range; the progress callback; an empty curve for a run shorter than a checkpoint; repeatability; the four outputs with exact `curve.csv` text and a loadable policy; refusal of an existing `policy.npz`; directory creation. `test_plot.py` (3): a PNG for a curve and for an empty curve; a subprocess check that `train`, `evaluate` and `policy_io` load no Matplotlib and `plot` selects Agg.

**Timing:** the suite is 443 tests in about 17 s here (the reviewer measures about 23 s on its run). Matplotlib's first import in a fresh environment builds a font cache once, which took 44 s; it does not recur.

---

## Issue #14 — Paired evaluation, simulation summaries and the difficulty check

**Date:** 2026-10-29
**Documents:** architecture v0.3 §13, §14.3, §14.4. Branch `feature/14-evaluate`, stacked on `feature/11-learned`.

**Implemented:** `evaluate.py` with `FULL_GAMES = 1_000`; `run_split(config, agent, split, games)` playing the first `games` seeds of the split in seed order through one environment and referee, with the agent exactly as given (callers pass a greedy learned agent; nothing branches on the agent); `summarise(agent, records) -> Summary` and `summary_of_validation(validation)` for the summary a policy file stores; `compare(learned, baseline) -> Comparison(b, c, p)` paired by seed through `stats.discordant_pairs` and `mcnemar_one_sided`; `format_header`, `format_summary_row`, `format_comparison` and `format_report` producing the §13.3 text; `write_eval_json` / `read_eval_json` with the configuration and every record (FR-14); `knob_differences(current, stored)`; `difficulty_check(config, policy=None)` playing the two baselines on the validation split and, with a policy, returning its stored validation summary and the game knobs that differ, never training (FR-15); `simulate(config, agents, split, games)` (FR-17). Printing belongs to the CLI (#17).

**A14-01 (cosmetic, kept):** the §13.3 example rows give the losses count one more column than the wins and ties counts. The rows are reproduced verbatim, widths 5, 6 and 5, so the documented example is exactly what the command prints.

**R-06 and R-07 (fixed by the reviewer session, commits `ea07cd3` and `04c7990`):** in evaluation mode `LearnedAgent.observe` returned before refreshing the cached state, so every greedy `act` after the first reused the first observation's row; training was unaffected and the #11 tests passed because the greedy test acted twice on one observation. The agent now keeps the pending transition apart from the cache for the next `act`, with two regression tests. `policy_io` was hardened without interface change: `Policy` is `eq=False` (value equality crashed on the array field), metadata is parsed with `config.py`'s duplicate-key and NaN rejection, a boolean `format_version` is refused, a non-archive file raises `ValueError` naming the path, and `save_policy` writes `<name>.partial` then `os.replace`s it; six regression tests.

**Tests:** `test_evaluate.py` (20): seed order and identical seeds per agent, records equal to direct `play_episode` results, split bounds, a greedy learned agent's table unchanged; summary counts, intervals and steps, no wins, empty input; pairing by seed independent of record order, seed-set mismatch; the two §13.3 example rows verbatim; header, comparison and report assembly; JSON round trip with every field; knob differences; the difficulty check with `FULL_GAMES` lowered, with and without a policy, table untouched; simulation per agent; and the full 1,000-seed validation run reproducing random 0 and pathfinding 142 wins with 13.6 steps per win.

---

## Issue #11 — Tabular Q-learning, epsilon-greedy actions and per-game cache reset

**Date:** 2026-10-29
**Documents:** architecture v0.3 §10.1, §10.5, §16 D-1; foundation handoff (per-game cache reset, explicit end signal). Branch `feature/11-learned`, stacked on `feature/10-baselines-features`.

**Implemented:** `agents/learned.py` with `LearnedAgent(settings, q=None, training=False)`. The table is a zero `float64` `(122880, 4)` array or a shared one of that shape (`policy_io.TABLE_SHAPE`); `training` and `epsilon` are public fields for the training loop. `act` reuses the index cached by the last continuing `observe`, else encodes the observation; in training it draws `u` and explores with `pick(rng, 4)` when `u < epsilon`, otherwise and always outside training it takes `numpy.argmax`, whose first maximum is the direction-order tie-break. `observe` does nothing outside training; in training the target is `reward` at `WIN` or `LOSS` (the final observation is never encoded), `reward + discount * max Q[s']` at `None`, and at `TIE` the same bootstrap under the confirmed default or `reward` alone under `"terminal"`; then `Q[s, a] += learning_rate * (target - Q[s, a])`. `s'` is cached only while the game continues; every final outcome clears the cache, `reset` clears it again for the next game and keeps the table, and an `observe` with nothing pending raises `RuntimeError`.

**Tests:** `test_learned.py` (19): protocol and zero table; shared table and shape checks; greedy first-best action with no draws and no learning; exploration on a second draw, strict `<` on epsilon; update arithmetic on a continuing step and with other settings; win and loss using the reward alone on observations that cannot be encoded; tie bootstrap by default and terminal by setting; the cached next index reused by the next `act`; no transition surviving any final outcome or a `reset`; whole training games changing the table while evaluation games do not; byte-identical tables from two 25-episode trainings; no draws from the episode stream in evaluation.

---

## Issue #10 — Baseline agents and the shared state feature encoder

**Date:** 2026-10-29
**Documents:** architecture v0.3 §10.2 to §10.4, PRD ADR-2 and §2.8. Branch `feature/10-baselines-features`, stacked on `feature/12-policy-io`.

**Implemented:** `agents/random_agent.py` (`RandomAgent`: `Direction(pick(rng, 4))`, one draw per step); `agents/pathfinding.py` (`CORNER_ORDER`, `remembered_open(observation)` as Pac-Man's passable predicate, `plan(observation) -> Plan(target, step, distances, remaining)` and `PathfindingAgent`, which replans from the latest memory every step and never reads the ghost); `agents/features.py` (`Features` with the six ADR-2 fields and an `index` property, `features(observation)`, `state_index(observation)`, `STATES = 122_880`). `features` raises on a capture or when no corner remains, so terminal observations are never encoded. `agents/__init__.py` is untouched for #17.

**Measured on the 1,000 validation seeds at the defaults:** random 0 wins, 1,000 losses; pathfinding 142 wins (14.2% [12.2, 16.5]), 858 losses, 0 ties, 13.6 mean steps per win. These are exactly the §13.3 figures from the exploratory probe, so the generator, ghost, environment, loop and baselines reproduce it end to end. Both runs took 4.2 s together (the architecture budgeted about 6 s).

**Tests:** `test_baselines.py` (16): protocol conformance; the random agent's draws match `pick` on its stream, one per step, all four actions used; nearest-corner choice, fixed corner order, direction-order path ties, routing round remembered walls with unknown as open, replanning after a sensed wall, the ghost ignored, deterministic whole games, and the still-ghost loss on the nearest corner. `test_features.py` (26): the start observation's features and index by hand (44,156), the extreme features filling the table exactly, `step` from the plan, `open_mask` from known-open in-grid neighbours only, every offset 0 to 79 used exactly once with the centre skipped, distance buckets on the remembered map, sight on the remembered map, remaining corners, captures and finished games refused, immutability, and every observation of 30 random games indexing inside the table.

---

## Issue #12 — Versioned policy save/load and replay metadata

**Date:** 2026-10-29
**Documents:** architecture v0.3 §12. Branch `feature/12-policy-io`, stacked on `feature/13-stats`.

**Implemented:** `policy_io.py` with `FORMAT_VERSION = 1`, `TABLE_SHAPE = (122_880, 4)`, the frozen dataclasses `ValidationSummary(seeds, games, wins, losses, ties, mean_steps_to_win)` and `Policy(q, config, training_seeds, learning_curve, validation)`, and `save_policy(path, policy)` / `load_policy(path)`. The file is exactly §12: a `float64` `q` and a `uint8` `meta` holding UTF-8 JSON with the five documented keys, written with `numpy.savez` into an opened file so the path is used as given, read with `allow_pickle=False`. Both dataclasses validate on construction: table shape and dtype, seed ranges in order, counts summing to `games`, and `mean_steps_to_win` null exactly when there are no wins.

**Persistence interface for #11, #14, #15 and #17:** build a `Policy` from the final table, the `Config` used, `(0, episodes - 1)`, the curve as `(episodes, win_rate)` pairs and a `ValidationSummary` of the final 1,000-seed greedy run, then `save_policy`. `load_policy` returns the same `Policy`; `policy.config` is a full `Config`, so `evaluate` and `check` compare knobs with `policy.config.game`. `TABLE_SHAPE` is the one definition of the table's shape; `features.py` (#10) should assert its state count against it.

**Tests:** `test_policy_io.py` (29): round trip of table, config and metadata; the file holds exactly the two arrays with the documented dtypes and JSON; null mean for zero wins; empty curve; a pickled array is refused (proving `allow_pickle=False`); unknown version, wrong shape or dtype, each missing top-level and validation key, a missing `meta` array, invalid or unknown configuration keys, and malformed JSON all raise `ValueError` naming the problem; inconsistent metadata is refused at construction; non-finite values cannot be saved; `policy_io` loads only `config` from the package.

---

## Issue #13 — Wilson intervals and paired exact McNemar statistics

**Date:** 2026-10-29
**Documents:** architecture v0.3 §13.2. Branch `feature/13-stats`, stacked on `feature/16-render`. Claimed by Christian as the shared ticket the tracker pairs with #4.

**Implemented:** `stats.py` with `Z_95 = 1.959963984540054`, `wilson(k, n, z=Z_95)`, `mcnemar_one_sided(b, c)`, `discordant_pairs(learned_won, baseline_won)` and `mean_steps(steps)`. Pure functions on counts; the module imports nothing from the game. `wilson` returns exact `0.0` and `1.0` bounds for `0/n` and `n/n` and clamps to the unit interval otherwise; `n = 0`, `k > n`, negatives, floats and booleans raise `ValueError`. `mcnemar_one_sided` sums binomial coefficients exactly with `fractions.Fraction` and converts to float once; `p = 1.0` when `b + c = 0`. `discordant_pairs` takes per-seed win flags for the two agents and requires the same seed set, so #14 only reduces records to win or not-win. `mean_steps` returns `None` where the report prints `n/a`.

**Measured:** the §13.3 example intervals reproduce at 0.1% precision (0/1000 → [0.0, 0.4]; 142/1000 → [12.2, 16.5]; 858/1000 → [83.5, 87.8]; 1000/1000 → [99.6, 100.0]).

**Tests:** `test_stats.py` (36): the report's intervals, a textbook 5/10 value, exact end bounds, symmetry, coverage of the observed rate over every `k` for several `n`, a custom `z`, rejected counts; McNemar hand-computed values in both directions, exactness against `Fraction`, 1000 discordant pairs without overflow, rejected counts; discordant-pair counting and seed-set mismatch; mean steps; a subprocess check that `stats` loads no other game module.

---

## Review of the environment lane (#5 to #9, #16)

**Date:** 2026-10-29
**Documents:** architecture v0.3 §4.1, §6, §7, §8, §9. Branch `feature/16-render`.

**Reviewed:** every module of the lane against the PRD and architecture tables, plus a profile of the real episode loop (about 0.5 ms per random-agent episode; the ghost's per-step BFS is the only measurable cost). No behaviour of an accepted map, a played game or a golden value changed; the fixes below make caller errors fail loudly instead of silently.

**R-01 (fixed):** `grid.move` indexed the delta table directly, so an action of `-1` wrapped to `LEFT` while `4` raised `IndexError`. Any value that is not a `Direction` now raises `ValueError`.

**R-02 (fixed):** `GameMap` accepted walls on corners or off the grid, so a hand-built test map could break the generator's guarantees without warning. Both now raise `ValueError`; `GameMap.is_open` is the one passable predicate for the true map and replaces four inline copies.

**R-03 (fixed):** `generate(seed, split, ...)` trusted the split it was given. It now raises `ValueError` unless `split` is `split_of_seed(seed)` (§4.2); every existing caller already passed the seed's own split.

**R-04 (fixed):** `is_winnable` and `loop_count` are only correct for connected maps. `is_winnable` now raises `ValueError` on a disconnected map; `generate` keeps a single connectivity check per candidate.

**R-05 (fixed):** `first_step` with `source == target`, and a ghost with no open neighbour, each raise a `ValueError` naming the precondition instead of an `AssertionError` or a bare `min()` error.

**Hygiene:** `Environment` declares its episode fields in `__init__`; `play_episode` and `_frame` are typed; the renderer's memory lookup table is a module constant; two tests use `ast.literal_eval` rather than `eval` on subprocess output; one tautological assertion in `test_ghost.py` removed.

**Tests:** 13 added across `test_grid.py` and `test_mapgen.py` for R-01 to R-04; 267 pass in about 3.4 s.

---

## Issue #16 — Terminal pixel-art and plain-text rendering

**Date:** 2026-10-29
**Documents:** architecture v0.3 §14.6, D-5. Branch `feature/16-render`, stacked on `feature/9-referee-episode`.

**Implemented:** `render.py` with `plain_frame(frame)`, `pixel_frame(frame)`, `render_frame(frame, plain)` and `use_plain(plain, isatty, environ)`. Both renderers take only the #9 `Frame` and return a string; nothing is written or slept. The plain grid uses the §14.6 characters; the pixel grid draws each cell as a 4x4 sprite over two `▀` lines with `ESC[38;5;Nm` / `ESC[48;5;Nm` colours and a reset at each line end. Walls and unknown cells come from memory (D-5); corners and actors are drawn where they are. The status line is `Move t/max_steps  Corners n/4  Ghost sees Pac-Man: yes|no`, with `Outcome: win|loss|tie` added on the last frame.

**A16-01 (minor, decided):** §14.6 does not say which actor shows when both share a cell, which happens on every loss frame. The ghost is drawn over Pac-Man, so the final frame shows the catch; actors are drawn over corners and corners over memory.

**A16-02 (minor, decided):** "when `NO_COLOR` is set" follows the no-color.org convention: set to a non-empty value. An empty `NO_COLOR` does not force plain text.

**Tests:** `test_render.py` (20): a hand-built plain grid and status line; the ruling on the last frame only; overlay precedence; known versus unknown walls; the four fallback rules; pixel frames with exactly ten 20-character half-block lines after stripping the three allowed escape forms, the same status text as plain, no other control characters, and seven distinguishable cell kinds; dispatch on the plain flag; nothing written to standard output.

---

## Issue #9 — Referee and the shared episode loop

**Date:** 2026-10-29
**Documents:** architecture v0.3 §7.4, §10.6, §14.6. Branch `feature/9-referee-episode`, stacked on `feature/8-environment`.

**Implemented:** `referee.py` with `Referee(max_steps)`: `start`, `next_move`, `rule`. `rule` passes a step's win or loss through and otherwise calls `TIE` once `moves` reaches `max_steps` (`>=` rather than `==`, so a loop that overran the clock could never un-tie; the loop stops on the first ruling, so the two never differ in practice). `episode.py` adds `play_episode(environment, referee, agent, seed, on_frame=None)` exactly as §10.6, plus the watch-mode callback of §14.6: a frozen `Frame(observation, t, outcome, ghost_sees, max_steps)` after reset (`t = 0`) and after every step, the last carrying the ruling. `ghost_sees` is line of sight on the true map, computed in the loop from `environment.game_map`, so the renderer (#16) needs only the frame and never the environment.

**A9-01 (minor, noted):** §14.6 says the status line shows `t` of `max_steps`; the frame carries `max_steps` from `referee.max_steps` so the renderer has it without reading configuration.

**Tests:** `test_referee.py` (7): counting from 1 after `start`, restart between games, `TIE` exactly on move `max_steps` and not before, a win or loss on any move passing through, observations carrying no move count. `test_episode.py` (14): call order with a fake environment and a spy agent; agent seeded from `derive(seed, "agent")`; the ruling delivered through `observe` and no `act` after it; the referee's tie; a win on the last move staying a win; latest observation each turn; consecutive games restarting the referee and reseeding the agent; return as the sum of rewards; frames at reset and after each step with the ruling last, `ghost_sees` from the true map, immutable; two real-environment games (a still Pac-Man and a still ghost tie on move `max_steps`; a moving ghost catches a still Pac-Man for a loss); `episode` importing neither the renderer nor concrete agents.

---

## Issue #8 — Environment, partial wall memory and reward accounting

**Date:** 2026-10-29
**Documents:** architecture v0.3 §7.1 to §7.3. Branch `feature/8-environment`, stacked on `feature/7-ghost`.

**Implemented:** `Environment.reset(seed, game_map=None)` generates the seed's map through `split_of_seed` unless a hand-built map is passed (then the seed only seeds the ghost), places Pac-Man at `(0, 0)` and the ghost at `(4, 4)` with `derive(seed, "ghost")` and `ghost_move_prob`, rebuilds memory and senses. `step(action, t)` runs the §7.2 table in order and returns `WIN`, `LOSS` or `None`; `t` goes only to `Ghost.act`. Stepping before `reset` or after a win or loss raises `RuntimeError`. The true map is held as `env.game_map`; nothing in an observation comes from it except through sensing.

**Tests:** `test_env.py` (25): reset state and start sensing (the start cell stays `UNKNOWN`); generation versus hand-built maps; sensing after a move, memory kept across steps and rebuilt at reset; snapshot observations; 40 random games checked against the true map for unsensed-wall leaks; every row of §7.2, including Pac-Man stepping onto the ghost's cell (no corner reward, ghost never moves), the last corner winning before the ghost moves, a loss on the ghost's move, a corner and a loss paying on the same step, and no sensing after a win; `ghost_move_prob` reaching the ghost; `t` stamping visits and otherwise ignored; never `TIE`; errors before reset and after the end; reward totals for a full win; rewards read from the learner settings; `env` importing only game modules.

Tests that need a mid-game state reset on a hand-built map and then set `env.pacman`, `env.visited` and `env.ghost.cell` directly; the environment carries no test-only entry points.

---

## Issue #7 — Ghost vision, chase, pursuit and search

**Date:** 2026-10-29
**Documents:** architecture v0.3 §8. Branch `feature/7-ghost`, stacked on `feature/6-map-generation`.

**Implemented:** `ghost.py` with `Ghost(rng, move_prob=1.0, start=GHOST_START)` holding the §8.1 fields (`cell`, `last_seen`, `last_visit`, `rng`) and `act(pacman, game_map, t)` running §8.2 in order: vision on the true map, the move draw every step, chase or pursue through `first_step`, else the least-recently-visited search with `pick` over tied candidates in direction order, then clearing `last_seen` on arrival and stamping `last_visit[cell] = t`. The ghost takes its generator and `ghost_move_prob` at construction, since `act` carries neither; the environment (#8) builds it with `derive(seed, "ghost")` and `config.game.ghost_move_prob`.

**Tests:** `test_ghost.py` (23) with hand-built maps and scripted draws: start state; chase, blocked and non-aligned vision; pursue with direction-order ties and routing round walls; clearing on arrival, including arrival on Pac-Man's cell; the chase, pursue, search chain; search's never-visited and least-recent rules, single-draw ties at the `pick` boundaries, never standing still, skipping walls; a failed move draw refreshing vision without a visit stamp; the strict `<` on the move draw; a draw taken every step at 1.0 and at 0.0; same seed, same path.

**Note for #8:** `first_step` is only defined for `source != target`, so the environment must end the episode on a collision before calling `act` again; a test that stepped past a collision hit that assertion.

---

## Issue #6 — Map generation, structural filters and splits

**Date:** 2026-10-29
**Documents:** architecture v0.3 §4.2, §6. Branch `feature/6-map-generation`, stacked on `feature/5-grid-primitives`.

**Implemented:** `mapgen.py` now has `split_of_seed`, `draw_candidate`, `is_connected`, `loop_count`, `is_winnable` (wall cap then loop floor), `layout_bytes`, `split_bucket`, `split_of_map` and `generate`, with the §4.2 seed ranges and buckets as the constants `SPLIT_SEEDS` and `SPLIT_BUCKETS`. `generate` runs the §6.2 checks in order and raises `MapGenerationError` naming the seed, split, generator knobs and draw count. `grid.py` gains `CORNERS` (row-major), which the generator, the environment and the agents all need.

**Measured on Python 3.14.10:** the §6.3 layout bytes, SHA-256 digests, buckets and golden maps for seeds 0, 1,000,000 and 2,000,000 all match, and the §6.4 table is reproduced exactly over 1,000 seeds per split (train 1.81 mean draws, max 10, 829 distinct; validation 15.35, 167, 579; test 12.96, 75, 491). `test_splits.py` generates all 3,000 maps in about 2.2 s; the whole suite runs in about 3 s.

**Tests:** `test_mapgen.py` (41) and `test_splits.py` (1): 21 draws per candidate at any density, draw order, corners never walls, connectivity, loop counts (empty 16, ring round a 3x3 block 1, single centre wall 13), the 7-wall cap case, split ranges and bucket membership, draw-limit errors for `min_loops` 17 and `max_walls` 0, same seed same map, non-generator knobs leave maps unchanged, and no global random state touched.

---

## Issue #5 — Grid primitives; environment lane claimed

**Date:** 2026-10-29
**Documents:** architecture v0.3 §9.

**Lane claim:** Christian takes the environment lane of tracker #19 (#5 grid, #6 maps, #7 ghost, #8 environment, #9 referee and episode loop, #16 rendering), starting with #5 on branch `feature/5-grid-primitives`. The learning lane (#10 to #15) and the shared tickets (#13, #17, #18) are open for Phillip.

**A5-01 (minor, fixed):** §9 opened with "all take a `passable` predicate" but listed `move(cell, direction, map)`. Issue #5 and the shared-predicate rule want `move(cell, direction, passable)`, so the signature is now the predicate; #8 passes the ghost's predicate `cell not in map.walls` where §7.2 writes `move(pacman, action, map)`.

**Implemented:** `neighbours`, `move`, `bfs`, `first_step` and `line_of_sight` in `grid.py`, tested in `test_grid.py` (39 tests: off-grid and wall moves, Manhattan and routed BFS distances, unreachable cells omitted, direction-order ties for `first_step`, sight in every direction clear and blocked, adjacent and non-aligned cases, and a subprocess check that `grid` imports no other game module). `first_step` raises `ValueError` when no path exists rather than relying on the §9 no-path argument.

---

## Issue #4 — Foundation implementation and confirmed decisions

**Date:** 2026-10-28
**Documents:** PRD v0.11, architecture v0.3 and `phase-1-foundation-handoff.md`.

**Confirmed by Phillip:** keep bootstrap as the default tie update; reset per-game transition caches without deleting learned Q values; deliver an explicit final outcome and stop action requests. Earlier answers establish that exact cross-machine identity is unnecessary and failed final test targets are documented without test-driven retuning. D-2 through D-7 retain the architecture's existing implementation defaults; no separate approval by Christian is claimed.

**Implemented:** editable Python package metadata, frozen/validated configuration and JSON round trips, component-local RNG helpers and golden value, shared enums and immutable snapshot types, and the Agent protocol. Gameplay entry points are explicit stubs assigned to their later tickets. `runs/` and local development artifacts are ignored. README setup instructions and the handoff document identify the module boundaries.

**Validation:** foundation tests on Python 3.11.14 with NumPy 2.4.6, Matplotlib 3.11.2 and pytest 9.2.0; editable installation and dependency checks pass. Tests cover input failures, JSON round trips, immutable snapshot isolation, RNG golden value/independence, and subprocess repeatability with different hash seeds. Full-game and training performance are still unverified; the earlier probe timings are marked preliminary.

**Scope:** win conditions, start cells, collision precedence, rewards, default structural map filters and ghost behavior are unchanged. No 2x2 wall-placement rule is introduced. No final test-split games were played.

---

## Team rulings — PRD v0.9 → v0.10, architecture v0.1 → v0.2

**Date:** 2026-10-28
**Answered:** OQ-14 and OQ-15, raised and ruled on by the team in one step.

| Question | Ruling | What changed |
|---|---|---|
| OQ-14 Move count | The move count is a referee outside the game. It counts moves, stops the episode at `max_steps`, and tells the agents it was a tie. | PRD §2.4 step 7 and a referee paragraph; §2.8 `observe` receives the outcome instead of an ended flag; FR-4; NFR-7 tests the referee. Architecture §7.4 adds `Referee`; the environment no longer counts moves or returns `TIE`; `Environment.step` takes the move number; the agent interface passes the ruling (§10.1); the learned agent reads the ruling instead of inferring it (§10.5); the episode loop joins the three (§10.6); `test_referee.py`. |
| OQ-15 Grid size | The grid is always 5x5, even in tests. | PRD non-goal. Architecture removes the grid configuration, writes the 5x5 numbers in directly (§4, §8, §10.4, §12), drops the CLI's size check, and has tests use hand-built 5x5 maps through `Environment.reset(seed, game_map=...)`. D-2 keeps the `5x5:` hash tag so the measured splits stand. |

**Consequences checked.**
- *No game plays differently.* The referee calls a tie at the same point the old step 7 did, after the ghost's move, and never overrides a win or loss on the last move. Checked by replaying 1,000 validation seeds per agent through a referee loop with the v0.1 trained table: identical records for all three agents, including the learned agent's 42 ties. No new training run was made; the v0.1 measurements stand.
- *D-1 is still open.* The referee tells the agent that a tie happened. Whether the Q update then bootstraps or stops is still the team's call; the default stays `bootstrap`.

---

## Architecture v0.1, and PRD v0.8 → v0.9

**Date:** 2026-10-28
**Written against:** PRD v0.8 at commit `dbc3077`.
**Method.** Before the document was written, a throwaway probe implemented its rules: the §5 seed derivation and `random()`-only draws, the §6 generator and split encoding, the §7 turn order, the §8 ghost, the three agents, the §10.4 state index, Q-learning with both tie settings, Wilson intervals and the exact McNemar test. It ran on the team laptop's Python 3.9.6, in a scratch environment with NumPy 2.0.2. The probe is not kept. Every number in the architecture doc comes from it.

### Measurements

- *Generator:* draws per accepted map, worst cases and distinct layouts per split (architecture §6.4); the golden hashes and seed-to-map rows (§6.3); an impossible configuration fails in 0.84 s.
- *Training, both tie settings, 50,000 episodes:* 330 s each, with four probe processes sharing the machine. Validation wins 69.1% (bootstrap) and 68.0% (terminal), against 14.2% for pathfinding and 0.0% for random; McNemar `p` below 1e-130 against both. Architecture §11.3.
- *Knob sweep:* both baselines on 1,000 validation seeds per setting. Architecture §15.
- *Replay:* a trained table played on 50 validation seeds twice gave identical records.
- The test split was not played.

### Findings while writing

**A1-01: The machine that ran the probe has no Python 3.11.** Its only Python is the macOS system 3.9.6, without NumPy, while ADR-1 requires 3.11+. *Fix:* architecture §2 says each team member installs 3.11+. The probe uses nothing newer than 3.9, so its numbers stand. *Resolved 2026-10-28:* Python 3.14.10 installed from python.org; the ADR-1 dependencies install on it, and the `derive` and seed-to-map golden values match those measured on 3.9.6.

**A1-02: The PRD's `ghost_move_prob` figure at 1.0 changed with the final split encoding.** Round 3's probe used a provisional encoding and put the pathfinding agent at 12%; under the architecture §6.3 encoding it is 14.2%. The other three figures agree. *Fix:* PRD v0.9 changes the figure in §2.7 to 14% and cites architecture §15. Nothing else in the PRD changes.

**A1-03: Fixed evaluation sizes would break NFR-3.** A `test_cli` run of `train`, `check` and `evaluate` at the PRD's sizes (1,000 seeds each, 200-seed curve checks) would take well over 30 s. *Fix:* the sizes are constants (architecture §11.1) that tests lower with `monkeypatch`.

**A1-04: No test reached a learning-curve point.** CLI tests train for a few episodes, fewer than 2,500. *Fix:* `test_train.py` lowers `CURVE_EVERY` and checks the points.

**A1-05: Training printed nothing.** PRD §2.9 says training runs in simulation mode, which ends with a summary. *Fix:* architecture §14.1 prints each curve point and the final validation summary.

**A1-06: `.gitignore` was missing from the file inventory** that D-6 relies on. *Fix:* added.

### For the team

Architecture §16 lists seven stated defaults (D-1 to D-7) to accept or override before the epics doc. The first validation reading at the default knobs puts the learned agent inside the 50% to 90% band (architecture §11.3); this is not a result, since the test maps have not been played.

---

## Round 3 — PRD v0.7 → v0.8

**Date:** 2026-10-28
**Reviewed:** `phase-1-grid-world-prd.md` v0.7 at commit `caf9415`, against `Design/project-proposal.md` (P1–P4) and rounds 1 and 2 of this log.
**Method:** a full read for architecture readiness, plus a throwaway probe that implements the PRD's generator, turn order, ghost and both baselines (Python standard library, `random.Random`, a provisional split encoding). The probe is not kept; the architecture doc re-runs whatever it relies on.
**Outcome:** 9 findings, all fixed in v0.8. None needed a team ruling: each corrects a claim or closes a gap without changing a decision. Four items added to the §7 handoff list.

### Probe results

- *Generator, 20,000 draws at the defaults:* 68.1% accepted before the split check (1.47 draws per map), mean 3.40 walls, empty map 1.25% of accepted maps. The empty grid has 16 loops and the ring around a 3x3 block has 1. The densest connected map with at least 4 loops had 8 walls. These match §2.3 and the v0.7 ruling.
- *Baselines, 1,000 validation seeds, other knobs at defaults:*

| `ghost_move_prob` | Random W / L / T | Pathfinding W / L / T |
|---|---|---|
| 0 | 0% / 45% / 55% | 0% / 100% / 0% |
| 0.05 | 1% / 78% / 21% | 2% / 98% / 0% |
| 0.1 | 1% / 90% / 9% | 4% / 96% / 0% |
| 0.25 | 0% / 99% / 0% | 10% / 90% / 0% |
| 0.5 | 0% / 100% / 0% | 18% / 82% / 0% |
| 0.75 | 0% / 100% / 0% | 20% / 80% / 0% |
| 1.0 | 0% / 100% / 0% | 12% / 88% / 0% |

### Serious

**R3-01: `ghost_move_prob` is not monotonic.** §2.7 said "raising it only ever makes the game harder". At 0 the ghost never leaves `(4, 4)`, a corner Pac-Man must reach, so no game can be won, and the pathfinding agent's win rate rises from 0% at 0 to 20% at 0.75, then falls to 12% at 1.0. Tuning that assumed one direction would have misread results below the peak.
*Fix:* §2.7 states the effect, the floor and the measured rates; §7 hands a measured knob sweep to the architecture doc. The OQ-10 decision, that `ghost_move_prob` is the primary knob, is unchanged.
*Watch:* the pathfinding agent peaks near 20%. Whether any knob setting lets the learned agent reach the 50% band is not known until it exists. The knob sweep should include the generator knobs and `max_steps`, so other levers toward an easier game are known early.

### Moderate

**R3-02: The 0% alarm fires on a true result.** §5 called any win rate of exactly 0% a sign of a broken measurement, but the random agent wins about 0% of games (0 of 1,000 at the defaults). *Fix:* the 0% alarm is scoped to the pathfinding and learned agents, and §5 says a near-0% random agent is expected.

**R3-03: The generator could loop forever.** v0.7 made `max_walls` and `min_loops` knobs, and some settings accept no map in a split (`min_loops` above 16, or `max_walls` 0, which accepts only the empty map, in one split). The v0.6 handoff item on bounded generation failure was dropped with the v0.6 rule. *Fix:* §2.3 adds a draw limit and an error, FR-2 and NFR-7 cover it, and §7 hands the limit's value to the architecture doc.

### Minor

**R3-04: §2.9 still said a game replays "from its seed and agent".** R2-09 fixed FR-14 but not §2.9. *Fix:* §2.9 points to FR-14.

**R3-05: The policy file did not hold what FR-15 shows.** FR-15 displays the stored policy's last validation results, but the §7 file-format item named only table, configuration and seed range. *Fix:* §7 adds the last validation results.

**R3-06: The tie update was unstated.** The state has no step count, so a tie is a time limit the agent cannot see. Whether the Q update treats it as terminal or bootstraps changes training results (NFR-1). *Fix:* handed to the architecture doc in §7 as a default for team veto.

**R3-07: "Fails to beat both baselines" could read as "beats neither".** §5 item 3 needs both tests to pass, so the fallback opens when either fails. *Fix:* the non-goal and ADR-2 say "fails to beat at least one baseline".

**R3-08: NFR-1 depends on the generator's stream staying fixed.** Some random-number generators do not promise the same stream across library versions. *Fix:* the §7 generator item now includes the algorithm and a version pin.

**R3-09: FR-17's summary was shorter than §2.9's.** *Fix:* FR-17 lists the intervals and mean steps to a win. Also: the status line now cites `caf9415`, and a stray blank line in §6 is removed.

### Checked and sound

- Table size, reward bounds (best tie -3, worst tie -5, every win at least 98, every loss at most -98.05), and the turn order are as stated.
- The ghost always has an open neighbour on an accepted map, and a ghost next to Pac-Man always sees him, so search mode never steps onto Pac-Man unseen.
- Probe speed: about 2.7 ms per game in pure Python including validation-map generation, consistent with NFR-3 and NFR-4.

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
| v0.5 | 2026-10-27 | OQ-10 to OQ-12 answered; `ghost_move_prob`, no retrain in the quick check, evaluation sizes (`d25782e`). |
| v0.6 | 2026-10-27 | OQ-13 answered; random maps must be winnable under the configured ghost rules and turn limit; verification, ghost randomness and bounded failure handed to architecture (`835ba30`). |
| v0.7 | 2026-10-28 | OQ-13 revised: winnability is a map-generator rule (`max_walls`, `min_loops`), independent of Pac-Man and the ghost (`caf9415`). |
| v0.8 | 2026-10-28 | Review round 3 fixes (R3-01 to R3-09): `ghost_move_prob` is not monotonic, bounded map generation, 0% alarm scoped, four items added to the architecture handoff (`dbc3077`). |
| v0.9 | 2026-10-28 | §2.7 pathfinding figure at `ghost_move_prob` 1.0 corrected to 14% under the final split encoding (A1-02); §7 points to the architecture doc. Architecture v0.1 written (`63f3971`). |
| v0.10 | 2026-10-28 | OQ-14 (referee owns the move count and calls the tie) and OQ-15 (grid fixed at 5x5) answered. Architecture v0.2. |
| v0.11 | 2026-10-28 | Issue #4 foundation; confirmed bootstrap default, per-game cache reset and explicit end signals; relaxed cross-machine identity and final-test stopping rule recorded. Architecture v0.3 and foundation handoff. |
