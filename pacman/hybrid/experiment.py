"""Reproducible, unfiltered staged learning and matched cross-play."""

from dataclasses import asdict
from hashlib import sha256
import json
import time
from pathlib import Path

from pacman.env import Outcome
from pacman.hybrid.artifacts import save_policy
from pacman.hybrid.game import Game
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import BaselinePac, FrozenTable, PacActor, QTable
from pacman.hybrid.models import GHOST_FEATURES, PAC_FEATURES, RULES_VERSION, Rules
from pacman.mapgen import SPLIT_SEEDS, Split, layout_bytes, split_of_seed
from pacman.rng import derive, pick
from pacman.stats import wilson

MAX_STAGE_EPISODES = 100_000
REPLICATE_BLOCK = 200_000
MAX_REPLICATES = 5


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def _key(replicate, label, index):
    return derive(replicate, f"hybrid-{label}-{index}").getrandbits(63)


def _identity(actor):
    table = getattr(actor, "table", None)
    if isinstance(table, FrozenTable):
        if not hasattr(actor, "_policy_digest"):
            actor._policy_digest = table.digest()
        return actor._policy_digest
    if isinstance(table, QTable):
        return f"training-update-{table.updates}"
    return f"baseline-{getattr(actor, 'name', 'sight-memory-bfs')}"


def _frozen_hash(actor):
    table = getattr(actor, "table", None)
    if isinstance(table, QTable) or getattr(actor, "training", False):
        raise ValueError("evaluation and opponents require frozen policies")
    return table.digest() if isinstance(table, FrozenTable) else _identity(actor)


def run_episode(layout_seed, episode_key, rules, pac, ghost, on_frame=None):
    """One authoritative loop; callback receives (game, result), initially None."""
    split_of_seed(layout_seed)
    _integer("episode_key", episode_key, 0, 2**63 - 1)
    game = Game(rules, ghost)
    observation = game.reset(layout_seed, episode_key)
    pac.reset(derive(episode_key, "hybrid-pacman"))
    pac_hash, ghost_hash = _identity(pac), _identity(ghost)
    pac_return = ghost_return = 0.0
    if on_frame is not None:
        on_frame(game, None)
    while game.outcome is None:
        result = game.step(pac.act(observation))
        pac.observe(result)
        observation = result.observation
        pac_return += result.pac_reward
        ghost_return += result.ghost_reward
        if on_frame is not None:
            on_frame(game, result)
    return {
        "layout_seed": layout_seed, "episode_key": episode_key,
        "layout_hash": sha256(layout_bytes(game.game_map)).hexdigest(),
        "initial_pellets": [list(cell) for cell in game.initial_pellets],
        "outcome": game.outcome.value, "turns": game.turn,
        "pac_return": pac_return, "ghost_return": ghost_return,
        "pac_policy_hash": pac_hash, "ghost_policy_hash": ghost_hash,
    }


def make_schedule(split="val", games=1000, repeats=1, replicate=0):
    """First N split seed slots, each repeated R times, without layout filtering."""
    if split not in ("val", "test"):
        raise ValueError("evaluation split must be val or test")
    _integer("games", games, 1, 1000)
    _integer("repeats", repeats, 1, 10)
    _integer("replicate", replicate, 0, MAX_REPLICATES - 1)
    start = SPLIT_SEEDS[Split(split)].start
    return [(start + index, _key(replicate, f"{split}-repeat-{repeat}", index))
            for index in range(games) for repeat in range(repeats)]


def summarize(records):
    if not records:
        raise ValueError("cannot summarize an empty schedule")
    counts = {outcome.value: sum(r["outcome"] == outcome.value for r in records)
              for outcome in Outcome}
    wins = counts[Outcome.WIN.value]
    grouped = {}
    for record in records:
        group = grouped.setdefault(record["layout_hash"], {"games": 0, "wins": 0})
        group["games"] += 1
        group["wins"] += record["outcome"] == Outcome.WIN.value
    return {
        "games": len(records), **counts, "win_rate": wins / len(records),
        "wilson_95": list(wilson(wins, len(records))),
        "wilson_caveat": "Unclustered game-level interval; repeated seeds and identical layouts are correlated.",
        "target_55_observed": wins / len(records) >= 0.55,
        "mean_turns": sum(r["turns"] for r in records) / len(records),
        "unique_layouts": len(grouped), "per_layout": grouped,
    }


def evaluate(pac, ghost, rules=None, *, split="val", games=1000, repeats=1,
             replicate=0, schedule=None):
    rules = rules or Rules()
    expected = make_schedule(split, games, repeats, replicate)
    schedule = expected if schedule is None else list(schedule)
    if not schedule:
        raise ValueError("evaluation schedule must not be empty")
    for seed, key in schedule:
        if split_of_seed(seed).value != split:
            raise ValueError("schedule contains a seed from the wrong split")
        _integer("episode_key", key, 0, 2**63 - 1)
    before = {"pacman": _frozen_hash(pac), "ghost": _frozen_hash(ghost)}
    coverage_before = {"pacman": _coverage(pac), "ghost": _coverage(ghost)}
    records = [run_episode(seed, key, rules, pac, ghost) for seed, key in schedule]
    after = {"pacman": _frozen_hash(pac), "ghost": _frozen_hash(ghost)}
    if before != after:
        raise RuntimeError("evaluation mutated a frozen opponent")
    coverage = {"pacman": _coverage(pac), "ghost": _coverage(ghost)}
    for role, values in coverage.items():
        for counter in ("decisions", "unseen"):
            values[counter] -= coverage_before[role][counter]
        values["unseen_fraction"] = values["unseen"] / values["decisions"] if values["decisions"] else None
    return {
        "rules_version": RULES_VERSION, "rules": asdict(rules),
        "features": {"pacman": PAC_FEATURES, "ghost": GHOST_FEATURES},
        "split": split, "replicate": replicate, "policy_hashes": before,
        "frozen_hashes_after": after, "coverage": coverage,
        "records": records, "summary": summarize(records),
    }


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _coverage(actor):
    decisions = getattr(actor, "decisions", getattr(actor, "search_decisions", 0))
    unseen = getattr(actor, "unseen", 0)
    table = getattr(actor, "table", None)
    return {"decisions": decisions, "unseen": unseen,
            "unseen_fraction": unseen / decisions if decisions else None,
            "states": len(table.rows) if table is not None else 0,
            "q_updates": getattr(table, "updates", 0)}


def train_run(out, replicate=0, *, base_episodes=30_000, ghost_episodes=30_000,
              adapt_episodes=50_000, validation_games=200, finaleval_games=1000,
              repeats=1, rules=None):
    """Fixed budgets, three frozen baseline snapshots, no test-set selection."""
    _integer("replicate", replicate, 0, MAX_REPLICATES - 1)
    for name, count in (("base_episodes", base_episodes), ("ghost_episodes", ghost_episodes),
                        ("adapt_episodes", adapt_episodes)):
        _integer(name, count, 1, MAX_STAGE_EPISODES)
    if base_episodes + ghost_episodes + adapt_episodes > REPLICATE_BLOCK:
        raise ValueError(f"total episodes must not exceed {REPLICATE_BLOCK}")
    _integer("validation_games", validation_games, 1, 1000)
    if finaleval_games is not None:
        _integer("finaleval_games", finaleval_games, 1, 1000)
    _integer("repeats", repeats, 1, 10)
    rules = rules or Rules()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "replicate": replicate, "rules_version": RULES_VERSION, "rules": asdict(rules),
        "features": {"pacman": PAC_FEATURES, "ghost": GHOST_FEATURES},
        "budgets": {"baseline": base_episodes, "ghost": ghost_episodes, "adapt": adapt_episodes},
        "layout_seed_block": [replicate * REPLICATE_BLOCK, (replicate + 1) * REPLICATE_BLOCK - 1],
        "episode_key_scheme": "derive(replicate, hybrid-{stage}-{index}).getrandbits(63)",
        "epsilon": "linear 1.0 to 0.05 during first 70% of each stage, then 0.05",
        "training_policy_identity": "Mutable policies use training-update-N revision identifiers; frozen policies use SHA256 table hashes.",
        "evaluation": "All scheduled games; ties and losses remain in denominator; validation only.",
        "stages": {},
        "source_files_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "learning_parameters": {
            "pacman": {"alpha": 0.1, "gamma": 1.0, "win": 100.0, "loss": -100.0,
                       "timeout": -100.0, "pellet": 1.0, "step": -0.05},
            "ghost": {"alpha": 0.1, "gamma": 0.97, "capture": 1.0,
                      "pacman_win": -1.0, "timeout": 0.0},
        },
    }
    write_json(out / "protocol.json", metadata)
    offset = 0

    def stage(name, episodes, pac, ghost_for, frozen_opponents=(), checkpoint=None):
        nonlocal offset
        before = [_frozen_hash(actor) for actor in frozen_opponents]
        counters = {outcome.value: 0 for outcome in Outcome}
        started = time.perf_counter()
        turns = 0
        with (out / f"{name}-games.jsonl").open("x", encoding="utf-8") as log:
            for index in range(episodes):
                epsilon = max(0.05, 1.0 - 0.95 * index / max(1, episodes * 0.7))
                pac_actor, ghost_actor = pac(index), ghost_for(index)
                if getattr(pac_actor, "training", False):
                    pac_actor.epsilon = epsilon
                if getattr(ghost_actor, "training", False):
                    ghost_actor.epsilon = epsilon
                record = run_episode(replicate * REPLICATE_BLOCK + offset + index,
                                     _key(replicate, name, index), rules, pac_actor, ghost_actor)
                counters[record["outcome"]] += 1
                turns += record["turns"]
                log.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                if checkpoint is not None:
                    checkpoint(index + 1)
                if (index + 1) % 5000 == 0 or index + 1 == episodes:
                    print(f"replicate={replicate} stage={name} episodes={index + 1}/{episodes} outcomes={counters}", flush=True)
        elapsed = time.perf_counter() - started
        after = [_frozen_hash(actor) for actor in frozen_opponents]
        if before != after:
            raise RuntimeError(f"{name} mutated a frozen opponent")
        metadata["stages"][name] = {"episodes": episodes, "outcomes": counters,
                                     "turns": turns, "elapsed_seconds": elapsed,
                                     "episodes_per_second": episodes / elapsed,
                                     "frozen_before": before, "frozen_after": after,
                                     "first_layout_seed": replicate * REPLICATE_BLOCK + offset}
        offset += episodes

    baseline_table = QTable()
    baseline = PacActor(baseline_table, training=True)
    fixed = GhostController(rules)
    pool = []
    checkpoint_counts = [max(1, (base_episodes + 2) // 3),
                         max(1, (2 * base_episodes + 2) // 3), base_episodes]

    def baseline_checkpoint(completed):
        for slot, count in enumerate(checkpoint_counts):
            if count == completed:
                frozen = PacActor(baseline_table.freeze())
                pool.append(frozen)
                save_policy(out / f"pacman-baseline-{slot + 1}.npz", "pacman", frozen.table, rules,
                            {"stage": "baseline", "episodes": completed, **_coverage(baseline), "replicate": replicate})
                report = evaluate(frozen, GhostController(rules), rules, games=validation_games, replicate=replicate)
                write_json(out / f"baseline-{slot + 1}-validation.json", report)

    stage("baseline", base_episodes, lambda _: baseline, lambda _: fixed, checkpoint=baseline_checkpoint)
    baseline_frozen = pool[-1]
    metadata["stages"]["baseline"]["coverage"] = _coverage(baseline)
    ghost_table = QTable()
    learned_ghost = GhostController(rules, ghost_table, training=True)
    pool_rng = derive(replicate, "hybrid-opponent-pool")
    stage("ghost", ghost_episodes, lambda _: pool[pick(pool_rng, len(pool))],
          lambda _: learned_ghost, frozen_opponents=pool)
    ghost_frozen = GhostController(rules, ghost_table.freeze())
    ghost_digest = _frozen_hash(ghost_frozen)
    metadata["stages"]["ghost"]["coverage"] = _coverage(learned_ghost)
    save_policy(out / "ghost.npz", "ghost", ghost_frozen.table, rules,
                {"stage": "ghost", "replicate": replicate, **_coverage(learned_ghost),
                 "frozen_pacman_pool": [_frozen_hash(actor) for actor in pool]})
    write_json(out / "ghost-validation.json", evaluate(baseline_frozen, ghost_frozen, rules,
                                                       games=validation_games, replicate=replicate))
    adapted = PacActor(QTable(baseline_frozen.table), training=True)
    # A balanced shuffled pair schedule gives exactly 50/50 for even budgets.
    mixture_rng = derive(replicate, "hybrid-ghost-mixture")
    pair = []

    def mixed_ghost(index):
        if index % 2 == 0:
            pair[:] = [fixed, ghost_frozen]
            if pick(mixture_rng, 2):
                pair.reverse()
        return pair[index % 2]

    stage("adapt", adapt_episodes, lambda _: adapted, mixed_ghost,
          frozen_opponents=(baseline_frozen, ghost_frozen))
    metadata["stages"]["adapt"]["coverage"] = _coverage(adapted)
    adapted_frozen = PacActor(adapted.table.freeze())
    save_policy(out / "pacman.npz", "pacman", adapted_frozen.table, rules,
                {"stage": "adapt", "replicate": replicate, **_coverage(adapted),
                 "warm_start": _frozen_hash(baseline_frozen), "frozen_ghost": ghost_digest})
    write_json(out / "adapt-validation.json", evaluate(adapted_frozen, ghost_frozen, rules,
                                                       games=validation_games, replicate=replicate))
    if finaleval_games is not None:
        matrix = {}
        for pac_name, pac_actor in (("baseline", baseline_frozen), ("adapted", adapted_frozen),
                                     ("random", BaselinePac("random")), ("bfs", BaselinePac())):
            for ghost_name, ghost_actor in (("fixed", GhostController(rules)), ("learned", ghost_frozen)):
                name = f"{pac_name}-vs-{ghost_name}"
                report = evaluate(pac_actor, ghost_actor, rules, games=finaleval_games,
                                  repeats=repeats, replicate=replicate)
                write_json(out / f"{name}-validation.json", report)
                matrix[name] = report["summary"]
        metadata["crossplay"] = matrix
    metadata["final_policy_hashes"] = {"baseline": _frozen_hash(baseline_frozen),
                                         "adapted": _frozen_hash(adapted_frozen),
                                         "ghost": _frozen_hash(ghost_frozen)}
    write_json(out / "summary.json", metadata)
    return metadata
