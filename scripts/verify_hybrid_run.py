"""Check saved hybrid evaluations and replay their recorded games."""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pacman.hybrid.artifacts import load_policy
from pacman.hybrid.experiment import make_schedule, run_episode, summarize
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import BaselinePac, PacActor
from pacman.hybrid.models import GHOST_FEATURES, PAC_FEATURES, RULES_VERSION


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def verify_run(directory, replay_limit=None, *, final_slots=1000, repeats=2, checkpoint_slots=200):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    protocol = read_json(directory / "protocol.json")
    if summary["replicate"] != protocol["replicate"] or summary["budgets"] != protocol["budgets"]:
        raise ValueError("run summary disagrees with its training protocol")
    policies = {name: load_policy(directory / filename, role)
                for name, filename, role in (
                    ("baseline-1", "pacman-baseline-1.npz", "pacman"),
                    ("baseline-2", "pacman-baseline-2.npz", "pacman"),
                    ("baseline-3", "pacman-baseline-3.npz", "pacman"),
                    ("adapted", "pacman.npz", "pacman"),
                    ("ghost", "ghost.npz", "ghost"))}
    rules = policies["adapted"].rules
    if any(policy.rules != rules for policy in policies.values()):
        raise ValueError("artifact rules differ within one run")
    if any(policy.metadata["replicate"] != summary["replicate"] for policy in policies.values()):
        raise ValueError("policy replicate disagrees with its training protocol")
    if summary["rules"] != asdict(rules) or protocol["rules"] != asdict(rules):
        raise ValueError("run rules disagree with saved policies")
    hashes = {name: policies[name].table.digest() for name in policies}
    if summary["final_policy_hashes"] != {
        "baseline": hashes["baseline-3"], "adapted": hashes["adapted"], "ghost": hashes["ghost"]
    }:
        raise ValueError("summary policy hashes do not match saved policies")
    for stage in summary["stages"].values():
        if stage["frozen_before"] != stage["frozen_after"]:
            raise ValueError("a training stage changed frozen opponents")
        if sum(stage["outcomes"].values()) != stage["episodes"]:
            raise ValueError("training outcomes do not count every episode")
    paths = sorted(directory.glob("*-validation.json")) + sorted(directory.glob("*-validation.json.gz"))
    test_paths = sorted(directory.glob("*-test.json")) + sorted(directory.glob("*-test.json.gz"))
    paths += test_paths
    validation_names = {f"baseline-{i}" for i in (1, 2, 3)} | {"ghost", "adapt"} | {
        f"{pac}-vs-{ghost}" for pac in ("baseline", "adapted", "random", "bfs") for ghost in ("fixed", "learned")}
    expected_names = {(name, "val") for name in validation_names}
    if test_paths:
        expected_names.update((f"{pac}-vs-{ghost}", "test") for pac in ("baseline", "adapted")
                              for ghost in ("fixed", "learned"))
    found = set()
    replays = checked = 0
    for path in paths:
        suffix = "-test.json" if "-test.json" in path.name else "-validation.json"
        name = path.name.split(suffix)[0]
        partition = "test" if suffix == "-test.json" else "val"
        label = name, partition
        if label in found or label not in expected_names:
            raise ValueError("unexpected or duplicate evaluation report")
        found.add(label)
        report = read_json(path)
        if report["split"] != partition:
            raise ValueError("report partition does not match its filename")
        if report["replicate"] != summary["replicate"]:
            raise ValueError("report replicate disagrees with registered protocol")
        if (report["rules"] != asdict(rules) or report["rules_version"] != RULES_VERSION
                or report["features"] != {"pacman": PAC_FEATURES, "ghost": GHOST_FEATURES}):
            raise ValueError("report rules or features disagree with saved policies")
        records = report["records"]
        if summarize(records) != report["summary"]:
            raise ValueError(f"evaluation counts or summary disagree with records: {path.name}")
        full_matrix = "-vs-" in name
        slots = final_slots if full_matrix else checkpoint_slots
        scheduled_repeats = repeats if full_matrix else 1
        schedule = make_schedule(partition, slots, scheduled_repeats, summary["replicate"])
        if schedule != [(row["layout_seed"], row["episode_key"]) for row in records]:
            raise ValueError(f"evaluation omitted or altered scheduled games: {path.name}")
        if report["policy_hashes"] != report["frozen_hashes_after"]:
            raise ValueError("evaluation changed a frozen opponent")
        if name.startswith("baseline-") and "-vs-" not in name:
            pac_name, ghost_name = name, "fixed"
        elif name == "ghost":
            pac_name, ghost_name = "baseline-3", "learned"
        elif name == "adapt":
            pac_name, ghost_name = "adapted", "learned"
        else:
            pac_name, ghost_name = name.split("-vs-")
            pac_name = "baseline-3" if pac_name == "baseline" else pac_name
        pac = (BaselinePac("pathfinding" if pac_name == "bfs" else "random")
               if pac_name in ("random", "bfs") else PacActor(policies[pac_name].table))
        ghost = GhostController(rules, policies["ghost"].table if ghost_name == "learned" else None)
        identities = {
            "pacman": pac.table.digest() if isinstance(pac, PacActor) else f"baseline-{pac.name}",
            "ghost": ghost.table.digest() if ghost.table is not None else "baseline-sight-memory-bfs",
        }
        if report["policy_hashes"] != identities or any(
            row["pac_policy_hash"] != identities["pacman"] or row["ghost_policy_hash"] != identities["ghost"]
            for row in records
        ):
            raise ValueError("report policy hashes disagree with saved policies")
        count = len(records) if replay_limit is None else min(replay_limit, len(records))
        indices = sorted({i * len(records) // count for i in range(count)}) if count else []
        for index in indices:
            row = records[index]
            replay = run_episode(row["layout_seed"], row["episode_key"], rules, pac, ghost)
            if replay != row:
                raise ValueError(f"replay differs from recorded game: {path.name} index {index}")
            replays += 1
        if partition == "val" and name in summary.get("crossplay", {}) and report["summary"] != summary["crossplay"][name]:
            raise ValueError("cross-play table disagrees with its evaluation")
        checked += len(records)
    if found != expected_names:
        raise ValueError(f"missing evaluation reports: {sorted(expected_names - found)}")
    return {"verified": checked == replays, "record_consistency_verified": True,
            "all_games_replayed": checked == replays,
            "registered_schedule": {"replicate": summary["replicate"], "final_slots": final_slots,
                                    "repeats": repeats, "checkpoint_slots": checkpoint_slots},
            "reports": len(found), "records_checked": checked,
            "games_replayed": replays, "policy_file_hashes": {key: p.digest for key, p in policies.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--all", action="store_true", help="Replay every evaluation game")
    parser.add_argument("--sample", type=int, help="Sample replays; does not establish complete verification")
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--checkpoint-slots", type=int, default=200)
    args = parser.parse_args()
    if args.sample is not None and (args.all or args.sample < 1):
        parser.error("--sample must be positive and cannot be combined with --all")
    print(json.dumps(verify_run(args.run, args.sample, final_slots=args.slots,
                               repeats=args.repeats, checkpoint_slots=args.checkpoint_slots), indent=2))


if __name__ == "__main__":
    main()
