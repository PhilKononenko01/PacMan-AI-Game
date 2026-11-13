"""Freeze all five trained runs before evaluating their declared test schedules."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacman.hybrid.artifacts import load_policy
from pacman.hybrid.experiment import evaluate, write_json
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import PacActor


def evaluate_run(arguments):
    root, replicate = arguments
    run = Path(root) / f"hybrid-55-replicate-{replicate}"
    baseline = load_policy(run / "pacman-baseline-3.npz", "pacman")
    adapted = load_policy(run / "pacman.npz", "pacman", baseline.rules)
    ghost = load_policy(run / "ghost.npz", "ghost", baseline.rules)
    summaries = {}
    for pac_name, policy in (("baseline", baseline), ("adapted", adapted)):
        for ghost_name, table in (("fixed", None), ("learned", ghost.table)):
            label = f"{pac_name}-vs-{ghost_name}"
            report = evaluate(PacActor(policy.table), GhostController(policy.rules, table),
                              policy.rules, split="test", games=1000, repeats=2, replicate=replicate)
            write_json(run / f"{label}-test.json", report)
            summaries[label] = {key: report["summary"][key] for key in ("games", "win", "loss", "tie", "win_rate")}
    return {"replicate": replicate, "test": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    files = {}
    for replicate in range(5):
        run = args.runs / f"hybrid-55-replicate-{replicate}"
        summary = json.loads((run / "summary.json").read_text())
        if summary["replicate"] != replicate or summary["budgets"] != {"baseline": 30000, "ghost": 30000, "adapt": 50000}:
            raise ValueError("training does not match declared replicate protocol")
        for path in sorted(run.glob("*.npz")):
            files[str(path)] = sha256(path.read_bytes()).hexdigest()
        for label in ("baseline", "adapted"):
            for ghost in ("fixed", "learned"):
                if (run / f"{label}-vs-{ghost}-test.json").exists():
                    raise FileExistsError("test evaluation already exists; refusing another acceptance run")
    write_json(args.freeze, {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "policy_files_sha256": files,
        "replicates": list(range(5)), "split": "test", "layout_seed_slots": 1000,
        "episode_keys_per_slot": 2, "all_scheduled_games_count": True,
        "selection": "All five final policies retained; no test-driven selection or tuning.",
        "historical_layout_use": "These wall-layout test partitions were evaluated for the earlier corner game.",
    })
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(evaluate_run, [(str(args.runs), i) for i in range(5)]):
            print(json.dumps(result, sort_keys=True), flush=True)
    for filename, digest in files.items():
        if sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError("a frozen policy file changed during evaluation")


if __name__ == "__main__":
    main()
