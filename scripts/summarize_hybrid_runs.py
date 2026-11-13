"""Archive five frozen experiments and summarize their matched test evaluations."""

from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_hybrid_run import read_json, verify_run


def clustered_interval(reports, baseline_reports=None, samples=5000):
    layouts = sorted({row["layout_hash"] for report in reports for row in report["records"]})
    indices = {layout: index for index, layout in enumerate(layouts)}
    wins = np.zeros((len(reports), len(layouts)))
    games = np.zeros_like(wins)
    for replicate, report in enumerate(reports):
        rows = report["records"]
        baseline = baseline_reports[replicate]["records"] if baseline_reports is not None else None
        if baseline is not None and [(r["layout_seed"], r["episode_key"]) for r in rows] != [
            (r["layout_seed"], r["episode_key"]) for r in baseline
        ]:
            raise ValueError("comparison schedules differ")
        if baseline is not None and [r["layout_hash"] for r in rows] != [r["layout_hash"] for r in baseline]:
            raise ValueError("comparison layouts differ")
        for index, row in enumerate(rows):
            column = indices[row["layout_hash"]]
            wins[replicate, column] += (row["outcome"] == "win") - (
                int(baseline[index]["outcome"] == "win") if baseline is not None else 0)
            games[replicate, column] += 1
    rng = np.random.default_rng(20261113)
    estimates = []
    for _ in range(samples):
        rows = rng.integers(len(reports), size=len(reports))
        cols = rng.integers(len(layouts), size=len(layouts))
        estimates.append(wins[rows][:, cols].sum() / games[rows][:, cols].sum())
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def summarize_runs(source, destination):
    source, destination = Path(source), Path(destination)
    runs = [source / f"hybrid-55-replicate-{i}" for i in range(5)]
    pairs = [f"{pac}-vs-{ghost}" for pac in ("baseline", "adapted") for ghost in ("fixed", "learned")]
    reports = {name: [read_json(run / f"{name}-test.json") for run in runs] for name in pairs}
    for values in reports.values():
        if any(report["summary"]["games"] != 2000 or report["split"] != "test"
               or report["replicate"] != index for index, report in enumerate(values)):
            raise ValueError("final protocol requires 2000 test games per replicate")
    verified_runs = []
    for run in runs:
        verified_runs.append(verify_run(run, final_slots=1000, repeats=2, checkpoint_slots=200))
        print(f"Replayed {run.name}: {verified_runs[-1]['games_replayed']} games", flush=True)
    if not all(verified["verified"] for verified in verified_runs):
        raise ValueError("archive requires a complete replay")
    summaries = {}
    for name, values in reports.items():
        counts = defaultdict(int)
        for report in values:
            for outcome in ("win", "loss", "tie", "games"):
                counts[outcome] += report["summary"][outcome]
        rates = [r["summary"]["win_rate"] for r in values]
        summaries[name] = {**counts, "win_rate": counts["win"] / counts["games"],
                           "replicate_win_rates": rates, "range": [min(rates), max(rates)],
                           "cluster_bootstrap_95": clustered_interval(values)}
    final = summaries["adapted-vs-learned"]
    result = {
        "target": 0.55, "replicates": 5, "test_games_per_replicate": 2000,
        "target_met_mean": final["win_rate"] >= 0.55,
        "target_met_each_replicate": all(r >= 0.55 for r in final["replicate_win_rates"]),
        "test": summaries,
        "adaptation_gain_vs_learned": final["win_rate"] - summaries["baseline-vs-learned"]["win_rate"],
        "adaptation_gain_cluster_bootstrap_95": clustered_interval(
            reports["adapted-vs-learned"], reports["baseline-vs-learned"]),
        "uncertainty_method": "5000 bootstrap samples; resample training replicate and wall-layout hash independently, preserving paired game outcomes. RNG 20261113. Five training replicates limit inference.",
        "limitations": ["Adaptation adds 50000 training games and mixed opponents together; no equal-budget fixed-only control.",
                        "Test wall layouts were used by the historical project under different rules; they are not a novel dataset.",
                        "New game rules differ from the historical corner game; compare policies only within matching rules."],
    }
    destination.mkdir(parents=True, exist_ok=False)
    for index, (run, verified) in enumerate(zip(runs, verified_runs)):
        target = destination / f"replicate-{index}"
        target.mkdir()
        for path in list(run.glob("*.npz")) + [run / "summary.json", run / "protocol.json"]:
            shutil.copy2(path, target / path.name)
        for path in sorted(run.glob("*-validation.json")) + sorted(run.glob("*-test.json")):
            with gzip.GzipFile(filename=str(target / (path.name + ".gz")), mode="wb", mtime=0) as stream:
                stream.write(path.read_bytes())
        (target / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    (destination / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    inventory = {str(path.relative_to(destination)): sha256(path.read_bytes()).hexdigest()
                 for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination / "sha256.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_hybrid_runs.py RUNS_DIRECTORY NEW_OUTPUT_DIRECTORY")
    print(json.dumps(summarize_runs(sys.argv[1], sys.argv[2]), indent=2))
