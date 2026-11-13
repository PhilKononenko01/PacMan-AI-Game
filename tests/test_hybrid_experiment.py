import json
from pathlib import Path
import subprocess
import sys

import pytest

from pacman.hybrid.artifacts import load_policy
from pacman.hybrid.experiment import evaluate, make_schedule, run_episode, summarize, train_run
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import BaselinePac, FrozenTable, PacActor, QTable
from pacman.hybrid.models import Rules


def test_episode_replays_and_callback_includes_initial_and_terminal():
    frames = []
    first = run_episode(0, 22, Rules(), BaselinePac(), GhostController(Rules()),
                        lambda game, result: frames.append((game.turn, game.outcome)))
    second = run_episode(0, 22, Rules(), BaselinePac(), GhostController(Rules()))
    assert first == second
    assert len(first["initial_pellets"]) == 2
    assert len(frames) == first["turns"] + 1
    assert frames[0] == (0, None)
    assert frames[-1][1] is not None


def test_schedule_is_unfiltered_matched_and_has_repeats():
    schedule = make_schedule("val", 3, 2, 1)
    assert [seed for seed, _ in schedule] == [1_000_000, 1_000_000, 1_000_001,
                                             1_000_001, 1_000_002, 1_000_002]
    assert len({key for _, key in schedule}) == 6
    assert schedule == make_schedule("val", 3, 2, 1)
    report = evaluate(BaselinePac(), GhostController(Rules()), games=3, repeats=2, replicate=1)
    assert report["summary"]["games"] == 6
    assert [(r["layout_seed"], r["episode_key"]) for r in report["records"]] == schedule
    assert sum(report["summary"][key] for key in ("win", "loss", "tie")) == 6
    assert report["policy_hashes"] == report["frozen_hashes_after"]


def test_summary_counts_ties_and_reports_layout_clustering():
    records = [{"outcome": outcome, "layout_hash": "same", "turns": 5}
               for outcome in ("win", "loss", "tie")]
    summary = summarize(records)
    assert summary["win_rate"] == 1 / 3
    assert summary["unique_layouts"] == 1
    assert summary["per_layout"]["same"] == {"games": 3, "wins": 1}
    assert "Unclustered" in summary["wilson_caveat"]


def test_evaluation_rejects_mutable_policy():
    with pytest.raises(ValueError, match="frozen"):
        evaluate(PacActor(QTable()), GhostController(Rules()), games=1)


def test_tiny_training_preserves_opponents_and_writes_loadable_artifacts(tmp_path):
    out = tmp_path / "run"
    report = train_run(out, 1, base_episodes=3, ghost_episodes=2, adapt_episodes=2,
                       validation_games=1, finaleval_games=1)
    pac = load_policy(out / "pacman.npz", expected_role="pacman")
    ghost = load_policy(out / "ghost.npz", expected_role="ghost", rules=pac.rules)
    assert isinstance(pac.table, FrozenTable)
    assert isinstance(ghost.table, FrozenTable)
    assert len(list(out.glob("pacman-baseline-*.npz"))) == 3
    for stage in report["stages"].values():
        assert stage["frozen_before"] == stage["frozen_after"]
    seeds = []
    for stage, count in (("baseline", 3), ("ghost", 2), ("adapt", 2)):
        records = [json.loads(line) for line in (out / f"{stage}-games.jsonl").read_text().splitlines()]
        assert len(records) == count
        seeds.extend(record["layout_seed"] for record in records)
    assert seeds == list(range(200_000, 200_007))
    assert len(report["crossplay"]) == 8
    with pytest.raises(FileExistsError):
        train_run(out, base_episodes=1, ghost_episodes=1, adapt_episodes=1, validation_games=1,
                  finaleval_games=None)
    assert not list(out.glob("*test*"))


@pytest.mark.parametrize("kwargs", [{"base_episodes": 0}, {"replicate": 5},
                                    {"validation_games": 1001}, {"repeats": 0},
                                    {"base_episodes": 100000, "ghost_episodes": 100000}])
def test_invalid_budgets_fail_before_creating_output(tmp_path, kwargs):
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        train_run(out, **kwargs)
    assert not out.exists()


def test_cli_drives_train_evaluate_watch(tmp_path):
    out = tmp_path / "cli"
    commands = [["train", "--out", str(out), "--base-episodes", "1", "--ghost-episodes", "1",
                 "--adapt-episodes", "1", "--validation-games", "1", "--games", "1"],
                ["evaluate", "--pac-policy", str(out / "pacman.npz"), "--ghost-policy",
                 str(out / "ghost.npz"), "--games", "1", "--out", str(tmp_path / "eval.json")],
                ["watch", "--pac-policy", str(out / "pacman.npz"), "--ghost-policy",
                 str(out / "ghost.npz"), "--seed", "1000000", "--key", "1", "--delay", "0", "--plain"]]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", "pacman.hybrid", *command],
                                text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
    assert "pellets=" in result.stdout
    assert json.loads((tmp_path / "eval.json").read_text())["summary"]["games"] == 1
