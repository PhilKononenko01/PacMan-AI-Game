import json

import pytest

from pacman.hybrid.experiment import summarize, train_run
from scripts.verify_hybrid_run import verify_run as verify_registered


def verify_run(path, replay_limit=None):
    return verify_registered(path, replay_limit, final_slots=3, repeats=1, checkpoint_slots=2)


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "run"
    train_run(path, base_episodes=3, ghost_episodes=3, adapt_episodes=3,
              validation_games=2, finaleval_games=3)
    return path


def test_verifier_checks_every_record_and_replays_all_games(run):
    result = verify_run(run, None)
    assert result["verified"]
    assert result["records_checked"] == result["games_replayed"]
    assert result["reports"] == 13


def test_verifier_rejects_incorrect_totals(run):
    path = run / "adapt-validation.json"
    report = json.loads(path.read_text())
    report["summary"]["win"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="summary disagree"):
        verify_run(run)


def test_verifier_rejects_wrong_game_even_when_aggregate_counts_match(run):
    path = run / "adapt-validation.json"
    report = json.loads(path.read_text())
    report["records"][0]["turns"] += 1
    report["summary"] = summarize(report["records"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replay differs"):
        verify_run(run, None)


def test_verifier_rejects_missing_evaluation(run):
    (run / "ghost-validation.json").unlink()
    with pytest.raises(ValueError, match="missing evaluation"):
        verify_run(run)


@pytest.mark.parametrize("mutation", ["hashes", "rules", "features", "replicate", "missing_game"])
def test_verifier_rejects_bad_provenance_and_missing_games(run, mutation):
    path = run / "adapt-validation.json"
    report = json.loads(path.read_text())
    if mutation == "hashes":
        report["policy_hashes"] = report["frozen_hashes_after"] = {"pacman": "wrong", "ghost": "wrong"}
    elif mutation == "rules":
        report["rules"] = {"bogus": True}
    elif mutation == "features":
        report["features"] = {"pacman": "wrong", "ghost": "wrong"}
    elif mutation == "replicate":
        report["replicate"] = 1
    else:
        report["records"].pop()
        report["summary"] = summarize(report["records"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_run(run)


def test_sampled_replay_never_claims_complete_verification(run):
    result = verify_run(run, 1)
    assert not result["verified"] and not result["all_games_replayed"]
    assert result["record_consistency_verified"]
