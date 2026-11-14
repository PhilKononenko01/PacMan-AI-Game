import json
from pathlib import Path
import subprocess
import sys

import pytest

from pacman.hybrid.experiment import make_schedule


ROOT = Path(__file__).resolve().parents[1]


def launch(*args):
    return subprocess.run(
        [sys.executable, "-m", "pacman.play", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )


def watch(*args):
    result = launch("--plain", "--delay", "0", *args)
    assert result.returncode == 0, result.stderr
    return result.stdout, json.loads(result.stdout.splitlines()[-1])


def test_default_launches_choose_fresh_episode_ids_with_saved_policies():
    first, second = watch()[1], watch()[1]
    assert (first["layout_seed"], first["episode_key"]) != (second["layout_seed"], second["episode_key"])
    for record in (first, second):
        assert 1_000_000 <= record["layout_seed"] <= 1_000_999
        assert 0 <= record["episode_key"] < 2**63
        assert not record["pac_policy_hash"].startswith("baseline-")
        assert not record["ghost_policy_hash"].startswith("baseline-")


def test_explicit_ids_replay_every_frame_with_both_argument_forms():
    first = watch("--seed", "1000000", "--key", "42")
    second = watch("--seed=1000000", "--key=42")
    assert first == second
    assert first[1]["layout_seed"] == 1_000_000
    assert first[1]["episode_key"] == 42


@pytest.mark.parametrize("baseline,identity", [("random", "random"), ("bfs", "pathfinding")])
def test_baseline_replaces_default_pacman_policy(baseline, identity):
    record = watch("--pac-baseline", baseline, "--seed", "1000000", "--key", "42")[1]
    assert record["pac_policy_hash"] == f"baseline-{identity}"
    assert not record["ghost_policy_hash"].startswith("baseline-")


def test_fixed_ghost_and_custom_policy_paths():
    policies = ROOT / "reports/hybrid-55/replicate-0"
    fixed = watch("--fixed-ghost", "--pac-policy", policies / "pacman.npz")[1]
    assert fixed["ghost_policy_hash"] == "baseline-sight-memory-bfs"
    custom = watch(
        "--pac-policy", policies / "pacman.npz", "--ghost-policy", policies / "ghost.npz",
        "--seed", "1000000", "--key", "42",
    )[1]
    assert custom == watch("--seed", "1000000", "--key", "42")[1]


def test_evaluate_batch_preserves_the_core_matched_schedule(tmp_path):
    out = tmp_path / "batch.json"
    result = launch("--evaluate", "--split", "test", "--games", "2", "--repeats", "2", "--replicate", "1", "--out", out)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert json.loads(result.stdout)["games"] == 4
    assert report["summary"]["games"] == 4
    assert [(r["layout_seed"], r["episode_key"]) for r in report["records"]] == make_schedule("test", 2, 2, 1)


def test_evaluate_supports_baseline_against_fixed_ghost():
    result = launch("--evaluate", "--pac-baseline", "bfs", "--fixed-ghost", "--games", "1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["games"] == 1


@pytest.mark.parametrize("args,message", [
    (("--seed", "bad"), "invalid int"),
    (("--seed", "5000000"), "split"),
    (("--key", "-1"), "key"),
    (("--key", str(2**63)), "key"),
    (("--delay", "nan"), "delay"),
    (("--pac-baseline", "random", "--pac-policy", "x.npz"), "not allowed"),
    (("--fixed-ghost", "--ghost-policy", "x.npz"), "not allowed"),
    (("--pac-policy", "missing.npz"), "missing.npz"),
    (("--evaluate", "--seed=1000000"), "matched schedule"),
    (("--evaluate", "--key=42"), "matched schedule"),
    (("--evaluate", "--games", "0"), "games"),
    (("--games", "1"), "unrecognized"),
    (("--unknown",), "unrecognized"),
])
def test_bad_inputs_are_clean_boundary_errors(args, message):
    result = launch(*args)
    assert result.returncode != 0
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_help_explains_freshness_replay_variations_and_native_options():
    result = launch("--help")
    assert result.returncode == 0
    for text in ("fresh", "--seed", "--key", "--pac-baseline", "--fixed-ghost", "--evaluate", "--games", "--out", "--delay", "--plain"):
        assert text in result.stdout


def test_shell_entry_point_uses_the_launcher():
    result = subprocess.run([str(ROOT / "play"), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Watch a fresh game" in result.stdout
