"""The five commands end to end with small sizes, and the agent factory (FR-9, FR-13 to FR-17)."""

import json
import subprocess
import sys

import numpy
import pytest

from pacman import evaluate as evaluate_module
from pacman import train as train_module
from pacman.agents import AGENT_NAMES, make_agent
from pacman.agents.learned import LearnedAgent
from pacman.agents.pathfinding import PathfindingAgent
from pacman.agents.random_agent import RandomAgent
from pacman.cli import load_config, main, parse_args
from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.evaluate import read_eval_json
from pacman.policy_io import Policy, ValidationSummary, load_policy


@pytest.fixture(autouse=True)
def small_sizes(monkeypatch):
    monkeypatch.setattr(train_module, "CURVE_EVERY", 2)
    monkeypatch.setattr(train_module, "CURVE_GAMES", 2)
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 4)


@pytest.fixture
def run(capsys):
    def run(*args):
        code = main([str(arg) for arg in args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err
    return run


@pytest.fixture
def trained(run, tmp_path):
    out = tmp_path / "runs" / "short"
    code, _, err = run("train", "--out", out, "--set", "episodes=5", "--set", "max_steps=20")
    assert code == 0, err
    return out


def policy_with(config):
    return Policy(
        q=numpy.zeros((122_880, 4)), config=config, training_seeds=(0, 4), learning_curve=(),
        validation=ValidationSummary((1_000_000, 1_000_003), 4, 0, 4, 0, None),
    )


def test_python_m_pacman_lists_the_five_commands():
    result = subprocess.run(
        [sys.executable, "-m", "pacman", "--help"], capture_output=True, text=True,
    )
    assert result.returncode == 0
    for command in ("train", "evaluate", "check", "simulate", "watch"):
        assert command in result.stdout


def test_make_agent_builds_each_named_agent():
    assert AGENT_NAMES == ("random", "pathfinding", "learned")
    config = Config()
    assert isinstance(make_agent("random", config), RandomAgent)
    assert isinstance(make_agent("pathfinding", config), PathfindingAgent)
    policy = policy_with(config)
    learned = make_agent("learned", config, policy)
    assert isinstance(learned, LearnedAgent)
    assert learned.q is policy.q and learned.training is False
    with pytest.raises(ValueError, match="policy"):
        make_agent("learned", config)
    with pytest.raises(ValueError, match="greedy"):
        make_agent("greedy", config)


def test_load_config_applies_sets_by_bare_name_on_top_of_a_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(Config(game=GameKnobs(max_walls=5)).to_json())
    config = load_config(path, ["max_steps=50", "learning_rate=0.3", "tie_update=terminal"])
    assert config.game == GameKnobs(max_walls=5, max_steps=50)
    assert config.learner == LearnerSettings(learning_rate=0.3, tie_update="terminal")
    assert load_config(None, []) == Config()


@pytest.mark.parametrize("setting", ["speed=1", "max_steps", "max_steps=5.0", "max_steps=fast", "episodes=0"])
def test_load_config_rejects_bad_sets(setting):
    with pytest.raises(ValueError):
        load_config(None, [setting])


def test_bad_set_values_fail_with_a_message_and_a_non_zero_exit(run):
    code, _, err = run("simulate", "--agent", "random", "--split", "val", "--games", 1, "--set", "speed=1")
    assert code == 1 and "speed" in err


def test_watch_delay_defaults_to_a_quarter_second():
    assert parse_args(["watch", "--agent", "random", "--seed", "0"]).delay == 0.25


def test_train_prints_curve_points_and_the_final_row_and_writes_the_run(run, tmp_path):
    out = tmp_path / "run"
    code, output, err = run("train", "--out", out, "--set", "episodes=5", "--set", "max_steps=20")
    assert code == 0, err
    lines = output.splitlines()
    assert [line for line in lines if line.startswith("episodes")] == [
        line for line in lines if line.startswith("episodes 2:") or line.startswith("episodes 4:")
    ]
    assert len([line for line in lines if line.startswith("episodes")]) == 2
    assert any(line.startswith("learned ") for line in lines)
    assert sorted(path.name for path in out.iterdir()) == ["config.json", "curve.csv", "curve.png", "policy.npz"]
    policy = load_policy(out / "policy.npz")
    assert policy.config.game.max_steps == 20 and policy.config.learner.episodes == 5
    assert policy.validation.games == 4


def test_train_refuses_an_existing_run_before_training(run, trained):
    code, _, err = run("train", "--out", trained, "--set", "episodes=5")
    assert code == 1 and "already exists" in err


def test_evaluate_reports_all_three_agents_and_writes_the_json(run, trained):
    code, output, err = run("evaluate", "--split", "val", "--policy", trained / "policy.npz")
    assert code == 0, err
    lines = output.splitlines()
    assert lines[0].startswith("split val, 4 games per agent, knobs:")
    assert "max_steps=20" in lines[0]
    assert [line.split()[0] for line in lines[2:5]] == ["random", "pathfinding", "learned"]
    assert lines[5].startswith("learned vs random:") and lines[6].startswith("learned vs pathfinding:")
    config, records = read_eval_json(trained / "eval-val.json")
    assert config.game.max_steps == 20
    assert len(records) == 12
    assert {record.agent for record in records} == {"random", "pathfinding", "learned"}
    assert sorted({record.seed for record in records}) == list(range(1_000_000, 1_000_004))


def test_evaluate_refuses_a_game_knob_that_differs_from_the_policy(run, trained):
    code, _, err = run("evaluate", "--split", "val", "--policy", trained / "policy.npz", "--set", "max_steps=50")
    assert code == 1 and "max_steps" in err
    code, _, err = run("evaluate", "--split", "val", "--policy", trained / "policy.npz", "--set", "learning_rate=0.5")
    assert code == 0, err


def test_check_plays_the_baselines_and_shows_the_stored_policy_with_differing_knobs(run, trained):
    code, output, err = run("check", "--set", "max_steps=20")
    assert code == 0, err
    rows = [line.split()[0] for line in output.splitlines() if line and not line.startswith(("split", "agent"))]
    assert rows == ["random", "pathfinding"]
    before = (trained / "policy.npz").read_bytes()
    code, output, err = run("check", "--policy", trained / "policy.npz", "--set", "max_steps=25")
    assert code == 0, err
    assert any(line.startswith("learned ") for line in output.splitlines())
    assert "max_steps=25*" in output and "policy: 20" in output
    assert (trained / "policy.npz").read_bytes() == before


def test_simulate_prints_a_row_per_agent_for_the_requested_games(run, trained):
    code, output, err = run("simulate", "--agent", "random", "--agent", "pathfinding", "--split", "val", "--games", 3)
    assert code == 0, err
    lines = output.splitlines()
    assert lines[0].startswith("split val, 3 games per agent")
    assert [line.split()[0] for line in lines[2:]] == ["random", "pathfinding"]
    code, output, err = run(
        "simulate", "--agent", "learned", "--split", "val", "--games", 2, "--policy", trained / "policy.npz",
    )
    assert code == 0, err
    assert output.splitlines()[2].startswith("learned ")


def test_simulate_enforces_split_bounds_and_the_policy_requirement(run):
    code, _, err = run("simulate", "--agent", "random", "--split", "val", "--games", 5000)
    assert code == 1 and "1000" in err
    code, _, err = run("simulate", "--agent", "learned", "--split", "val", "--games", 1)
    assert code == 1 and "policy" in err


def test_a_map_generation_failure_is_a_clean_error(run):
    code, _, err = run("simulate", "--agent", "random", "--split", "val", "--games", 1, "--set", "min_loops=17")
    assert code == 1 and "no val map accepted" in err


def test_watch_prints_one_frame_per_step_with_the_ruling_last(run):
    code, output, err = run("watch", "--agent", "pathfinding", "--seed", 1_000_000, "--plain", "--delay", 0)
    assert code == 0, err
    frames = output.strip().split("\n\n")
    assert frames[0].splitlines()[0].startswith("P")
    assert "Move 0/100" in frames[0]
    last = frames[-1]
    assert "Outcome:" in last
    steps = int([line for line in last.splitlines() if line.startswith("Move ")][0].split()[1].split("/")[0])
    assert len(frames) == steps + 1


def test_watch_falls_back_to_plain_text_when_output_is_not_a_terminal(run):
    code, output, err = run("watch", "--agent", "random", "--seed", 0, "--delay", 0)
    assert code == 0, err
    assert "\x1b[" not in output


def test_pixel_art_watch_redraws_in_place_instead_of_scrolling(run, monkeypatch):
    from pacman import cli as cli_module

    monkeypatch.setattr(cli_module, "use_plain", lambda plain, isatty, environ: False)
    code, output, err = run("watch", "--agent", "pathfinding", "--seed", 1_000_000, "--delay", 0)
    assert code == 0, err
    assert "\n\n" not in output
    frames = output.split("\x1b[H\x1b[2J")
    assert frames[0] == ""                       # the screen is cleared before the first frame
    assert all("▀" in frame for frame in frames[1:])
    assert "Outcome:" in frames[-1]


def test_pacman_debug_re_raises_instead_of_reporting(run, monkeypatch):
    monkeypatch.setenv("PACMAN_DEBUG", "1")
    with pytest.raises(ValueError, match="policy"):
        run("simulate", "--agent", "learned", "--split", "val", "--games", 1)


def test_watch_rejects_a_seed_outside_every_split(run):
    code, _, err = run("watch", "--agent", "random", "--seed", 5_000_000, "--delay", 0)
    assert code == 1 and "split" in err


def test_watch_replays_an_evaluated_game_from_the_json(run, trained):
    assert run("evaluate", "--split", "val", "--policy", trained / "policy.npz")[0] == 0
    _, records = read_eval_json(trained / "eval-val.json")
    record = next(r for r in records if r.agent == "learned")
    code, output, err = run(
        "watch", "--agent", "learned", "--seed", record.seed, "--policy", trained / "policy.npz",
        "--plain", "--delay", 0,
    )
    assert code == 0, err
    last = output.strip().split("\n\n")[-1]
    assert f"Outcome: {record.outcome.value}" in last
    assert f"Move {record.steps}/20" in last
