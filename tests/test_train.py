"""Training: seeds, the epsilon schedule, curve checkpoints, final validation, outputs (FR-9, FR-12)."""

import json

import numpy
import pytest

from pacman import evaluate as evaluate_module
from pacman import train as train_module
from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.evaluate import Summary
from pacman.referee import Referee
from pacman.mapgen import Split
from pacman.policy_io import ValidationSummary, load_policy
from pacman.train import (
    CURVE_EVERY,
    CURVE_GAMES,
    TrainingResult,
    epsilon_for,
    prepare_output_dir,
    train,
    write_outputs,
)

SHORT = Config(
    game=GameKnobs(max_steps=30),
    learner=LearnerSettings(episodes=5, epsilon_start=0.9, epsilon_end=0.1, epsilon_decay_fraction=0.8),
)


@pytest.fixture
def small_sizes(monkeypatch):
    monkeypatch.setattr(train_module, "CURVE_EVERY", 2)
    monkeypatch.setattr(train_module, "CURVE_GAMES", 3)
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 4)


@pytest.fixture
def spy(monkeypatch):
    """Record every evaluation run and every training episode the loop makes."""
    calls = {"runs": [], "episodes": [], "agents": []}
    real_run_split = train_module.run_split
    real_play = train_module.play_episode

    def run_split(config, agent, split, games):
        before = agent.q.copy()
        records = real_run_split(config, agent, split, games)
        calls["runs"].append({
            "split": split, "games": games, "training": agent.training,
            "unchanged": numpy.array_equal(agent.q, before), "table": agent.q.copy(),
        })
        return records

    def play_episode(environment, referee, agent, seed):
        calls["episodes"].append((seed, agent.epsilon, agent.training))
        calls["agents"].append(agent)
        return real_play(environment, referee, agent, seed)

    monkeypatch.setattr(train_module, "run_split", run_split)
    monkeypatch.setattr(train_module, "play_episode", play_episode)
    return calls


def test_sizes_are_the_prd_constants():
    assert (CURVE_EVERY, CURVE_GAMES) == (2_500, 200)


@pytest.mark.parametrize("episode,expected", [
    (0, 1.0), (12_500, 0.525), (25_000, 0.05), (25_001, 0.05), (49_999, 0.05),
])
def test_epsilon_falls_linearly_over_the_decay_fraction_then_holds(episode, expected):
    assert epsilon_for(episode, LearnerSettings()) == pytest.approx(expected)


def test_epsilon_schedule_uses_the_settings():
    settings = LearnerSettings(episodes=10, epsilon_start=0.5, epsilon_end=0.1, epsilon_decay_fraction=0.5)
    assert [round(epsilon_for(k, settings), 3) for k in range(7)] == [0.5, 0.42, 0.34, 0.26, 0.18, 0.1, 0.1]


def test_episode_k_plays_training_seed_k_with_its_epsilon(small_sizes, spy):
    train(SHORT)
    assert [seed for seed, _, _ in spy["episodes"]] == [0, 1, 2, 3, 4]
    assert [eps for _, eps, _ in spy["episodes"]] == pytest.approx(
        [epsilon_for(k, SHORT.learner) for k in range(5)]
    )
    assert all(training for _, _, training in spy["episodes"])


def test_curve_points_come_every_curve_every_episodes_on_validation_seeds(small_sizes, spy):
    result = train(SHORT)
    curve_runs = spy["runs"][:-1]
    assert [run["games"] for run in curve_runs] == [3, 3]
    assert {run["split"] for run in spy["runs"]} == {Split.VAL}
    assert not any(run["training"] for run in spy["runs"])
    assert all(run["unchanged"] for run in spy["runs"])
    assert [episodes for episodes, _ in result.curve] == [2, 4]
    assert all(0.0 <= rate <= 1.0 for _, rate in result.curve)
    assert result.policy.learning_curve == result.curve


def test_training_continues_after_a_checkpoint_and_the_last_table_is_saved(small_sizes, spy):
    result = train(SHORT)
    at_checkpoint = spy["runs"][1]["table"]
    final = spy["runs"][-1]["table"]
    assert not numpy.array_equal(at_checkpoint, final)
    assert numpy.array_equal(result.policy.q, final)


def test_a_failing_checkpoint_propagates_and_leaves_the_learner_training(small_sizes, spy, monkeypatch):
    def failing(config, agent, split, games):
        raise RuntimeError("validation broke")

    monkeypatch.setattr(train_module, "run_split", failing)
    with pytest.raises(RuntimeError, match="validation broke"):
        train(SHORT)
    assert [seed for seed, _, _ in spy["episodes"]] == [0, 1]   # it broke at the first checkpoint
    learner = spy["agents"][-1]
    assert learner.training is True
    before = learner.q.copy()
    play_episode(Environment(SHORT), Referee(SHORT.game.max_steps), learner, 2)
    assert not numpy.array_equal(learner.q, before)             # updates resume


def test_final_validation_uses_full_games_and_is_stored_in_the_policy(small_sizes, spy):
    result = train(SHORT)
    final_run = spy["runs"][-1]
    assert final_run["games"] == 4
    assert isinstance(result, TrainingResult)
    assert isinstance(result.validation, Summary)
    assert result.validation.games == 4 and result.validation.agent == "learned"
    assert result.policy.validation == ValidationSummary(
        seeds=(1_000_000, 1_000_003), games=4, wins=result.validation.wins,
        losses=result.validation.losses, ties=result.validation.ties,
        mean_steps_to_win=result.validation.mean_steps_to_win,
    )
    assert result.policy.training_seeds == (0, 4)
    assert result.policy.config == SHORT


def test_progress_callback_sees_each_curve_point(small_sizes):
    seen = []
    result = train(SHORT, on_progress=lambda episodes, rate: seen.append((episodes, rate)))
    assert seen == list(result.curve)


def test_a_run_shorter_than_a_checkpoint_has_an_empty_curve(monkeypatch):
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 2)
    result = train(Config(learner=LearnerSettings(episodes=3)))
    assert result.curve == ()
    assert result.policy.learning_curve == ()


def test_training_is_repeatable(small_sizes):
    first, second = train(SHORT), train(SHORT)
    assert first.curve == second.curve
    assert first.policy.q.tobytes() == second.policy.q.tobytes()
    assert first.validation == second.validation


def test_outputs_are_the_four_documented_files(small_sizes, tmp_path):
    result = train(SHORT)
    out = tmp_path / "runs" / "short"
    write_outputs(out, result)
    assert sorted(path.name for path in out.iterdir()) == ["config.json", "curve.csv", "curve.png", "policy.npz"]
    assert (out / "config.json").read_text() == SHORT.to_json()
    assert (out / "curve.csv").read_text() == "episodes,win_rate\n" + "".join(
        f"{episodes},{rate}\n" for episodes, rate in result.curve
    )
    assert (out / "curve.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    policy = load_policy(out / "policy.npz")
    assert numpy.array_equal(policy.q, result.policy.q)
    assert policy.config == SHORT
    assert policy.learning_curve == result.curve
    assert policy.validation == result.policy.validation


def test_an_existing_policy_in_the_output_directory_is_refused(small_sizes, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "policy.npz").write_bytes(b"")
    with pytest.raises(FileExistsError):
        prepare_output_dir(out)
    with pytest.raises(FileExistsError):
        write_outputs(out, train(SHORT))
    assert (out / "policy.npz").read_bytes() == b""


def test_prepare_output_dir_creates_missing_directories(tmp_path):
    out = tmp_path / "a" / "b"
    assert prepare_output_dir(out) == out
    assert out.is_dir()
    assert prepare_output_dir(out) == out   # empty existing directory is fine


def test_the_policy_is_saved_before_the_curve_is_drawn(small_sizes, tmp_path, monkeypatch):
    """A broken plotting stack must not lose a finished training run."""
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "pacman.plot" or name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    result = train(SHORT)
    out = tmp_path / "run"
    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(ImportError):
        write_outputs(out, result)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert numpy.array_equal(load_policy(out / "policy.npz").q, result.policy.q)
