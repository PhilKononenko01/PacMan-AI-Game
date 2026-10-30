"""Paired evaluation, summaries, the report layout, replay JSON and the difficulty check."""

import json

import numpy
import pytest

from pacman import evaluate as evaluate_module
from pacman.agents.learned import LearnedAgent
from pacman.agents.pathfinding import PathfindingAgent
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config, GameKnobs
from pacman.env import Environment, Outcome
from pacman.episode import EpisodeRecord, play_episode
from pacman.evaluate import (
    FULL_GAMES,
    Comparison,
    Summary,
    compare,
    difficulty_check,
    format_comparison,
    format_header,
    format_report,
    format_summary_row,
    knob_differences,
    read_eval_json,
    run_split,
    simulate,
    summarise,
    write_eval_json,
)
from pacman.mapgen import Split
from pacman.policy_io import Policy, ValidationSummary
from pacman.referee import Referee
from pacman.stats import mcnemar_one_sided, wilson

CONFIG = Config()


def record(seed, outcome, agent="x", steps=10, episode_return=-0.5):
    return EpisodeRecord(seed, agent, outcome, steps, episode_return)


def records_from(agent, outcomes, steps=10):
    return [record(seed, outcome, agent, steps) for seed, outcome in enumerate(outcomes)]


def test_full_evaluation_size_is_the_prd_constant():
    assert FULL_GAMES == 1_000


def test_run_split_plays_the_first_seeds_of_the_split_in_order():
    records = run_split(CONFIG, PathfindingAgent(), Split.VAL, 5)
    assert [r.seed for r in records] == list(range(1_000_000, 1_000_005))
    assert {r.agent for r in records} == {"pathfinding"}
    expected = [
        play_episode(Environment(CONFIG), Referee(CONFIG.game.max_steps), PathfindingAgent(), seed)
        for seed in range(1_000_000, 1_000_005)
    ]
    assert records == expected
    assert [r.seed for r in run_split(CONFIG, RandomAgent(), Split.TRAIN, 3)] == [0, 1, 2]


def test_every_agent_gets_the_same_seeds():
    random_records = run_split(CONFIG, RandomAgent(), Split.VAL, 4)
    path_records = run_split(CONFIG, PathfindingAgent(), Split.VAL, 4)
    assert [r.seed for r in random_records] == [r.seed for r in path_records]


@pytest.mark.parametrize("games", [0, -1, 1_001])
def test_run_split_stays_inside_the_split(games):
    with pytest.raises(ValueError):
        run_split(CONFIG, RandomAgent(), Split.VAL, games)


def test_run_split_leaves_a_greedy_learned_agent_unchanged():
    agent = LearnedAgent(CONFIG.learner)
    agent.q[:, 1] = 1.0
    before = agent.q.copy()
    run_split(CONFIG, agent, Split.VAL, 3)
    assert numpy.array_equal(agent.q, before)
    assert agent.training is False


@pytest.mark.parametrize("as_iterable", [list, iter])
def test_summarise_counts_rates_intervals_and_steps(as_iterable):
    records = records_from("a", [Outcome.WIN, Outcome.LOSS, Outcome.WIN, Outcome.TIE], steps=12)
    records[2] = record(2, Outcome.WIN, "a", steps=20)
    summary = summarise("a", as_iterable(records))
    assert summary == Summary(
        agent="a", games=4, wins=2, losses=1, ties=1,
        win_interval=wilson(2, 4), loss_interval=wilson(1, 4), tie_interval=wilson(1, 4),
        mean_steps_to_win=16.0,
    )


def test_summarise_without_wins_has_no_mean_and_needs_records():
    assert summarise("a", records_from("a", [Outcome.LOSS, Outcome.TIE])).mean_steps_to_win is None
    with pytest.raises(ValueError):
        summarise("a", [])


def test_compare_pairs_by_seed_and_uses_the_exact_test():
    learned = records_from("learned", [Outcome.WIN, Outcome.WIN, Outcome.LOSS, Outcome.TIE, Outcome.WIN])
    baseline = records_from("random", [Outcome.WIN, Outcome.LOSS, Outcome.WIN, Outcome.LOSS, Outcome.LOSS])
    comparison = compare(learned, baseline)
    assert comparison == Comparison("learned", "random", b=2, c=1, p=mcnemar_one_sided(2, 1))
    assert compare(learned, list(reversed(baseline))) == comparison   # order of records is irrelevant


def test_compare_refuses_different_seed_sets():
    learned = records_from("learned", [Outcome.WIN, Outcome.WIN])
    baseline = records_from("random", [Outcome.LOSS])
    with pytest.raises(ValueError):
        compare(learned, baseline)


def test_summary_rows_reproduce_the_architecture_example():
    random_summary = summarise("random", records_from("random", [Outcome.LOSS] * 1000))
    assert format_summary_row(random_summary) == (
        "random          0   0.0% [ 0.0,  0.4]  1000 100.0% [99.6,100.0]    0   0.0% [ 0.0,  0.4]   n/a"
    )
    assert format_summary_row(Summary(
        agent="pathfinding", games=1000, wins=142, losses=858, ties=0,
        win_interval=wilson(142, 1000), loss_interval=wilson(858, 1000), tie_interval=wilson(0, 1000),
        mean_steps_to_win=13.6,
    )) == "pathfinding   142  14.2% [12.2, 16.5]   858  85.8% [83.5, 87.8]    0   0.0% [ 0.0,  0.4]   13.6"


def test_header_and_comparison_lines_follow_the_layout():
    assert format_header(Split.VAL, 1000, CONFIG.game) == (
        "split val, 1000 games per agent, knobs: ghost_move_prob=1.0 wall_density=0.2"
        " max_walls=6 min_loops=4 max_steps=100"
    )
    line = format_comparison(Comparison("learned", "pathfinding", 30, 10, 0.001))
    assert line.startswith("learned vs pathfinding:  b=30 c=10 one-sided exact McNemar p=")
    assert format_comparison(Comparison("learned", "random", 1, 2, 0.5)).startswith("learned vs random:       b=1 c=2")


def test_report_assembles_header_table_and_comparisons():
    learned = records_from("learned", [Outcome.WIN, Outcome.LOSS])
    random_records = records_from("random", [Outcome.LOSS, Outcome.LOSS])
    summaries = [summarise("random", random_records), summarise("learned", learned)]
    report = format_report(Split.VAL, 2, CONFIG.game, summaries, [compare(learned, random_records)])
    lines = report.split("\n")
    assert lines[0].startswith("split val, 2 games per agent")
    assert lines[1] == "agent         wins                  losses                ties                  steps/win"
    assert lines[2].startswith("random ") and lines[3].startswith("learned ")
    assert lines[4].startswith("learned vs random:")
    assert len(lines) == 5


def test_eval_json_round_trips_configuration_and_every_record(tmp_path):
    config = Config(game=GameKnobs(max_steps=42))
    records = run_split(config, RandomAgent(), Split.VAL, 3) + run_split(config, PathfindingAgent(), Split.VAL, 3)
    path = tmp_path / "eval-val.json"
    write_eval_json(path, config, records)
    data = json.loads(path.read_text())
    assert set(data) == {"config", "records"}
    assert data["config"] == json.loads(config.to_json())
    assert len(data["records"]) == 6
    assert set(data["records"][0]) == {"seed", "agent", "outcome", "steps", "episode_return"}
    assert data["records"][0]["outcome"] in {"win", "loss", "tie"}
    loaded_config, loaded_records = read_eval_json(path)
    assert loaded_config == config
    assert loaded_records == records


def test_knob_differences_name_each_changed_game_knob():
    current = GameKnobs(max_steps=100, wall_density=0.2)
    stored = GameKnobs(max_steps=80, wall_density=0.2, min_loops=3)
    assert knob_differences(current, stored) == {"max_steps": (100, 80), "min_loops": (4, 3)}
    assert knob_differences(current, current) == {}


def policy_with(config, **validation):
    fields = dict(seeds=(1_000_000, 1_000_999), games=1000, wins=300, losses=650, ties=50,
                  mean_steps_to_win=12.5)
    fields.update(validation)
    return Policy(
        q=numpy.zeros((122_880, 4)), config=config, training_seeds=(0, 99),
        learning_curve=(), validation=ValidationSummary(**fields),
    )


def test_difficulty_check_plays_the_baselines_on_validation_without_training(monkeypatch):
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 8)
    check = difficulty_check(CONFIG)
    assert [summary.agent for summary in check.baselines] == ["random", "pathfinding"]
    assert all(summary.games == 8 for summary in check.baselines)
    assert check.stored is None and check.knobs == {}
    expected = summarise("pathfinding", run_split(CONFIG, PathfindingAgent(), Split.VAL, 8))
    assert check.baselines[1] == expected


def test_difficulty_check_shows_the_stored_learned_summary_and_flags_knobs(monkeypatch):
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 5)
    trained = Config(game=GameKnobs(max_steps=80))
    policy = policy_with(trained)
    table_before = policy.q.copy()
    check = difficulty_check(CONFIG, policy)
    assert check.stored == Summary(
        agent="learned", games=1000, wins=300, losses=650, ties=50,
        win_interval=wilson(300, 1000), loss_interval=wilson(650, 1000), tie_interval=wilson(50, 1000),
        mean_steps_to_win=12.5,
    )
    assert check.knobs == {"max_steps": (100, 80)}
    assert numpy.array_equal(policy.q, table_before)
    same = difficulty_check(trained, policy_with(trained, wins=0, losses=950, mean_steps_to_win=None))
    assert same.knobs == {} and same.stored.mean_steps_to_win is None


def test_simulate_summarises_each_agent_on_the_same_games():
    summaries = simulate(CONFIG, [RandomAgent(), PathfindingAgent()], Split.VAL, 6)
    assert [summary.agent for summary in summaries] == ["random", "pathfinding"]
    assert all(summary.games == 6 for summary in summaries)
    assert summaries[1] == summarise("pathfinding", run_split(CONFIG, PathfindingAgent(), Split.VAL, 6))
    with pytest.raises(ValueError):
        simulate(CONFIG, [RandomAgent()], Split.VAL, 5_000)


def test_the_full_validation_run_reproduces_the_architecture_figures():
    random_summary = summarise("random", run_split(CONFIG, RandomAgent(), Split.VAL, FULL_GAMES))
    path_summary = summarise("pathfinding", run_split(CONFIG, PathfindingAgent(), Split.VAL, FULL_GAMES))
    assert (random_summary.wins, random_summary.losses, random_summary.ties) == (0, 1000, 0)
    assert (path_summary.wins, path_summary.losses, path_summary.ties) == (142, 858, 0)
    assert path_summary.mean_steps_to_win == pytest.approx(13.6, abs=0.05)


def test_run_split_refuses_an_agent_still_in_training_mode():
    agent = LearnedAgent(CONFIG.learner, training=True)
    before = agent.q.copy()
    with pytest.raises(ValueError, match="training"):
        run_split(CONFIG, agent, Split.TEST, 2)
    assert numpy.array_equal(agent.q, before)        # nothing was learned from held-out seeds
    agent.training = False
    assert len(run_split(CONFIG, agent, Split.VAL, 2)) == 2
