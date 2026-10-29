"""One shared episode loop: reset, act, step, rule, observe, explicit stop (FR-4, FR-8)."""

import random
import subprocess
import sys

import pytest

from pacman.config import Config, GameKnobs
from pacman.env import Environment, KnownCell, Observation, Outcome, StepResult
from pacman.episode import EpisodeRecord, Frame, play_episode
from pacman.grid import Direction
from pacman.mapgen import GameMap
from pacman.referee import Referee
from pacman.rng import derive

UNKNOWN = (KnownCell.UNKNOWN,) * 25
START = Observation((0, 0), (4, 4), frozenset({(0, 0)}), UNKNOWN)
ALIGNED = Observation((0, 2), (4, 2), frozenset({(0, 0)}), UNKNOWN)


class FakeEnvironment:
    """Returns scripted step results; records what the loop asked for."""

    game_map = GameMap(())

    def __init__(self, *results):
        self.results = list(results)
        self.resets = []
        self.steps = []

    def reset(self, seed):
        self.resets.append(seed)
        return START

    def step(self, action, t):
        self.steps.append((action, t))
        return self.results.pop(0)


class SpyAgent:
    name = "spy"

    def __init__(self, action=Direction.UP):
        self.action = action
        self.calls = []

    def reset(self, rng):
        self.calls.append(("reset", rng))

    def act(self, observation):
        self.calls.append(("act", observation))
        return self.action

    def observe(self, reward, observation, outcome):
        self.calls.append(("observe", reward, observation, outcome))


def going(observation=START, reward=-0.05):
    return StepResult(observation, reward, None)


def ended(outcome, observation=START, reward=-0.05):
    return StepResult(observation, reward, outcome)


def test_the_loop_runs_reset_act_step_rule_observe_in_order():
    env = FakeEnvironment(going(), ended(Outcome.WIN, reward=100.95))
    agent = SpyAgent(Direction.RIGHT)
    record = play_episode(env, Referee(100), agent, 7)
    assert env.resets == [7]
    assert env.steps == [(Direction.RIGHT, 1), (Direction.RIGHT, 2)]
    kinds = [call[0] for call in agent.calls]
    assert kinds == ["reset", "act", "observe", "act", "observe"]
    assert record == EpisodeRecord(7, "spy", Outcome.WIN, 2, pytest.approx(100.90))


def test_the_agent_is_seeded_from_the_episode_seed():
    agent = SpyAgent()
    play_episode(FakeEnvironment(ended(Outcome.LOSS)), Referee(100), agent, 11)
    rng = agent.calls[0][1]
    assert isinstance(rng, random.Random)
    assert rng.random() == derive(11, "agent").random()


def test_the_final_ruling_reaches_the_agent_and_no_action_is_asked_after_it():
    env = FakeEnvironment(going(), going(), ended(Outcome.LOSS, reward=-100.05))
    agent = SpyAgent()
    play_episode(env, Referee(100), agent, 1)
    observes = [call for call in agent.calls if call[0] == "observe"]
    assert [call[3] for call in observes] == [None, None, Outcome.LOSS]
    assert agent.calls[-1][0] == "observe"
    assert sum(call[0] == "act" for call in agent.calls) == 3


def test_the_referee_stops_the_episode_with_a_tie():
    env = FakeEnvironment(*[going()] * 10)
    agent = SpyAgent()
    record = play_episode(env, Referee(4), agent, 2)
    assert record.outcome is Outcome.TIE
    assert record.steps == 4
    assert len(env.steps) == 4
    assert agent.calls[-1][3] is Outcome.TIE


def test_a_win_on_the_last_move_is_not_a_tie():
    env = FakeEnvironment(going(), going(), going(), ended(Outcome.WIN, reward=100.95))
    record = play_episode(env, Referee(4), SpyAgent(), 2)
    assert record.outcome is Outcome.WIN
    assert record.steps == 4


def test_the_agent_sees_the_latest_observation_each_turn():
    later = Observation((0, 1), (4, 4), frozenset({(0, 0)}), UNKNOWN)
    env = FakeEnvironment(going(later), ended(Outcome.LOSS, later))
    agent = SpyAgent()
    play_episode(env, Referee(100), agent, 1)
    acts = [call[1] for call in agent.calls if call[0] == "act"]
    assert acts == [START, later]


def test_consecutive_games_restart_the_referee_and_reseed_the_agent():
    referee = Referee(2)
    agent = SpyAgent()
    first = play_episode(FakeEnvironment(*[going()] * 5), referee, agent, 1)
    second = play_episode(FakeEnvironment(*[going()] * 5), referee, agent, 2)
    assert (first.steps, second.steps) == (2, 2)
    resets = [call[1] for call in agent.calls if call[0] == "reset"]
    assert resets[0].random() != resets[1].random()


def test_the_episode_return_sums_every_reward():
    env = FakeEnvironment(going(reward=-0.05), going(reward=0.95), ended(Outcome.LOSS, reward=-100.05))
    record = play_episode(env, Referee(100), SpyAgent(), 1)
    assert record.episode_return == pytest.approx(-99.15)


def test_frames_are_sent_at_reset_and_after_every_step_with_the_ruling_last():
    env = FakeEnvironment(going(ALIGNED), ended(Outcome.WIN, ALIGNED))
    frames = []
    play_episode(env, Referee(50), SpyAgent(), 1, on_frame=frames.append)
    assert all(isinstance(frame, Frame) for frame in frames)
    assert [frame.t for frame in frames] == [0, 1, 2]
    assert [frame.outcome for frame in frames] == [None, None, Outcome.WIN]
    assert [frame.observation for frame in frames] == [START, ALIGNED, ALIGNED]
    assert {frame.max_steps for frame in frames} == {50}


def test_frames_report_whether_the_ghost_sees_pacman_on_the_true_map():
    env = FakeEnvironment(going(ALIGNED), ended(Outcome.LOSS, ALIGNED))
    env.game_map = GameMap({(2, 2)})
    frames = []
    play_episode(env, Referee(50), SpyAgent(), 1, on_frame=frames.append)
    assert [frame.ghost_sees for frame in frames] == [False, False, False]
    env = FakeEnvironment(ended(Outcome.LOSS, ALIGNED))
    frames = []
    play_episode(env, Referee(50), SpyAgent(), 1, on_frame=frames.append)
    assert [frame.ghost_sees for frame in frames] == [False, True]


def test_frames_are_immutable():
    frame = Frame(START, 0, None, False, 100)
    with pytest.raises(AttributeError):
        frame.t = 1


def test_a_still_pacman_and_a_still_ghost_tie_on_move_max_steps():
    config = Config(game=GameKnobs(ghost_move_prob=0.0, max_steps=30))
    agent = SpyAgent(Direction.UP)
    record = play_episode(Environment(config), Referee(config.game.max_steps), agent, 2_000_000)
    assert record == EpisodeRecord(2_000_000, "spy", Outcome.TIE, 30, pytest.approx(-1.5))
    assert agent.calls[-1][3] is Outcome.TIE


def test_a_still_pacman_is_caught_by_a_moving_ghost():
    config = Config()
    record = play_episode(Environment(config), Referee(config.game.max_steps), SpyAgent(Direction.UP), 0)
    assert record.outcome is Outcome.LOSS
    assert record.steps < config.game.max_steps
    assert record.episode_return == pytest.approx(-100 - 0.05 * record.steps)


def test_episode_imports_neither_the_renderer_nor_concrete_agents():
    code = (
        "import sys, pacman.episode;"
        " print(sorted(m for m in sys.modules if m.startswith('pacman')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    loaded = set(eval(result.stdout))
    assert "pacman.render" not in loaded
    assert not any(name.startswith("pacman.agents") for name in loaded)
