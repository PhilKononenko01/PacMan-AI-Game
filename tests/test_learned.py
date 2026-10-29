"""Tabular Q-learning: epsilon-greedy training, greedy play, the update, cache reset (FR-8, FR-9)."""

import random

import numpy
import pytest

from pacman.agents import learned as learned_module
from pacman.agents.base import Agent
from pacman.agents.features import state_index
from pacman.agents.learned import LearnedAgent
from pacman.config import Config, LearnerSettings
from pacman.env import Environment, KnownCell, Observation, Outcome
from pacman.episode import play_episode
from pacman.grid import COLS, Direction
from pacman.policy_io import TABLE_SHAPE
from pacman.referee import Referee
from pacman.rng import derive

UP, RIGHT, DOWN, LEFT = Direction
SETTINGS = LearnerSettings()


def memory(open=(), wall=()):
    known = [KnownCell.UNKNOWN] * 25
    for row, col in open:
        known[row * COLS + col] = KnownCell.OPEN
    for row, col in wall:
        known[row * COLS + col] = KnownCell.WALL
    return tuple(known)


START = Observation((0, 0), (4, 4), frozenset({(0, 0)}), memory(open=[(0, 1), (1, 0)]))
NEXT = Observation((0, 1), (4, 3), frozenset({(0, 0)}), memory(open=[(0, 0), (0, 2), (1, 1)]))
WON = Observation((4, 4), (3, 4), frozenset({(0, 0), (0, 4), (4, 0), (4, 4)}), memory())
CAUGHT = Observation((2, 2), (2, 2), frozenset({(0, 0)}), memory())


class Scripted(random.Random):
    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        if not self.values:
            raise AssertionError("more draws were taken than the step allows")
        return self.values.pop(0)


def trainer(*draws, epsilon=0.0, **settings):
    agent = LearnedAgent(LearnerSettings(**settings), training=True)
    agent.epsilon = epsilon
    agent.reset(Scripted(*draws))
    return agent


def test_learned_agent_satisfies_the_protocol_with_a_zero_table():
    agent = LearnedAgent(SETTINGS)
    assert isinstance(agent, Agent)
    assert agent.name == "learned"
    assert agent.q.shape == TABLE_SHAPE and agent.q.dtype == numpy.float64
    assert not agent.q.any()
    assert agent.training is False


def test_a_given_table_is_shared_not_copied():
    table = numpy.zeros(TABLE_SHAPE)
    agent = LearnedAgent(SETTINGS, q=table)
    assert agent.q is table
    with pytest.raises(ValueError):
        LearnedAgent(SETTINGS, q=numpy.zeros((10, 4)))
    with pytest.raises(ValueError):
        LearnedAgent(SETTINGS, q=numpy.zeros(TABLE_SHAPE, dtype=numpy.float32))


def test_greedy_play_takes_the_first_best_action_without_drawing_or_learning():
    agent = LearnedAgent(SETTINGS)
    rng = Scripted()
    agent.reset(rng)
    s = state_index(START)
    agent.q[s] = [0.0, 5.0, 5.0, 1.0]
    assert agent.act(START) == RIGHT
    agent.q[s] = [3.0, 3.0, 3.0, 3.0]
    assert agent.act(START) == UP
    before = agent.q.copy()
    agent.observe(-0.05, NEXT, None)
    agent.observe(100.95, WON, Outcome.WIN)
    assert numpy.array_equal(agent.q, before)
    assert rng.calls == 0


def test_training_explores_with_a_second_draw_when_u_is_below_epsilon():
    agent = trainer(0.5, 0.74, epsilon=1.0)
    assert agent.act(START) == DOWN           # pick: int(0.74 * 4) == 2
    assert agent.rng.calls == 2


def test_training_is_greedy_on_one_draw_when_u_is_not_below_epsilon():
    agent = trainer(0.5, epsilon=0.5)        # strict: 0.5 < 0.5 is false
    agent.q[state_index(START)] = [0.0, 0.0, 2.0, 0.0]
    assert agent.act(START) == DOWN
    assert agent.rng.calls == 1


def test_update_on_a_continuing_step_bootstraps_from_the_next_state():
    agent = trainer(0.9, learning_rate=0.1, discount=0.99)
    s, s_next = state_index(START), state_index(NEXT)
    agent.q[s_next] = [1.0, 2.0, 3.0, 4.0]
    assert agent.act(START) == UP
    agent.observe(-0.05, NEXT, None)
    assert agent.q[s, UP] == pytest.approx(0.1 * (-0.05 + 0.99 * 4.0))
    assert agent.q[s, RIGHT] == 0.0 and agent.q[s, DOWN] == 0.0 and agent.q[s, LEFT] == 0.0
    assert numpy.count_nonzero(agent.q) == 5


def test_update_uses_the_learner_settings():
    agent = trainer(0.9, learning_rate=0.5, discount=0.5)
    s, s_next = state_index(START), state_index(NEXT)
    agent.q[s] = [2.0, 0.0, 0.0, 0.0]
    agent.q[s_next] = [0.0, 0.0, 0.0, 10.0]
    agent.act(START)
    agent.observe(1.0, NEXT, None)
    assert agent.q[s, UP] == pytest.approx(2.0 + 0.5 * (1.0 + 0.5 * 10.0 - 2.0))


@pytest.mark.parametrize("reward,observation,outcome", [
    (100.95, WON, Outcome.WIN), (-100.05, CAUGHT, Outcome.LOSS),
])
def test_win_and_loss_use_the_reward_alone_and_never_encode_the_final_observation(
    reward, observation, outcome,
):
    agent = trainer(0.9)
    s = state_index(START)
    agent.act(START)
    agent.observe(reward, observation, outcome)   # these observations cannot be encoded
    assert agent.q[s, UP] == pytest.approx(0.1 * reward)


def test_a_tie_bootstraps_by_default():
    agent = trainer(0.9)
    assert agent.settings.tie_update == "bootstrap"
    s, s_tie = state_index(START), state_index(NEXT)
    agent.q[s_tie] = [0.0, 7.0, 0.0, 0.0]
    agent.act(START)
    agent.observe(-0.05, NEXT, Outcome.TIE)
    assert agent.q[s, UP] == pytest.approx(0.1 * (-0.05 + 0.99 * 7.0))


def test_a_tie_can_be_terminal_by_setting():
    agent = trainer(0.9, tie_update="terminal")
    s, s_tie = state_index(START), state_index(NEXT)
    agent.q[s_tie] = [0.0, 7.0, 0.0, 0.0]
    agent.act(START)
    agent.observe(-0.05, NEXT, Outcome.TIE)
    assert agent.q[s, UP] == pytest.approx(0.1 * -0.05)


def test_the_next_state_index_is_cached_while_the_game_continues(monkeypatch):
    calls = []
    real = learned_module.state_index
    monkeypatch.setattr(learned_module, "state_index", lambda obs: calls.append(obs) or real(obs))
    agent = trainer(0.9, 0.9)
    agent.act(START)
    assert len(calls) == 1
    agent.observe(-0.05, NEXT, None)
    assert len(calls) == 2
    agent.act(NEXT)
    assert len(calls) == 2


@pytest.mark.parametrize("outcome,observation", [
    (Outcome.WIN, WON), (Outcome.LOSS, CAUGHT), (Outcome.TIE, NEXT),
])
def test_no_transition_survives_a_final_outcome(monkeypatch, outcome, observation):
    calls = []
    real = learned_module.state_index
    monkeypatch.setattr(learned_module, "state_index", lambda obs: calls.append(obs) or real(obs))
    agent = trainer(0.9, 0.9)
    agent.act(START)
    agent.observe(-0.05, observation, outcome)
    before = agent.q.copy()
    agent.reset(Scripted(0.9))
    agent.act(NEXT)
    assert calls[-1] is NEXT                     # recomputed, not taken from a stale cache
    assert numpy.array_equal(agent.q, before)    # the table survives the reset
    with pytest.raises(RuntimeError):
        LearnedAgent(SETTINGS, training=True).observe(-0.05, NEXT, None)


def test_reset_between_games_clears_a_pending_transition_but_keeps_the_table():
    agent = trainer(0.9, 0.9)
    agent.act(START)
    agent.observe(-0.05, NEXT, None)              # continuing: s' cached
    learned_before = agent.q.copy()
    agent.reset(Scripted(0.9))
    with pytest.raises(RuntimeError):
        agent.observe(-0.05, NEXT, None)          # nothing pending in the new game
    assert numpy.array_equal(agent.q, learned_before)


def test_whole_training_games_change_the_table_and_evaluation_games_do_not():
    config = Config()
    agent = LearnedAgent(config.learner, training=True)
    agent.epsilon = 0.5
    for seed in range(10):
        play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
    assert numpy.count_nonzero(agent.q) > 0
    trained = agent.q.copy()
    agent.training = False
    records = [
        play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
        for seed in (1_000_000, 1_000_001)
    ]
    assert numpy.array_equal(agent.q, trained)
    assert all(record.outcome in Outcome for record in records)


def test_training_is_repeatable_to_the_byte():
    tables = []
    for _ in range(2):
        config = Config()
        agent = LearnedAgent(config.learner, training=True)
        agent.epsilon = 0.3
        for seed in range(25):
            play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
        tables.append(agent.q.tobytes())
    assert tables[0] == tables[1]


def test_evaluation_draws_nothing_from_the_episode_stream():
    agent = LearnedAgent(SETTINGS)
    rng = derive(4, "agent")
    before = rng.getstate()
    agent.reset(rng)
    agent.act(START)
    assert rng.getstate() == before
