"""The random and pathfinding baselines (FR-8, PRD §2.8)."""

import random

import pytest

from pacman.agents.base import Agent
from pacman.agents.pathfinding import CORNER_ORDER, PathfindingAgent, Plan, plan
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config, GameKnobs
from pacman.env import Environment, KnownCell, Observation, Outcome
from pacman.episode import play_episode
from pacman.grid import COLS, Direction
from pacman.referee import Referee
from pacman.rng import derive, pick

UP, RIGHT, DOWN, LEFT = Direction


def memory(open=(), wall=()):
    known = [KnownCell.UNKNOWN] * 25
    for row, col in open:
        known[row * COLS + col] = KnownCell.OPEN
    for row, col in wall:
        known[row * COLS + col] = KnownCell.WALL
    return tuple(known)


def observation(pacman=(0, 0), ghost=(4, 4), visited=((0, 0),), known=None):
    return Observation(pacman, ghost, frozenset(visited), memory() if known is None else known)


class Counting(random.Random):
    calls = 0

    def random(self):
        self.calls += 1
        return super().random()


def test_both_baselines_satisfy_the_agent_protocol():
    assert isinstance(RandomAgent(), Agent)
    assert isinstance(PathfindingAgent(), Agent)
    assert (RandomAgent().name, PathfindingAgent().name) == ("random", "pathfinding")


def test_random_agent_draws_one_uniform_action_per_step_from_its_episode_stream():
    agent = RandomAgent()
    rng = Counting(0)
    rng.setstate(derive(5, "agent").getstate())
    agent.reset(rng)
    actions = [agent.act(observation()) for _ in range(200)]
    assert rng.calls == 200
    control = derive(5, "agent")
    assert actions == [Direction(pick(control, 4)) for _ in range(200)]
    assert set(actions) == set(Direction)


def test_random_agent_ignores_observations_and_outcomes():
    agent = RandomAgent()
    agent.reset(derive(0, "agent"))
    agent.observe(-0.05, observation(), None)
    agent.observe(-100.05, observation(), Outcome.LOSS)
    assert isinstance(agent.act(observation(pacman=(2, 2), ghost=(2, 3))), Direction)


def test_corner_order_is_the_prd_tie_break():
    assert CORNER_ORDER == ((0, 4), (4, 0), (4, 4))


def test_plan_targets_the_nearest_unvisited_corner_in_fixed_order():
    result = plan(observation())
    assert isinstance(result, Plan)
    assert result.target == (0, 4)       # (0, 4) and (4, 0) are both 4 away
    assert result.step == RIGHT
    assert result.remaining == 3
    assert result.distances[(4, 4)] == 8


def test_plan_prefers_a_strictly_nearer_corner_whatever_the_order():
    result = plan(observation(pacman=(3, 1)))
    assert result.target == (4, 0)
    assert result.step == DOWN               # DOWN then LEFT are both shortest; DOWN is first


def test_plan_breaks_path_ties_in_direction_order():
    result = plan(observation(pacman=(2, 2), visited=[(0, 0), (0, 4), (4, 0)]))
    assert result.target == (4, 4)
    assert result.step == RIGHT
    assert result.remaining == 1


def test_plan_routes_round_remembered_walls_and_treats_unknown_as_open():
    known = memory(open=[(1, 0)], wall=[(0, 1)])
    result = plan(observation(known=known))
    assert result.target == (4, 0)           # (0, 4) is now 6 away through the unknown cells
    assert result.step == DOWN
    assert result.distances[(0, 4)] == 6


def test_plan_replans_from_the_latest_memory_each_step():
    agent = PathfindingAgent()
    agent.reset(None)
    assert agent.act(observation()) == RIGHT
    sensed_wall = observation(pacman=(0, 1), known=memory(open=[(0, 0), (1, 1)], wall=[(0, 2)]))
    assert agent.act(sensed_wall) == DOWN


def test_pathfinding_ignores_the_ghost():
    assert PathfindingAgent().act(observation(ghost=(0, 1))) == RIGHT
    assert plan(observation(ghost=(0, 1))).step == plan(observation(ghost=(4, 4))).step


def test_pathfinding_agent_ignores_the_generator_and_outcomes():
    agent = PathfindingAgent()
    agent.reset(derive(3, "agent"))
    agent.observe(-0.05, observation(), None)
    agent.observe(0.0, observation(), Outcome.TIE)
    assert agent.act(observation()) == RIGHT


def test_baselines_play_whole_games_deterministically():
    config = Config()
    for agent_type in (RandomAgent, PathfindingAgent):
        records = [
            play_episode(Environment(config), Referee(config.game.max_steps), agent_type(), seed)
            for _ in range(2)
            for seed in (0, 1_000_000)
        ]
        assert records[:2] == records[2:]
        assert all(record.outcome in Outcome for record in records)


def test_pathfinding_walks_into_a_still_ghost_on_the_nearest_corner():
    config = Config(game=GameKnobs(ghost_move_prob=0.0))
    env = Environment(config)
    record = play_episode(env, Referee(config.game.max_steps), PathfindingAgent(), 2_000_000)
    assert record.outcome is Outcome.LOSS
    assert env.visited == {(0, 0), (0, 4)}   # from (0, 4) the nearest corner is (4, 4)
    assert record.steps == 8


def test_plan_needs_an_unvisited_corner():
    with pytest.raises(ValueError):
        plan(observation(pacman=(4, 4), visited=[(0, 0), (0, 4), (4, 0), (4, 4)]))
