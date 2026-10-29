"""The ADR-2 state features and the 122,880-row index, from observations only."""

import pytest

from pacman.agents.features import STATES, Features, features, state_index
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config
from pacman.env import Environment, KnownCell, Observation
from pacman.episode import play_episode
from pacman.grid import COLS, CORNERS, Direction
from pacman.policy_io import TABLE_SHAPE
from pacman.referee import Referee


def memory(open=(), wall=()):
    known = [KnownCell.UNKNOWN] * 25
    for row, col in open:
        known[row * COLS + col] = KnownCell.OPEN
    for row, col in wall:
        known[row * COLS + col] = KnownCell.WALL
    return tuple(known)


def observation(pacman=(0, 0), ghost=(4, 4), visited=((0, 0),), known=None):
    visited = set(visited) | ({pacman} & set(CORNERS))  # a corner is visited on arrival
    return Observation(pacman, ghost, frozenset(visited), memory() if known is None else known)


def test_state_count_matches_the_policy_table():
    assert STATES == 122_880 == TABLE_SHAPE[0]


def test_features_and_index_of_the_start_observation():
    start = observation(known=memory(open=[(0, 1), (1, 0)]))
    result = features(start)
    assert result == Features(step=1, open_mask=0b0110, offset=79, distance=3, seen=0, remaining=2)
    assert result.index == ((((1 * 16 + 6) * 80 + 79) * 4 + 3) * 2 + 0) * 3 + 2 == 44156
    assert state_index(start) == 44156


def test_the_largest_features_fill_the_table_exactly():
    assert Features(3, 15, 79, 3, 1, 2).index == STATES - 1
    assert Features(0, 0, 0, 0, 0, 0).index == 0


def test_step_follows_the_pathfinding_plan():
    assert features(observation(pacman=(2, 2))).step == int(Direction.UP)
    assert features(observation(pacman=(2, 2), visited=[(0, 0), (0, 4), (4, 0)])).step == int(Direction.RIGHT)
    assert features(observation(known=memory(wall=[(0, 1)]))).step == int(Direction.DOWN)


def test_open_mask_counts_only_in_grid_neighbours_known_open():
    centre = (2, 2)
    all_open = memory(open=[(1, 2), (2, 3), (3, 2), (2, 1)])
    assert features(observation(pacman=centre, known=all_open)).open_mask == 0b1111
    assert features(observation(pacman=centre)).open_mask == 0           # unknown is not open
    one_wall = memory(open=[(1, 2), (3, 2), (2, 1)], wall=[(2, 3)])
    assert features(observation(pacman=centre, known=one_wall)).open_mask == 0b1101
    corner = memory(open=[(0, 1), (1, 0)])
    assert features(observation(known=corner)).open_mask == 0b0110         # UP and LEFT are off the grid


@pytest.mark.parametrize("pacman,ghost,expected", [
    ((4, 4), (0, 0), 0),
    ((0, 0), (4, 4), 79),
    ((2, 2), (2, 1), 39),
    ((2, 2), (2, 3), 40),
    ((2, 2), (1, 2), 31),
    ((2, 2), (3, 2), 48),
    ((0, 4), (4, 0), 71),
    ((4, 0), (0, 4), 8),
])
def test_offset_skips_the_impossible_centre(pacman, ghost, expected):
    assert features(observation(pacman=pacman, ghost=ghost)).offset == expected


def test_every_offset_value_is_used_exactly_once():
    seen = set()
    for pacman_row in range(5):
        for pacman_col in range(5):
            for ghost_row in range(5):
                for ghost_col in range(5):
                    if (pacman_row, pacman_col) == (ghost_row, ghost_col):
                        continue
                    seen.add(features(observation(
                        pacman=(pacman_row, pacman_col), ghost=(ghost_row, ghost_col),
                    )).offset)
    assert seen == set(range(80))


@pytest.mark.parametrize("ghost,expected", [
    ((0, 1), 0), ((0, 2), 1), ((0, 3), 2), ((0, 4), 3), ((4, 4), 3), ((1, 1), 1),
])
def test_distance_buckets_at_1_2_3_and_4_or_more(ghost, expected):
    assert features(observation(ghost=ghost)).distance == expected


def test_distance_is_measured_on_the_remembered_map():
    assert features(observation(ghost=(0, 2))).distance == 1
    assert features(observation(ghost=(0, 2), known=memory(wall=[(0, 1)]))).distance == 3


def test_seen_is_line_of_sight_on_the_remembered_map():
    assert features(observation(ghost=(0, 3))).seen == 1                       # unknown counts open
    assert features(observation(ghost=(0, 3), known=memory(wall=[(0, 1)]))).seen == 0
    assert features(observation(ghost=(0, 3), known=memory(wall=[(1, 1)]))).seen == 1
    assert features(observation(ghost=(1, 3))).seen == 0


@pytest.mark.parametrize("visited,expected", [
    ([(0, 0)], 2), ([(0, 0), (0, 4)], 1), ([(0, 0), (0, 4), (4, 0)], 0),
])
def test_remaining_is_unvisited_corners_minus_one(visited, expected):
    assert features(observation(pacman=(2, 2), visited=visited)).remaining == expected


def test_a_capture_or_a_finished_game_is_never_a_state():
    with pytest.raises(ValueError):
        features(observation(pacman=(2, 2), ghost=(2, 2)))
    with pytest.raises(ValueError):
        features(observation(pacman=(4, 4), visited=[(0, 0), (0, 4), (4, 0), (4, 4)]))


def test_features_are_immutable_and_within_range():
    result = features(observation())
    with pytest.raises(AttributeError):
        result.step = 2
    assert 0 <= result.index < STATES


def test_every_observation_of_real_games_indexes_inside_the_table():
    class Indexing(RandomAgent):
        indices = []

        def act(self, observation):
            self.indices.append(state_index(observation))
            return super().act(observation)

    config = Config()
    agent = Indexing()
    for seed in range(30):
        play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
    assert agent.indices
    assert all(0 <= index < STATES for index in agent.indices)
    assert len(set(agent.indices)) > 50
