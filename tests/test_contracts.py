"""Shared snapshots must not retain mutable state owned by a previous caller."""

from dataclasses import FrozenInstanceError

import pytest

from pacman.agents.base import Agent
from pacman.env import KnownCell, Observation, Outcome, StepResult
from pacman.episode import EpisodeRecord
from pacman.grid import COLS, DELTAS, ROWS, Direction
from pacman.mapgen import GameMap, Split


def test_fixed_grid_and_action_ids_are_the_architecture_contract():
    assert (ROWS, COLS) == (5, 5)
    assert list(Direction) == [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    assert [int(direction) for direction in Direction] == [0, 1, 2, 3]
    assert DELTAS == ((-1, 0), (0, 1), (1, 0), (0, -1))
    assert [split.value for split in Split] == ["train", "val", "test"]


def test_observation_copies_memory_and_corner_history_without_exposing_true_map():
    visited = {(0, 0)}
    memory = [KnownCell.UNKNOWN] * 25
    observation = Observation((0, 0), (4, 4), visited, memory)
    visited.add((0, 4))
    memory[1] = KnownCell.WALL
    assert observation.visited == frozenset({(0, 0)})
    assert observation.known[1] == KnownCell.UNKNOWN
    assert not hasattr(observation, "walls")
    assert not hasattr(observation, "moves")
    with pytest.raises(FrozenInstanceError):
        observation.ghost = (3, 4)


def test_true_map_copies_its_wall_set():
    walls = {(1, 1)}
    game_map = GameMap(walls)
    walls.add((1, 2))
    assert game_map.walls == frozenset({(1, 1)})


def test_observation_memory_has_the_fixed_grid_length():
    with pytest.raises(ValueError, match="25"):
        Observation((0, 0), (4, 4), frozenset({(0, 0)}), ())


def test_shared_results_carry_an_explicit_outcome():
    observation = Observation((0, 0), (4, 4), frozenset({(0, 0)}), (KnownCell.UNKNOWN,) * 25)
    result = StepResult(observation, -0.05, Outcome.TIE)
    record = EpisodeRecord(0, "test-agent", result.outcome, 100, -5.0)
    assert record.outcome is Outcome.TIE
    assert record.steps == 100


def test_agent_protocol_can_be_implemented_without_game_internals():
    class ExampleAgent:
        name = "example"

        def reset(self, rng):
            self.rng = rng

        def act(self, observation):
            return Direction.UP

        def observe(self, reward, observation, outcome):
            pass

    assert isinstance(ExampleAgent(), Agent)
