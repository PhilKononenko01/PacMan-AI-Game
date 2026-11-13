"""The pathfinding baseline and the plan it shares with the features."""

from collections.abc import Mapping
from dataclasses import dataclass
import random

from pacman.env import KnownCell, Observation, Outcome
from pacman.grid import COLS, Cell, Direction, Passable, bfs, first_step

# Equally near corners are chosen in this order.
CORNER_ORDER: tuple[Cell, ...] = ((0, 4), (4, 0), (4, 4))


def remembered_open(observation: Observation) -> Passable:
    """Pac-Man's passable predicate: anything not remembered as a wall counts as open."""
    known = observation.known
    return lambda cell: known[cell[0] * COLS + cell[1]] != KnownCell.WALL


@dataclass(frozen=True, slots=True)
class Plan:
    target: Cell
    step: Direction
    distances: Mapping[Cell, int]
    remaining: int


def plan(observation: Observation) -> Plan:
    """The nearest unvisited target on the remembered map and the first step towards it.

    Equally near targets go in reading order, which for the corners is CORNER_ORDER.
    """
    passable = remembered_open(observation)
    distances = bfs(observation.pacman, passable)
    unvisited = [cell for cell in observation.targets if cell not in observation.visited]
    if not unvisited:
        raise ValueError("every target is visited; there is nothing left to plan")
    target = min(unvisited, key=lambda cell: distances[cell])
    return Plan(target, first_step(observation.pacman, target, passable), distances, len(unvisited))


class PathfindingAgent:
    name = "pathfinding"

    def reset(self, rng: random.Random | None) -> None:
        return None

    def act(self, observation: Observation) -> Direction:
        return plan(observation).step

    def observe(self, reward: float, observation: Observation, outcome: Outcome | None) -> None:
        return None
