"""Fixed-grid types, movement, planning and sight (architecture §4.1, §9)."""

from collections import deque
from collections.abc import Callable
from enum import IntEnum
from typing import TypeAlias

ROWS = COLS = 5
Cell: TypeAlias = tuple[int, int]
Passable: TypeAlias = Callable[[Cell], bool]


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DELTAS: tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


def in_grid(cell: Cell) -> bool:
    row, col = cell
    return 0 <= row < ROWS and 0 <= col < COLS


def neighbours(cell: Cell) -> tuple[Cell, ...]:
    """The in-grid neighbours of a cell, in direction order."""
    row, col = cell
    candidates = ((row + d_row, col + d_col) for d_row, d_col in DELTAS)
    return tuple(candidate for candidate in candidates if in_grid(candidate))


def move(cell: Cell, direction: Direction, passable: Passable) -> Cell:
    """The neighbour in that direction, or the cell itself when blocked (FR-1)."""
    d_row, d_col = DELTAS[direction]
    target = (cell[0] + d_row, cell[1] + d_col)
    return target if in_grid(target) and passable(target) else cell


def bfs(source: Cell, passable: Passable) -> dict[Cell, int]:
    """Steps from the source to every passable cell it can reach."""
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        cell = frontier.popleft()
        for neighbour in neighbours(cell):
            if neighbour not in distances and passable(neighbour):
                distances[neighbour] = distances[cell] + 1
                frontier.append(neighbour)
    return distances


def first_step(source: Cell, target: Cell, passable: Passable) -> Direction:
    """First direction of a shortest path; ties go to the earlier direction."""
    distances = bfs(target, passable)
    if source not in distances:
        raise ValueError(f"no path from {source} to {target}")
    for direction in Direction:
        if distances.get(move(source, direction, passable)) == distances[source] - 1:
            return direction
    raise AssertionError("unreachable: a reachable source has a closer neighbour")


def line_of_sight(a: Cell, b: Cell, passable: Passable) -> bool:
    """True when a and b share a row or column with only passable cells between."""
    if a[0] == b[0]:
        row, (lo, hi) = a[0], sorted((a[1], b[1]))
        between = ((row, col) for col in range(lo + 1, hi))
    elif a[1] == b[1]:
        col, (lo, hi) = a[1], sorted((a[0], b[0]))
        between = ((row, col) for row in range(lo + 1, hi))
    else:
        return False
    return all(passable(cell) for cell in between)
