"""Fixed-grid types. Movement and planning are implemented in issue #5."""

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
