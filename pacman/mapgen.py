"""Map contracts. Generation and split logic are implemented in issue #6."""

from dataclasses import dataclass
from enum import Enum

from pacman.config import Config
from pacman.grid import Cell

MAX_DRAWS = 10_000


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class GameMap:
    walls: frozenset[Cell]

    def __post_init__(self) -> None:
        object.__setattr__(self, "walls", frozenset(self.walls))


class MapGenerationError(RuntimeError):
    """No acceptable map was found within the draw limit."""


def generate(
    seed: int, split: Split, config: Config, max_draws: int = MAX_DRAWS,
) -> GameMap:
    """Return a structurally valid map in the requested layout split."""
    raise NotImplementedError("Map generation is implemented in issue #6")
