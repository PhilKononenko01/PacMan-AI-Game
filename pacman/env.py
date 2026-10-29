"""Observation contracts. Environment gameplay is implemented in issue #8."""

from dataclasses import dataclass
from enum import Enum, IntEnum

from pacman.config import Config
from pacman.grid import COLS, ROWS, Cell, Direction
from pacman.mapgen import GameMap


class Outcome(Enum):
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"


class KnownCell(IntEnum):
    UNKNOWN = 0
    OPEN = 1
    WALL = 2


@dataclass(frozen=True, slots=True)
class Observation:
    pacman: Cell
    ghost: Cell
    visited: frozenset[Cell]
    known: tuple[KnownCell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "visited", frozenset(self.visited))
        object.__setattr__(self, "known", tuple(self.known))
        if len(self.known) != ROWS * COLS:
            raise ValueError("known must contain 25 row-major cells")


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    outcome: Outcome | None


class Environment:
    """Interface scaffold; does not yet run games or decide outcomes."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def reset(self, seed: int, game_map: GameMap | None = None) -> Observation:
        raise NotImplementedError("Environment reset is implemented in issue #8")

    def step(self, action: Direction, t: int) -> StepResult:
        raise NotImplementedError("Environment steps are implemented in issue #8")
