from dataclasses import dataclass

from pacman.env import KnownCell, Outcome
from pacman.grid import Cell

RULES_VERSION = "two-pellets-sight-memory-v1"
PAC_FEATURES = "pellet-route-ghost-clock-v1"
GHOST_FEATURES = "sighting-search-v1"
PAC_START = (0, 0)
GHOST_START = (4, 4)


@dataclass(frozen=True, slots=True)
class Rules:
    max_steps: int = 100
    memory_turns: int = 12
    search_distance: int = 3
    search_moves: int = 4
    search_choices: int = 3

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.search_choices > 3:
            raise ValueError("search_choices must be at most 3 for this feature schema")


@dataclass(frozen=True, slots=True)
class PacObservation:
    pacman: Cell
    ghost: Cell
    pellets: frozenset[Cell]
    known: tuple[KnownCell, ...]
    turns_left: int


@dataclass(frozen=True, slots=True)
class GhostObservation:
    cell: Cell
    visible_pacman: Cell | None
    turns_left: int


@dataclass(frozen=True, slots=True)
class TurnResult:
    observation: PacObservation
    pac_reward: float
    ghost_reward: float
    outcome: Outcome | None
