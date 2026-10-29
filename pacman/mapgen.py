"""Random 5x5 maps: candidate draws, structural filters and held-out splits (§6)."""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import random

from pacman.config import Config, GameKnobs
from pacman.grid import COLS, CORNERS, ROWS, Cell, bfs
from pacman.rng import derive

MAX_DRAWS = 10_000


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# PRD §2.7 / architecture §4.2: what the splits are, not configuration.
SPLIT_SEEDS: dict[Split, range] = {
    Split.TRAIN: range(0, 1_000_000),
    Split.VAL: range(1_000_000, 1_001_000),
    Split.TEST: range(2_000_000, 2_001_000),
}
SPLIT_BUCKETS: dict[Split, range] = {
    Split.TRAIN: range(0, 80),
    Split.VAL: range(80, 90),
    Split.TEST: range(90, 100),
}

ALL_CELLS: tuple[Cell, ...] = tuple((row, col) for row in range(ROWS) for col in range(COLS))
NON_CORNERS: tuple[Cell, ...] = tuple(cell for cell in ALL_CELLS if cell not in CORNERS)


@dataclass(frozen=True, slots=True)
class GameMap:
    walls: frozenset[Cell]

    def __post_init__(self) -> None:
        object.__setattr__(self, "walls", frozenset(self.walls))


class MapGenerationError(RuntimeError):
    """No acceptable map was found within the draw limit."""


def split_of_seed(seed: int) -> Split:
    """The split whose seed range holds the seed."""
    for split, seeds in SPLIT_SEEDS.items():
        if seed in seeds:
            return split
    raise ValueError(f"seed {seed} belongs to no split")


def draw_candidate(rng: random.Random, wall_density: float) -> GameMap:
    """One draw per non-corner cell in row-major order; corners stay open (§6.1)."""
    return GameMap(cell for cell in NON_CORNERS if rng.random() < wall_density)


def is_connected(game_map: GameMap) -> bool:
    """Every open cell is reachable from the start corner (§6.2 check 1)."""
    walls = game_map.walls
    reached = bfs((0, 0), lambda cell: cell not in walls)
    return len(reached) == ROWS * COLS - len(walls)


def loop_count(game_map: GameMap) -> int:
    """Independent loops of a connected map: open edges minus open cells plus one."""
    walls = game_map.walls
    open_cells = [cell for cell in ALL_CELLS if cell not in walls]
    edges = 0
    for row, col in open_cells:
        edges += col + 1 < COLS and (row, col + 1) not in walls
        edges += row + 1 < ROWS and (row + 1, col) not in walls
    return edges - len(open_cells) + 1


def is_winnable(game_map: GameMap, game: GameKnobs) -> bool:
    """PRD §2.3 winnability: at most max_walls walls and at least min_loops loops."""
    return len(game_map.walls) <= game.max_walls and loop_count(game_map) >= game.min_loops


def layout_bytes(game_map: GameMap) -> bytes:
    """The hashed form: a format tag plus row-major wall bits (§6.3)."""
    bits = b"".join(b"1" if cell in game_map.walls else b"0" for cell in ALL_CELLS)
    return b"5x5:" + bits


def split_bucket(game_map: GameMap) -> int:
    return int.from_bytes(sha256(layout_bytes(game_map)).digest(), "big") % 100


def split_of_map(game_map: GameMap) -> Split:
    """The split a layout belongs to, decided by its hash bucket alone."""
    bucket = split_bucket(game_map)
    for split, buckets in SPLIT_BUCKETS.items():
        if bucket in buckets:
            return split
    raise AssertionError(f"unreachable: bucket {bucket} is outside 0..99")


def generate(
    seed: int, split: Split, config: Config, max_draws: int = MAX_DRAWS,
) -> GameMap:
    """Return a structurally valid map in the requested layout split."""
    game = config.game
    rng = derive(seed, "map")
    for _ in range(max_draws):
        candidate = draw_candidate(rng, game.wall_density)
        if (
            is_connected(candidate)
            and is_winnable(candidate, game)
            and split_of_map(candidate) is split
        ):
            return candidate
    raise MapGenerationError(
        f"no {split.value} map accepted for seed {seed} after {max_draws} draws "
        f"(wall_density={game.wall_density}, max_walls={game.max_walls}, "
        f"min_loops={game.min_loops})"
    )
