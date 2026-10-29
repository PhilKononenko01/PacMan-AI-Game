"""The environment: one episode's true map, ghost, Pac-Man memory and rewards (§7)."""

from dataclasses import dataclass
from enum import Enum, IntEnum

from pacman.config import Config
from pacman.ghost import Ghost
from pacman.grid import COLS, CORNERS, ROWS, Cell, Direction, move, neighbours
from pacman.mapgen import GameMap, generate, split_of_seed
from pacman.rng import derive

PACMAN_START: Cell = (0, 0)


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
    """Holds one episode at a time; the only owner of the true map and the ghost."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.game_map: GameMap | None = None
        self.outcome: Outcome | None = None
        # Episode state; `reset` gives every field its real value.
        self.pacman: Cell = PACMAN_START
        self.ghost: Ghost | None = None
        self.visited: set[Cell] = set()
        self.known: list[KnownCell] = []

    def reset(self, seed: int, game_map: GameMap | None = None) -> Observation:
        """Start an episode on the seed's map, or on a hand-built map for tests (§7.1)."""
        if game_map is None:
            game_map = generate(seed, split_of_seed(seed), self.config)
        self.game_map = game_map
        self.pacman = PACMAN_START
        self.ghost = Ghost(derive(seed, "ghost"), self.config.game.ghost_move_prob)
        self.visited = {PACMAN_START}
        self.known = [KnownCell.UNKNOWN] * (ROWS * COLS)
        self.outcome = None
        self._sense(game_map)
        return self._observation()

    def step(self, action: Direction, t: int) -> StepResult:
        """PRD §2.4 steps 1 to 6; `t` only stamps the ghost's visit record (§7.2)."""
        game_map, ghost = self.game_map, self.ghost
        if game_map is None or ghost is None:
            raise RuntimeError("reset the environment before stepping")
        if self.outcome is not None:
            raise RuntimeError(f"the episode ended in a {self.outcome.value}")
        rewards = self.config.learner
        reward = rewards.reward_step
        self.pacman = move(self.pacman, action, game_map.is_open)
        if self.pacman == ghost.cell:
            reward += rewards.reward_loss
            return self._end(Outcome.LOSS, reward)
        if self.pacman in CORNERS and self.pacman not in self.visited:
            self.visited.add(self.pacman)
            reward += rewards.reward_corner
            if len(self.visited) == len(CORNERS):
                reward += rewards.reward_win
                return self._end(Outcome.WIN, reward)
        self._sense(game_map)
        ghost.act(self.pacman, game_map, t)
        if ghost.cell == self.pacman:
            reward += rewards.reward_loss
            return self._end(Outcome.LOSS, reward)
        return StepResult(self._observation(), reward, None)

    def _sense(self, game_map: GameMap) -> None:
        """Record each in-grid neighbour of Pac-Man as open or a wall (§7.3)."""
        for row, col in neighbours(self.pacman):
            self.known[row * COLS + col] = (
                KnownCell.OPEN if game_map.is_open((row, col)) else KnownCell.WALL
            )

    def _observation(self) -> Observation:
        assert self.ghost is not None
        return Observation(self.pacman, self.ghost.cell, self.visited, self.known)

    def _end(self, outcome: Outcome, reward: float) -> StepResult:
        self.outcome = outcome
        return StepResult(self._observation(), reward, outcome)
