"""The ghost: it always knows where Pac-Man is and takes a shortest-path step towards him."""

import random

from pacman.grid import Cell, first_step, move
from pacman.mapgen import GameMap

GHOST_START: Cell = (4, 4)


def chase_step(cell: Cell, pacman: Cell, game_map: GameMap) -> Cell:
    """The ghost's next cell: the first step of a shortest true-map path to Pac-Man."""
    passable = game_map.is_open
    return move(cell, first_step(cell, pacman, passable), passable)


class Ghost:
    """One ghost for one episode; the environment calls `act` once per step."""

    def __init__(
        self, rng: random.Random, move_prob: float = 1.0, start: Cell = GHOST_START,
    ) -> None:
        self.cell = start
        self.rng = rng
        self.move_prob = move_prob

    def act(self, pacman: Cell, game_map: GameMap) -> None:
        """The move draw, then the first step of a shortest true-map path to Pac-Man.

        The environment ends the game before the ghost acts if Pac-Man is on its cell.
        """
        if not self.rng.random() < self.move_prob:
            return
        self.cell = chase_step(self.cell, pacman, game_map)
