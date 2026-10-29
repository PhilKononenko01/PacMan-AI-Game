"""The ghost: row/column vision, BFS chase, pursuit of the last sighting, search (§8)."""

import random

from pacman.grid import Cell, Passable, first_step, line_of_sight, move, neighbours
from pacman.mapgen import GameMap
from pacman.rng import pick

GHOST_START: Cell = (4, 4)


class Ghost:
    """One ghost for one episode; the environment calls `act` once per step."""

    def __init__(
        self, rng: random.Random, move_prob: float = 1.0, start: Cell = GHOST_START,
    ) -> None:
        self.cell = start
        self.last_seen: Cell | None = None
        self.last_visit: dict[Cell, int] = {start: 0}
        self.rng = rng
        self.move_prob = move_prob

    def act(self, pacman: Cell, game_map: GameMap, t: int) -> None:
        """One ghost move (§8.2): vision, the move draw, then chase, pursue or search."""
        passable = game_map.is_open
        if line_of_sight(self.cell, pacman, passable):
            self.last_seen = pacman
        if not self.rng.random() < self.move_prob:
            return
        if self.last_seen is not None:
            self.cell = move(self.cell, first_step(self.cell, self.last_seen, passable), passable)
        else:
            self.cell = self._least_recent(passable)
        if self.cell == self.last_seen:
            self.last_seen = None
        self.last_visit[self.cell] = t

    def _least_recent(self, passable: Passable) -> Cell:
        """The least recently visited open neighbour; never visited counts as oldest."""
        candidates = [cell for cell in neighbours(self.cell) if passable(cell)]
        if not candidates:
            raise ValueError(f"the ghost at {self.cell} has no open neighbour to move to")
        oldest = min(self.last_visit.get(cell, -1) for cell in candidates)
        ties = [cell for cell in candidates if self.last_visit.get(cell, -1) == oldest]
        if len(ties) == 1:
            return ties[0]
        return ties[pick(self.rng, len(ties))]
