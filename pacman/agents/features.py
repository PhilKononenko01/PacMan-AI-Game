"""The state features and the table index, from an observation alone."""

from dataclasses import dataclass

from pacman.agents.pathfinding import plan, remembered_open
from pacman.env import KnownCell, Observation
from pacman.grid import COLS, DELTAS, Direction, in_grid, line_of_sight

STATES = 4 * 16 * 80 * 4 * 2 * 3  # 122,880 rows; policy_io.TABLE_SHAPE must agree


@dataclass(frozen=True, slots=True)
class Features:
    step: int        # first step to the nearest unvisited corner, 0..3
    open_mask: int   # bit i set when the neighbour in direction i is known open, 0..15
    offset: int      # ghost's (row, col) offset from Pac-Man, centre removed, 0..79
    distance: int    # shortest-path distance to the ghost, bucketed 1, 2, 3, 4+, as 0..3
    seen: int        # line of sight on the remembered map, 0..1
    remaining: int   # unvisited corners minus one, 0..2

    @property
    def index(self) -> int:
        return (
            (((self.step * 16 + self.open_mask) * 80 + self.offset) * 4 + self.distance) * 2
            + self.seen
        ) * 3 + self.remaining


def features(observation: Observation) -> Features:
    """Encode an observation the game is still running from; captures are never states."""
    (pacman_row, pacman_col), (ghost_row, ghost_col) = observation.pacman, observation.ghost
    if observation.pacman == observation.ghost:
        raise ValueError("a captured Pac-Man is never a state")
    current = plan(observation)  # raises when no corner remains
    known = observation.known
    open_mask = 0
    for direction in Direction:
        d_row, d_col = DELTAS[direction]
        cell = (pacman_row + d_row, pacman_col + d_col)
        if in_grid(cell) and known[cell[0] * COLS + cell[1]] == KnownCell.OPEN:
            open_mask |= 1 << direction
    k = (ghost_row - pacman_row + 4) * 9 + (ghost_col - pacman_col + 4)
    offset = k - 1 if k > 40 else k
    distance = min(current.distances[observation.ghost], 4) - 1
    seen = int(line_of_sight(observation.ghost, observation.pacman, remembered_open(observation)))
    return Features(int(current.step), open_mask, offset, distance, seen, current.remaining - 1)


def state_index(observation: Observation) -> int:
    return features(observation).index
