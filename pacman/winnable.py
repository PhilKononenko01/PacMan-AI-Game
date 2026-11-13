"""Which games can be won at all, found by an exact search over Pac-Man's moves.

The ghost's move depends only on the true map, its cell, Pac-Man's cell and its one move draw
per step, and the draws come from the episode seed whatever Pac-Man does. Once the seed is
known a game is a fixed puzzle, so a breadth-first search over Pac-Man's moves, with the whole
map known, finds the shortest win if one exists. Evaluation counts only games that have one.
"""

from functools import lru_cache

from pacman.config import Config, GameKnobs
from pacman.env import Environment
from pacman.ghost import chase_step
from pacman.grid import CORNERS, Cell, Direction, move
from pacman.rng import derive

State = tuple[Cell, Cell, frozenset[Cell]]  # Pac-Man's cell, the ghost's cell, visited corners


def winning_moves(config: Config, seed: int) -> tuple[Direction, ...] | None:
    """The shortest sequence of Pac-Man's moves that wins the seed's game, or None.

    It follows `Environment.step`'s order: Pac-Man moves, walking into the ghost loses, a new
    corner counts and the last one wins, then the ghost takes its drawn move.
    """
    env = Environment(config)
    env.reset(seed)
    game_map, targets, game = env.game_map, frozenset(CORNERS), config.game
    passable = game_map.is_open
    draws = derive(seed, "ghost")  # the ghost's own stream: one draw per step, as in Ghost.act
    ghost_moves = [draws.random() < game.ghost_move_prob for _ in range(game.max_steps)]
    # When the ghost moves on every step or on none, a state's future does not depend on the
    # move number, so a state reached once never needs reaching again later.
    timeless = len(set(ghost_moves)) == 1
    chase: dict[tuple[Cell, Cell], Cell] = {}
    start: State = (env.pacman, env.ghost.cell, frozenset(env.visited))
    seen: set[State] = {start}
    frontier: list[tuple[State, tuple[Direction, ...]]] = [(start, ())]
    for t in range(1, game.max_steps + 1):
        if not timeless:
            seen = set()
        reached = []
        for (pacman, ghost, visited), path in frontier:
            for action in Direction:
                cell = move(pacman, action, passable)
                if cell == ghost:
                    continue
                now = visited
                if cell in targets and cell not in visited:
                    now = visited | {cell}
                    if len(now) == len(targets):
                        return path + (action,)
                after = ghost
                if ghost_moves[t - 1]:
                    if (ghost, cell) not in chase:
                        chase[ghost, cell] = chase_step(ghost, cell, game_map)
                    after = chase[ghost, cell]
                if after == cell:
                    continue
                state = (cell, after, now)
                if state not in seen:
                    seen.add(state)
                    reached.append((state, path + (action,)))
        frontier = reached
        if not frontier:
            return None
    return None


@lru_cache(maxsize=None)
def _winnable(game: GameKnobs, seed: int) -> bool:
    return winning_moves(Config(game=game), seed) is not None


def is_winnable(config: Config, seed: int) -> bool:
    """Whether perfect play could win the seed's game; cached per game knobs and seed."""
    return _winnable(config.game, seed)
