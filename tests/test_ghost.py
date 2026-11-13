"""The pathfinding ghost: a shortest-path step towards Pac-Man every step, gated by its move draw (FR-5, FR-6)."""

import random

from pacman.ghost import Ghost
from pacman.mapgen import GameMap
from pacman.rng import derive

EMPTY = GameMap(())


class Scripted(random.Random):
    """Returns the given draws in order; running out is a test failure."""

    def __init__(self, *values):
        self.values = list(values)
        self.calls = 0

    def random(self):
        self.calls += 1
        if not self.values:
            raise AssertionError("more draws were taken than the step allows")
        return self.values.pop(0)


def test_the_ghost_starts_in_the_bottom_right_corner():
    assert Ghost(derive(0, "ghost")).cell == (4, 4)


def test_shortest_path_ties_go_to_the_earlier_direction():
    ghost = Ghost(Scripted(0.5))
    ghost.act((0, 0), EMPTY)
    assert ghost.cell == (3, 4)  # UP and LEFT are both shortest; UP comes first


def test_the_ghost_closes_in_on_pacman_without_seeing_him():
    # (0, 0) shares no row or column with (4, 4): sight plays no part in the chase.
    rng = Scripted(*[0.5] * 8)
    ghost = Ghost(rng)
    path = []
    for _ in range(8):
        ghost.act((0, 0), GameMap({(2, 2)}))
        path.append(ghost.cell)
    assert path == [(3, 4), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0)]
    assert rng.calls == 8  # one move draw per step and nothing else


def test_the_ghost_routes_around_walls_on_the_true_map():
    open_row = Ghost(Scripted(0.5), start=(4, 3))
    open_row.act((4, 0), EMPTY)
    assert open_row.cell == (4, 2)
    walled = Ghost(Scripted(0.5), start=(4, 3))
    walled.act((4, 0), GameMap({(4, 2)}))
    assert walled.cell == (3, 3)


def test_a_failed_move_draw_keeps_the_ghost_still():
    rng = Scripted(0.7)
    ghost = Ghost(rng, move_prob=0.5)
    ghost.act((4, 0), EMPTY)
    assert ghost.cell == (4, 4)
    assert rng.calls == 1


def test_move_draw_succeeds_strictly_below_the_probability():
    stays = Ghost(Scripted(0.5), move_prob=0.5)
    stays.act((4, 0), EMPTY)
    assert stays.cell == (4, 4)
    moves = Ghost(Scripted(0.4999), move_prob=0.5)
    moves.act((4, 0), EMPTY)
    assert moves.cell == (4, 3)


def test_move_draw_is_taken_every_step_even_at_probability_one():
    rng = Scripted(0.5, 0.5)
    ghost = Ghost(rng, move_prob=1.0)
    ghost.act((0, 0), EMPTY)
    ghost.act((0, 0), EMPTY)
    assert rng.calls == 2


def test_zero_probability_never_moves():
    ghost = Ghost(Scripted(0.0), move_prob=0.0)
    ghost.act((4, 0), EMPTY)
    assert ghost.cell == (4, 4)


def test_the_same_seed_gives_the_same_path():
    paths = []
    for _ in range(2):
        ghost = Ghost(derive(9, "ghost"), move_prob=0.5)
        path = []
        for _ in range(12):
            ghost.act((0, 0), GameMap({(2, 2)}))
            path.append(ghost.cell)
            if ghost.cell == (0, 0):  # the environment ends the game here
                break
        paths.append(path)
    assert paths[0] == paths[1]
    assert len(paths[0]) > len(set(paths[0]))  # some draws failed, so the ghost waited
