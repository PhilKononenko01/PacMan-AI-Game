"""Ghost vision, chase, pursuit, search and draw order on hand-built maps (FR-5, FR-6)."""

import random

import pytest

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


def test_start_state_matches_the_architecture_table():
    ghost = Ghost(derive(0, "ghost"))
    assert ghost.cell == (4, 4)
    assert ghost.last_seen is None
    assert ghost.last_visit == {(4, 4): 0}


def test_chase_takes_the_first_step_towards_a_visible_pacman():
    ghost = Ghost(Scripted(0.5))
    ghost.act((4, 0), EMPTY, 1)
    assert ghost.cell == (4, 3)
    assert ghost.last_seen == (4, 0)
    assert ghost.last_visit[(4, 3)] == 1


def test_a_wall_between_blocks_vision():
    ghost = Ghost(Scripted(0.5, 0.0))
    ghost.act((4, 0), GameMap({(4, 2)}), 1)
    assert ghost.last_seen is None


def test_vision_needs_a_shared_row_or_column():
    ghost = Ghost(Scripted(0.5, 0.0))
    ghost.act((3, 3), EMPTY, 1)
    assert ghost.last_seen is None


def test_pursue_heads_for_the_last_sighting_with_ties_in_direction_order():
    ghost = Ghost(Scripted(0.5))
    ghost.last_seen = (0, 0)
    ghost.act((1, 3), EMPTY, 1)
    assert ghost.cell == (3, 4)
    assert ghost.last_seen == (0, 0)


def test_pursue_routes_around_walls():
    ghost = Ghost(Scripted(0.5))
    ghost.last_seen = (0, 4)
    ghost.act((1, 2), GameMap({(3, 4), (2, 4)}), 1)
    assert ghost.cell == (4, 3)


def test_reaching_the_last_sighting_clears_it():
    ghost = Ghost(Scripted(0.5), start=(3, 4))
    ghost.last_seen = (4, 4)
    ghost.act((0, 0), EMPTY, 1)
    assert ghost.cell == (4, 4)
    assert ghost.last_seen is None


def test_chase_then_pursue_then_search():
    rng = Scripted(0.5, 0.5, 0.5, 0.5, 0.5, 0.0)
    ghost = Ghost(rng)
    ghost.act((4, 0), EMPTY, 1)                       # seen: chase
    assert (ghost.cell, ghost.last_seen) == ((4, 3), (4, 0))
    ghost.act((3, 0), EMPTY, 2)                       # hidden: pursue
    assert (ghost.cell, ghost.last_seen) == ((4, 2), (4, 0))
    ghost.act((3, 0), EMPTY, 3)                       # still hidden: pursue
    assert (ghost.cell, ghost.last_seen) == ((4, 1), (4, 0))
    ghost.act((2, 0), EMPTY, 4)                       # arrives: cleared
    assert (ghost.cell, ghost.last_seen) == ((4, 0), None)
    ghost.act((2, 2), EMPTY, 5)                       # search: (3, 0) never visited
    assert ghost.cell == (3, 0)
    assert rng.calls == 5


def test_moving_onto_pacman_clears_the_sighting():
    ghost = Ghost(Scripted(0.5))
    ghost.act((4, 3), EMPTY, 1)
    assert ghost.cell == (4, 3)
    assert ghost.last_seen is None


def test_search_prefers_a_never_visited_neighbour_without_a_tie_draw():
    rng = Scripted(0.5)
    ghost = Ghost(rng)
    ghost.last_visit[(3, 4)] = 1
    ghost.act((0, 0), EMPTY, 2)
    assert ghost.cell == (4, 3)
    assert ghost.last_visit[(4, 3)] == 2
    assert rng.calls == 1


def test_search_prefers_the_least_recently_visited_neighbour():
    ghost = Ghost(Scripted(0.5))
    ghost.last_visit.update({(3, 4): 1, (4, 3): 2})
    ghost.act((0, 0), EMPTY, 3)
    assert ghost.cell == (3, 4)


@pytest.mark.parametrize("draw,expected", [(0.0, (3, 4)), (0.49, (3, 4)), (0.5, (4, 3)), (0.99, (4, 3))])
def test_search_ties_use_one_draw_over_candidates_in_direction_order(draw, expected):
    rng = Scripted(0.5, draw)
    ghost = Ghost(rng)
    ghost.act((0, 0), EMPTY, 1)
    assert ghost.cell == expected
    assert rng.calls == 2


def test_search_never_stands_still_while_a_move_exists():
    ghost = Ghost(Scripted(0.5))
    ghost.last_visit[(4, 3)] = 1
    ghost.act((0, 0), GameMap({(3, 4)}), 2)
    assert ghost.cell == (4, 3)
    assert ghost.last_visit[(4, 3)] == 2


def test_search_skips_walls_when_choosing_candidates():
    ghost = Ghost(Scripted(0.5), start=(2, 2))
    ghost.act((0, 0), GameMap({(1, 2), (2, 3), (3, 2)}), 1)
    assert ghost.cell == (2, 1)


def test_failed_move_draw_keeps_the_ghost_still_and_refreshes_vision():
    rng = Scripted(0.7)
    ghost = Ghost(rng, move_prob=0.5)
    ghost.act((4, 0), EMPTY, 1)
    assert ghost.cell == (4, 4)
    assert ghost.last_seen == (4, 0)
    assert ghost.last_visit == {(4, 4): 0}
    assert rng.calls == 1


def test_move_draw_succeeds_strictly_below_the_probability():
    stays = Ghost(Scripted(0.5), move_prob=0.5)
    stays.act((4, 0), EMPTY, 1)
    assert stays.cell == (4, 4)
    moves = Ghost(Scripted(0.4999), move_prob=0.5)
    moves.act((4, 0), EMPTY, 1)
    assert moves.cell == (4, 3)


def test_move_draw_is_taken_every_step_even_at_probability_one():
    rng = Scripted(0.5, 0.5)
    ghost = Ghost(rng, move_prob=1.0)
    ghost.act((4, 0), EMPTY, 1)
    ghost.act((4, 0), EMPTY, 2)
    assert rng.calls == 2


def test_zero_probability_never_moves():
    ghost = Ghost(Scripted(0.0), move_prob=0.0)
    ghost.act((4, 0), EMPTY, 1)
    assert ghost.cell == (4, 4)


def test_same_seed_gives_the_same_search_path():
    paths = []
    for _ in range(2):
        ghost = Ghost(derive(9, "ghost"))
        path = []
        for t in range(1, 30):
            ghost.act((0, 0), GameMap({(2, 2)}), t)
            path.append(ghost.cell)
            if ghost.cell == (0, 0):  # the environment ends the game here
                break
        paths.append(path)
    assert paths[0] == paths[1]
    assert len(set(paths[0])) > 1


def test_ghost_uses_the_true_map_for_sight_and_movement():
    walls = {(4, 1), (4, 2), (4, 3)}
    ghost = Ghost(Scripted(0.5))
    ghost.act((4, 0), GameMap(walls), 1)
    assert ghost.last_seen is None
    assert ghost.cell == (3, 4)
