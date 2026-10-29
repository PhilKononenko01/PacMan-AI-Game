"""Movement, planning and sight on the fixed 5x5 grid (FR-1, FR-5)."""

import subprocess
import sys

import pytest

from pacman.grid import CORNERS, Direction, bfs, first_step, line_of_sight, move, neighbours


def test_corners_are_the_four_grid_corners_in_row_major_order():
    assert CORNERS == ((0, 0), (0, 4), (4, 0), (4, 4))


def open_except(*walls):
    """A passable predicate for a map whose only walls are the given cells."""
    blocked = frozenset(walls)
    return lambda cell: cell not in blocked


EMPTY = open_except()


@pytest.mark.parametrize("cell,expected", [
    ((0, 0), ((0, 1), (1, 0))),
    ((0, 2), ((0, 3), (1, 2), (0, 1))),
    ((2, 2), ((1, 2), (2, 3), (3, 2), (2, 1))),
    ((4, 4), ((3, 4), (4, 3))),
])
def test_neighbours_are_in_grid_and_in_direction_order(cell, expected):
    assert neighbours(cell) == expected


@pytest.mark.parametrize("cell,direction", [
    ((0, 2), Direction.UP),
    ((2, 4), Direction.RIGHT),
    ((4, 2), Direction.DOWN),
    ((2, 0), Direction.LEFT),
])
def test_move_off_the_grid_stays_in_place(cell, direction):
    assert move(cell, direction, EMPTY) == cell


def test_move_into_a_wall_stays_in_place():
    assert move((2, 2), Direction.RIGHT, open_except((2, 3))) == (2, 2)


@pytest.mark.parametrize("direction,expected", [
    (Direction.UP, (1, 2)),
    (Direction.RIGHT, (2, 3)),
    (Direction.DOWN, (3, 2)),
    (Direction.LEFT, (2, 1)),
])
def test_move_into_an_open_cell_follows_the_delta(direction, expected):
    assert move((2, 2), direction, EMPTY) == expected


def test_bfs_on_an_empty_grid_gives_manhattan_distances():
    distances = bfs((0, 0), EMPTY)
    assert len(distances) == 25
    assert distances[(0, 0)] == 0
    assert distances[(4, 4)] == 8
    assert distances[(1, 3)] == 4


def test_bfs_routes_around_walls_and_excludes_them():
    passable = open_except((0, 2), (1, 2), (2, 2), (3, 2))
    distances = bfs((0, 0), passable)
    assert distances[(0, 4)] == 12
    assert (2, 2) not in distances


def test_bfs_omits_unreachable_cells():
    passable = open_except((0, 1), (1, 0))
    assert (0, 0) not in bfs((2, 2), passable)
    assert bfs((0, 0), passable) == {(0, 0): 0}


@pytest.mark.parametrize("target,expected", [
    ((0, 0), Direction.UP),
    ((0, 4), Direction.UP),
    ((4, 4), Direction.RIGHT),
    ((4, 0), Direction.DOWN),
])
def test_first_step_breaks_shortest_path_ties_in_direction_order(target, expected):
    assert first_step((2, 2), target, EMPTY) == expected


def test_first_step_skips_a_blocked_direction():
    assert first_step((2, 2), (0, 2), open_except((1, 2))) == Direction.RIGHT


def test_first_step_routes_around_a_wall():
    assert first_step((2, 2), (2, 0), open_except((2, 1))) == Direction.UP


def test_first_step_to_an_adjacent_target_is_direct():
    assert first_step((0, 0), (0, 1), EMPTY) == Direction.RIGHT


def test_first_step_with_no_path_is_an_error():
    with pytest.raises(ValueError, match="path"):
        first_step((2, 2), (0, 0), open_except((0, 1), (1, 0)))


@pytest.mark.parametrize("other", [(0, 2), (2, 4), (4, 2), (2, 0)])
def test_line_of_sight_is_clear_in_all_four_directions(other):
    assert line_of_sight((2, 2), other, EMPTY)
    assert line_of_sight(other, (2, 2), EMPTY)


@pytest.mark.parametrize("other,wall", [
    ((0, 2), (1, 2)),
    ((2, 4), (2, 3)),
    ((4, 2), (3, 2)),
    ((2, 0), (2, 1)),
])
def test_line_of_sight_is_blocked_by_a_wall_between(other, wall):
    assert not line_of_sight((2, 2), other, open_except(wall))
    assert not line_of_sight(other, (2, 2), open_except(wall))


def test_line_of_sight_checks_only_the_cells_strictly_between():
    passable = open_except((2, 0), (2, 3))
    assert line_of_sight((2, 0), (2, 2), passable)


def test_adjacent_cells_always_see_each_other():
    assert line_of_sight((2, 2), (2, 3), lambda cell: False)
    assert line_of_sight((3, 2), (2, 2), lambda cell: False)


@pytest.mark.parametrize("other", [(1, 3), (3, 1), (0, 4), (4, 1)])
def test_line_of_sight_needs_a_shared_row_or_column(other):
    assert not line_of_sight((2, 2), other, EMPTY)


def test_grid_imports_no_other_game_module():
    code = (
        "import sys, pacman.grid;"
        " print(sorted(m for m in sys.modules if m.startswith('pacman')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "['pacman', 'pacman.grid']"


@pytest.mark.parametrize("action", [-1, 4, 7])
def test_move_rejects_an_action_outside_the_four_directions(action):
    with pytest.raises(ValueError):
        move((2, 2), action, lambda cell: True)


def test_first_step_with_source_equal_to_target_is_an_error():
    with pytest.raises(ValueError, match="target"):
        first_step((1, 1), (1, 1), lambda cell: True)


def test_the_neighbour_table_covers_every_cell_in_direction_order():
    from pacman.grid import NEIGHBOURS, in_grid

    assert len(NEIGHBOURS) == 25
    for (row, col), cells in NEIGHBOURS.items():
        expected = tuple(
            cell for cell in ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
            if in_grid(cell)
        )
        assert cells == expected
