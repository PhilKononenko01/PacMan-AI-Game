"""Plain-text and pixel-art frames for hand-built states (FR-13)."""

import re

import pytest

from pacman.env import KnownCell, Observation, Outcome
from pacman.episode import Frame
from pacman.grid import COLS
from pacman.render import pixel_frame, plain_frame, render_frame, use_plain

ESCAPE = re.compile(r"\x1b\[(?:38;5;\d{1,3}|48;5;\d{1,3}|0)m")


def memory(**cells):
    """Row-major memory with the named cells set, e.g. memory(open=[...], wall=[...])."""
    known = [KnownCell.UNKNOWN] * 25
    for row, col in cells.get("open", ()):
        known[row * COLS + col] = KnownCell.OPEN
    for row, col in cells.get("wall", ()):
        known[row * COLS + col] = KnownCell.WALL
    return tuple(known)


def frame(pacman=(0, 1), ghost=(4, 3), visited=((0, 0),), known=None, t=3, outcome=None,
          ghost_sees=False, max_steps=100):
    if known is None:
        known = memory(open=[(0, 0), (1, 1)], wall=[(0, 2)])
    observation = Observation(pacman, ghost, frozenset(visited), known)
    return Frame(observation, t, outcome, ghost_sees, max_steps)


def grid_lines(text):
    return text.split("\n")[:5]


def test_plain_grid_draws_memory_actors_and_corners():
    assert grid_lines(plain_frame(frame())) == [
        "*P#?o",
        "?.???",
        "?????",
        "?????",
        "o??Go",
    ]


def test_plain_status_line_shows_move_corners_and_sight():
    lines = plain_frame(frame(t=3, ghost_sees=False)).split("\n")
    assert lines[5] == "Move 3/100  Corners 1/4  Ghost sees Pac-Man: no"
    assert len(lines) == 6
    lines = plain_frame(frame(t=0, ghost_sees=True, visited=[(0, 0), (0, 4)])).split("\n")
    assert lines[5] == "Move 0/100  Corners 2/4  Ghost sees Pac-Man: yes"


@pytest.mark.parametrize("outcome,label", [
    (Outcome.WIN, "Outcome: win"), (Outcome.LOSS, "Outcome: loss"), (Outcome.TIE, "Outcome: tie"),
])
def test_the_last_frame_adds_the_ruling(outcome, label):
    lines = plain_frame(frame(outcome=outcome)).split("\n")
    assert lines[6] == label
    assert len(lines) == 7


def test_the_ghost_is_drawn_over_pacman_on_a_collision():
    text = plain_frame(frame(pacman=(2, 2), ghost=(2, 2)))
    assert grid_lines(text)[2] == "??G??"


def test_actors_are_drawn_over_corners_and_corners_over_memory():
    text = plain_frame(frame(pacman=(0, 4), ghost=(4, 0), visited=[(0, 0), (0, 4)]))
    lines = grid_lines(text)
    assert lines[0][4] == "P"
    assert lines[4][0] == "G"
    assert lines[0][0] == "*"
    assert lines[4][4] == "o"
    known = memory(open=[(0, 4), (4, 4)])
    lines = grid_lines(plain_frame(frame(pacman=(2, 2), ghost=(2, 3), known=known)))
    assert lines[0][4] == "o"
    assert lines[4][4] == "o"


def test_unknown_walls_look_unknown_and_known_walls_look_like_walls():
    lines = grid_lines(plain_frame(frame(known=memory(wall=[(1, 1)]))))
    assert lines[1][1] == "#"
    assert lines[1][2] == "?"


@pytest.mark.parametrize("plain,isatty,environ,expected", [
    (True, True, {}, True),
    (False, False, {}, True),
    (False, True, {"NO_COLOR": "1"}, True),
    (False, True, {"NO_COLOR": ""}, False),
    (False, True, {"TERM": "dumb"}, True),
    (False, True, {"TERM": "xterm-256color"}, False),
    (False, True, {}, False),
])
def test_plain_fallback_rules(plain, isatty, environ, expected):
    assert use_plain(plain, isatty, environ) is expected


def test_pixel_frame_is_two_half_block_lines_per_row_in_256_colours():
    text = pixel_frame(frame())
    lines = text.split("\n")
    grid = lines[:10]
    for line in grid:
        assert line.endswith("\x1b[0m")
        assert ESCAPE.sub("", line) == "▀" * 20
        assert "\x1b[38;5;" in line and "\x1b[48;5;" in line
    assert lines[10:] == plain_frame(frame()).split("\n")[5:]


def test_pixel_frame_contains_no_other_control_characters():
    text = pixel_frame(frame(outcome=Outcome.WIN))
    stripped = ESCAPE.sub("", text)
    assert "\x1b" not in stripped
    assert all(char.isprintable() or char == "\n" for char in stripped)


def test_pixel_sprites_distinguish_every_cell_kind():
    base = frame(pacman=(2, 2), ghost=(2, 3), visited=[(0, 0)], known=memory(open=[(1, 2)], wall=[(3, 2)]))
    lines = pixel_frame(base).split("\n")
    # A cell is 4 characters wide with a foreground and a background escape each.
    kinds = {}
    for name, (row, col) in {
        "pacman": (2, 2), "ghost": (2, 3), "open": (1, 2), "wall": (3, 2),
        "unknown": (0, 2), "visited": (0, 0), "unvisited": (0, 4),
    }.items():
        top = ESCAPE.findall(lines[2 * row])
        bottom = ESCAPE.findall(lines[2 * row + 1])
        kinds[name] = (tuple(top[col * 8:(col + 1) * 8]), tuple(bottom[col * 8:(col + 1) * 8]))
    assert len(set(kinds.values())) == 7


def test_render_frame_dispatches_on_the_plain_flag():
    assert render_frame(frame(), plain=True) == plain_frame(frame())
    assert render_frame(frame(), plain=False) == pixel_frame(frame())


def test_rendering_writes_nothing_to_standard_output(capsys):
    plain_frame(frame())
    pixel_frame(frame())
    assert capsys.readouterr() == ("", "")
