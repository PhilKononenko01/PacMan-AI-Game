"""Watch-mode frames as strings: terminal pixel art and the plain-text fallback.

The renderer draws walls and unknown cells from Pac-Man's memory (D-5), and the
actors and corners where they are. It never writes output or sleeps; the CLI does.
"""

from collections.abc import Mapping

from pacman.env import KnownCell, Observation
from pacman.episode import Frame
from pacman.grid import COLS, CORNERS, ROWS, Cell

HALF_BLOCK = "▀"  # upper half block: foreground is the upper pixel, background the lower

MEMORY_KIND = {KnownCell.WALL: "wall", KnownCell.OPEN: "open", KnownCell.UNKNOWN: "unknown"}

PLAIN = {
    "pacman": "P", "ghost": "G", "wall": "#", "unknown": "?",
    "unvisited": "o", "visited": "*", "open": ".",
}

# 256-colour indices.
FLOOR, SHADE, SHADE_DARK = 233, 238, 236
WALL, WALL_EDGE = 27, 33
CORNER, CORNER_DONE = 46, 22
PACMAN, GHOST, EYE = 226, 196, 231

Sprite = tuple[tuple[int | None, ...], ...]

SPRITES: dict[str, Sprite] = {
    "open": ((FLOOR,) * 4,) * 4,
    "unknown": (
        (SHADE, SHADE_DARK, SHADE, SHADE_DARK),
        (SHADE_DARK, SHADE, SHADE_DARK, SHADE),
        (SHADE, SHADE_DARK, SHADE, SHADE_DARK),
        (SHADE_DARK, SHADE, SHADE_DARK, SHADE),
    ),
    "wall": (
        (WALL_EDGE, WALL_EDGE, WALL_EDGE, WALL_EDGE),
        (WALL_EDGE, WALL, WALL, WALL_EDGE),
        (WALL_EDGE, WALL, WALL, WALL_EDGE),
        (WALL_EDGE, WALL_EDGE, WALL_EDGE, WALL_EDGE),
    ),
    "unvisited": (
        (FLOOR, FLOOR, FLOOR, FLOOR),
        (FLOOR, CORNER, CORNER, FLOOR),
        (FLOOR, CORNER, CORNER, FLOOR),
        (FLOOR, FLOOR, FLOOR, FLOOR),
    ),
    "visited": (
        (FLOOR, FLOOR, FLOOR, FLOOR),
        (FLOOR, CORNER_DONE, CORNER_DONE, FLOOR),
        (FLOOR, CORNER_DONE, CORNER_DONE, FLOOR),
        (FLOOR, FLOOR, FLOOR, FLOOR),
    ),
    # Actors are drawn over their cell's sprite; None shows the pixel beneath.
    "pacman": (
        (None, PACMAN, PACMAN, None),
        (PACMAN, PACMAN, PACMAN, PACMAN),
        (PACMAN, PACMAN, PACMAN, PACMAN),
        (None, PACMAN, PACMAN, None),
    ),
    "ghost": (
        (None, GHOST, GHOST, None),
        (GHOST, EYE, GHOST, EYE),
        (GHOST, GHOST, GHOST, GHOST),
        (GHOST, None, GHOST, None),
    ),
}


def use_plain(plain: bool, isatty: bool, environ: Mapping[str, str]) -> bool:
    """The plain-rendering fallback rule: --plain, no terminal, NO_COLOR set, or TERM=dumb."""
    return plain or not isatty or bool(environ.get("NO_COLOR")) or environ.get("TERM") == "dumb"


def render_frame(frame: Frame, plain: bool = False) -> str:
    return plain_frame(frame) if plain else pixel_frame(frame)


def plain_frame(frame: Frame) -> str:
    """One character per cell, then the status lines."""
    observation = frame.observation
    rows = [
        "".join(PLAIN[_actor(observation, (row, col)) or _base(observation, (row, col))]
                for col in range(COLS))
        for row in range(ROWS)
    ]
    return "\n".join(rows + _status(frame))


def pixel_frame(frame: Frame) -> str:
    """Two half-block lines per grid row, four characters per cell, then the status lines."""
    observation = frame.observation
    lines = []
    for row in range(ROWS):
        sprites = [_pixels(observation, (row, col)) for col in range(COLS)]
        for upper in (0, 2):
            chars = [
                f"\x1b[38;5;{sprite[upper][col]}m\x1b[48;5;{sprite[upper + 1][col]}m{HALF_BLOCK}"
                for sprite in sprites
                for col in range(4)
            ]
            lines.append("".join(chars) + "\x1b[0m")
    return "\n".join(lines + _status(frame))


def _base(observation: Observation, cell: Cell) -> str:
    if cell in CORNERS:
        return "visited" if cell in observation.visited else "unvisited"
    row, col = cell
    return MEMORY_KIND[observation.known[row * COLS + col]]


def _actor(observation: Observation, cell: Cell) -> str | None:
    if cell == observation.ghost:
        return "ghost"
    if cell == observation.pacman:
        return "pacman"
    return None


def _pixels(observation: Observation, cell: Cell) -> Sprite:
    base = SPRITES[_base(observation, cell)]
    actor = _actor(observation, cell)
    if actor is None:
        return base
    overlay = SPRITES[actor]
    return tuple(
        tuple(top if top is not None else under for top, under in zip(overlay[row], base[row]))
        for row in range(4)
    )


def _status(frame: Frame) -> list[str]:
    corners = sum(corner in frame.observation.visited for corner in CORNERS)
    sees = "yes" if frame.ghost_sees else "no"
    lines = [f"Move {frame.t}/{frame.max_steps}  Corners {corners}/4  Ghost sees Pac-Man: {sees}"]
    if frame.outcome is not None:
        lines.append(f"Outcome: {frame.outcome.value}")
    return lines
