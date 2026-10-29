"""Immutable game and learner settings, with validated JSON serialisation."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from typing import Any


def _integer(name: str, value: Any, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be at least {minimum}{upper}")


def _number(
    name: str,
    value: Any,
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    minimum_exclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        if value < minimum or (minimum_exclusive and value == minimum):
            relation = "greater than" if minimum_exclusive else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


@dataclass(frozen=True, slots=True)
class GameKnobs:
    ghost_move_prob: float = 1.0
    wall_density: float = 0.20
    max_walls: int = 6
    min_loops: int = 4
    max_steps: int = 100

    def __post_init__(self) -> None:
        _number("ghost_move_prob", self.ghost_move_prob, 0, 1)
        _number("wall_density", self.wall_density, 0, 1)
        _integer("max_walls", self.max_walls, 0)
        _integer("min_loops", self.min_loops, 0)
        _integer("max_steps", self.max_steps, 1)


@dataclass(frozen=True, slots=True)
class LearnerSettings:
    reward_win: float = 100.0
    reward_loss: float = -100.0
    reward_corner: float = 1.0
    reward_step: float = -0.05
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5
    episodes: int = 50_000
    tie_update: str = "bootstrap"

    def __post_init__(self) -> None:
        for name in ("reward_win", "reward_loss", "reward_corner", "reward_step"):
            _number(name, getattr(self, name))
        _number("learning_rate", self.learning_rate, 0, 1, minimum_exclusive=True)
        _number("discount", self.discount, 0, 1)
        _number("epsilon_start", self.epsilon_start, 0, 1)
        _number("epsilon_end", self.epsilon_end, 0, self.epsilon_start)
        _number(
            "epsilon_decay_fraction", self.epsilon_decay_fraction,
            0, 1, minimum_exclusive=True,
        )
        _integer("episodes", self.episodes, 1, 1_000_000)
        if not isinstance(self.tie_update, str) or self.tie_update not in (
            "bootstrap", "terminal",
        ):
            raise ValueError("tie_update must be 'bootstrap' or 'terminal'")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _section(data: Any, model: type, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = data.keys() - {item.name for item in fields(model)}
    if unknown:
        raise ValueError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
    return data


@dataclass(frozen=True, slots=True)
class Config:
    game: GameKnobs = field(default_factory=GameKnobs)
    learner: LearnerSettings = field(default_factory=LearnerSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.game, GameKnobs):
            raise ValueError("game must be GameKnobs")
        if not isinstance(self.learner, LearnerSettings):
            raise ValueError("learner must be LearnerSettings")

    def to_json(self) -> str:
        """Return a stable JSON representation of every setting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Config":
        """Read settings; omitted keys use defaults, unknown keys are errors."""
        data = json.loads(text, parse_constant=_reject_json_constant)
        data = _section(data, cls, "configuration")
        game = _section(data.get("game", {}), GameKnobs, "game")
        learner = _section(data.get("learner", {}), LearnerSettings, "learner")
        return cls(game=GameKnobs(**game), learner=LearnerSettings(**learner))
