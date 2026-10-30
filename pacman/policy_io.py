"""Versioned policy files: the Q-table and the metadata needed to replay a run.

Imports only the configuration and NumPy, so training, evaluation and the CLI can
all use it without pulling in each other.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy

from pacman.config import Config, _reject_json_constant, _unique_json_object

FORMAT_VERSION = 1
# The state index has 122,880 rows; features.py must agree with this shape.
TABLE_SHAPE = (122_880, 4)

_META_KEYS = ("format_version", "config", "training_seeds", "learning_curve", "validation")
_VALIDATION_KEYS = ("seeds", "games", "wins", "losses", "ties", "mean_steps_to_win")


def _count(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed_range(name: str, value: Any) -> tuple[int, int]:
    try:
        first, last = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a [first, last] pair") from None
    if _count(name, first) > _count(name, last):
        raise ValueError(f"{name} must run from the first seed to the last")
    return first, last


def _check_table(q: Any) -> numpy.ndarray:
    if not isinstance(q, numpy.ndarray) or q.dtype != numpy.float64 or q.shape != TABLE_SHAPE:
        raise ValueError(f"q must be a float64 array of shape {TABLE_SHAPE}")
    return q


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """The final greedy run on the validation split, as stored in the policy file."""

    seeds: tuple[int, int]
    games: int
    wins: int
    losses: int
    ties: int
    mean_steps_to_win: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", _seed_range("validation seeds", self.seeds))
        for name in ("games", "wins", "losses", "ties"):
            _count(name, getattr(self, name))
        if self.wins + self.losses + self.ties != self.games:
            raise ValueError("wins, losses and ties must sum to games")
        mean = self.mean_steps_to_win
        if (self.wins == 0) != (mean is None):
            raise ValueError("mean_steps_to_win must be null exactly when there are no wins")
        if mean is not None:
            if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean):
                raise ValueError("mean_steps_to_win must be a finite number")
            object.__setattr__(self, "mean_steps_to_win", float(mean))


@dataclass(frozen=True, slots=True, eq=False)  # a NumPy field makes value equality ambiguous
class Policy:
    q: numpy.ndarray
    config: Config
    training_seeds: tuple[int, int]
    learning_curve: tuple[tuple[int, float], ...]
    validation: ValidationSummary

    def __post_init__(self) -> None:
        _check_table(self.q)
        if not isinstance(self.config, Config):
            raise ValueError("config must be a Config")
        object.__setattr__(self, "training_seeds", _seed_range("training_seeds", self.training_seeds))
        try:
            curve = tuple((_count("learning_curve episodes", e), float(w)) for e, w in self.learning_curve)
        except (TypeError, ValueError) as error:
            raise ValueError(f"learning_curve must be [episodes, win_rate] pairs: {error}") from None
        object.__setattr__(self, "learning_curve", curve)
        if not isinstance(self.validation, ValidationSummary):
            raise ValueError("validation must be a ValidationSummary")


def save_policy(path: str | Path, policy: Policy) -> None:
    """Write the two-array .npz to exactly the given path."""
    validation = policy.validation
    meta = {
        "format_version": FORMAT_VERSION,
        "config": json.loads(policy.config.to_json()),
        "training_seeds": list(policy.training_seeds),
        "learning_curve": [list(point) for point in policy.learning_curve],
        "validation": {
            "seeds": list(validation.seeds),
            "games": validation.games,
            "wins": validation.wins,
            "losses": validation.losses,
            "ties": validation.ties,
            "mean_steps_to_win": validation.mean_steps_to_win,
        },
    }
    text = json.dumps(meta, sort_keys=True, allow_nan=False)
    meta_array = numpy.frombuffer(text.encode("utf-8"), dtype=numpy.uint8)
    # Write beside the target and rename, so an interrupted save never leaves a truncated policy.
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    try:
        with open(partial, "wb") as file:
            numpy.savez(file, q=policy.q, meta=meta_array)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _require(mapping: Any, key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"policy {where} lacks '{key}'")
    return mapping[key]


def load_policy(path: str | Path) -> Policy:
    """Read a policy file, refusing unknown versions, bad tables and missing keys."""
    try:
        data = numpy.load(path, allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise ValueError(f"{path} is not an .npz policy archive: {error}") from None
    if not isinstance(data, numpy.lib.npyio.NpzFile):
        raise ValueError(f"{path} is not an .npz policy archive")
    with data:
        for key in ("q", "meta"):
            if key not in data.files:
                raise ValueError(f"policy file lacks the '{key}' array")
        q = data["q"]
        raw = data["meta"]
    try:
        meta = json.loads(
            bytes(raw).decode("utf-8"),
            parse_constant=_reject_json_constant,  # the same strictness as Config.from_json
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"policy metadata is not UTF-8 JSON: {error}") from None
    version = _require(meta, "format_version", "metadata")
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError(f"unknown policy format_version {version!r}; this build reads {FORMAT_VERSION}")
    for key in _META_KEYS:
        _require(meta, key, "metadata")
    validation = meta["validation"]
    for key in _VALIDATION_KEYS:
        _require(validation, key, "validation")
    return Policy(
        q=_check_table(q),
        config=Config.from_json(json.dumps(meta["config"])),
        training_seeds=tuple(meta["training_seeds"]),
        learning_curve=tuple(tuple(point) for point in meta["learning_curve"]),
        validation=ValidationSummary(
            seeds=tuple(validation["seeds"]),
            games=validation["games"],
            wins=validation["wins"],
            losses=validation["losses"],
            ties=validation["ties"],
            mean_steps_to_win=validation["mean_steps_to_win"],
        ),
    )
