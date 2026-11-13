"""Versioned numeric artifacts for the two-pellet hybrid experiment."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np

from pacman.hybrid.learning import FrozenTable
from pacman.hybrid.models import GHOST_FEATURES, PAC_FEATURES, RULES_VERSION, Rules

NAMESPACE = "pacman.hybrid.policy"
VERSION = 1
_FEATURES = {"pacman": PAC_FEATURES, "ghost": GHOST_FEATURES}
_FIELDS = frozenset(("namespace", "version", "rules_version", "features_version", "role",
                     "rules", "metadata", "states", "q"))
_GHOST_BOUNDS = ((0, 24), (0, 24), (0, 4), (1, 15), (0, 3), (0, 2), (0, 3), (0, 1))
_PAC_BOUNDS = ((0, 3), (1, 15), (0, 80), (1, 4), (0, 1), (1, 2), (0, 3), (0, 3))


@dataclass(frozen=True, slots=True)
class Policy:
    role: str
    table: FrozenTable
    rules: Rules
    metadata: Mapping
    digest: str


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _parse(value):
    return json.loads(value, parse_constant=_reject_constant, object_pairs_hook=_pairs)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate(role, states, q):
    if role not in _FEATURES:
        raise ValueError("role must be pacman or ghost")
    if states.dtype.kind != "i" or states.dtype.itemsize != 8 or states.ndim != 2 or states.shape[1] != 8:
        raise ValueError("states must be an int64 matrix with eight columns")
    if q.dtype.kind != "f" or q.dtype.itemsize != 8 or q.shape != (len(states), 4):
        raise ValueError("q must be a float64 matrix with four columns")
    if not np.isfinite(q).all():
        raise ValueError("q values must be finite")
    if (states < 0).any():
        raise ValueError("state values must be nonnegative")
    bounds = _PAC_BOUNDS if role == "pacman" else _GHOST_BOUNDS
    for column, (low, high) in enumerate(bounds):
        if ((states[:, column] < low) | (states[:, column] > high)).any():
            raise ValueError(f"state column {column} is outside feature bounds")
    if len(set(map(tuple, states.tolist()))) != len(states):
        raise ValueError("state rows must be unique")


def save_policy(path: Path, role: str, table: FrozenTable, rules: Rules, metadata: dict) -> str:
    """Atomically publish a new artifact without replacing an existing policy."""
    path = Path(path)
    if role not in _FEATURES:
        raise ValueError("role must be pacman or ghost")
    if not isinstance(table, FrozenTable) or not isinstance(rules, Rules):
        raise ValueError("a frozen table and Rules are required")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    metadata_json = _json(metadata)
    keys = sorted(table.rows)
    if any(len(key) != 8 or any(type(item) is not int for item in key) for key in keys):
        raise ValueError("state keys must contain eight integers")
    states = np.asarray(keys, dtype=np.int64).reshape((-1, 8))
    q = np.asarray([table.rows[key] for key in keys], dtype=np.float64).reshape((-1, 4))
    _validate(role, states, q)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, namespace=np.array(NAMESPACE), version=np.array(VERSION),
                                rules_version=np.array(RULES_VERSION), features_version=np.array(_FEATURES[role]),
                                role=np.array(role), rules=np.array(_json(asdict(rules))),
                                metadata=np.array(metadata_json), states=states, q=q)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.link(temporary, path)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_policy(path: Path, expected_role: str | None = None, rules: Rules | None = None) -> Policy:
    """Reject legacy or incompatible policies before exposing immutable rows."""
    payload = Path(path).read_bytes()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if len(archive.files) != len(_FIELDS) or set(archive.files) != _FIELDS:
                raise ValueError("not a hybrid policy artifact")
            def scalar(name):
                value = archive[name]
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError(f"{name} must be a string scalar")
                return value.item()
            version = archive["version"]
            if version.shape != () or version.dtype.kind not in "iu" or version.item() != VERSION:
                raise ValueError("unsupported hybrid artifact version")
            if scalar("namespace") != NAMESPACE or scalar("rules_version") != RULES_VERSION:
                raise ValueError("incompatible artifact namespace or rules version")
            role = scalar("role")
            if role not in _FEATURES or (expected_role is not None and role != expected_role):
                raise ValueError("policy role mismatch")
            if scalar("features_version") != _FEATURES[role]:
                raise ValueError("incompatible feature version")
            saved_rules = _parse(scalar("rules"))
            if not isinstance(saved_rules, dict) or set(saved_rules) != set(Rules.__dataclass_fields__):
                raise ValueError("invalid policy rules")
            parsed_rules = Rules(**saved_rules)
            if rules is not None and parsed_rules != rules:
                raise ValueError("policy rules mismatch")
            metadata = _parse(scalar("metadata"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            _json(metadata)
            states, q = archive["states"], archive["q"]
            _validate(role, states, q)
            table = FrozenTable({tuple(map(int, state)): tuple(map(float, row)) for state, row in zip(states, q)})
    except (OSError, TypeError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError("invalid hybrid policy artifact") from error
    return Policy(role, table, parsed_rules, _freeze(metadata), hashlib.sha256(payload).hexdigest())
