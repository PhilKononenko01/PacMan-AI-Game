from dataclasses import dataclass
import hashlib
import random
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from pacman.env import KnownCell
from pacman.grid import Direction, DELTAS, bfs, first_step, in_grid, line_of_sight
from pacman.hybrid.models import PacObservation, TurnResult
from pacman.rng import pick

State = tuple[int, ...]


@dataclass(frozen=True)
class FrozenTable:
    rows: Mapping[State, tuple[float, ...]]

    def __post_init__(self):
        copied = {tuple(key): tuple(float(x) for x in row) for key, row in self.rows.items()}
        object.__setattr__(self, "rows", MappingProxyType(copied))

    def values(self, state: State) -> tuple[float, ...]:
        return self.rows.get(state, (0.0, 0.0, 0.0, 0.0))

    def digest(self) -> str:
        digest = hashlib.sha256()
        for key, values in sorted(self.rows.items()):
            digest.update(np.asarray(key, dtype="<i8").tobytes())
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        return digest.hexdigest()


class QTable:
    def __init__(self, frozen: FrozenTable | None = None):
        self.rows = {} if frozen is None else {key: list(row) for key, row in frozen.rows.items()}
        self.updates = 0

    def values(self, state: State):
        return self.rows.get(state, (0.0, 0.0, 0.0, 0.0))

    def update(self, state: State, action: int, target: float, alpha: float):
        values = self.rows.setdefault(state, [0.0] * 4)
        values[action] += alpha * (target - values[action])
        self.updates += 1

    def freeze(self) -> FrozenTable:
        return FrozenTable(self.rows)


def choose(table, state: State, legal: tuple[int, ...], rng: random.Random, epsilon: float) -> int:
    if not legal:
        raise ValueError("a decision requires at least one legal action")
    if epsilon and rng.random() < epsilon:
        return legal[pick(rng, len(legal))]
    values = table.values(state)
    return max(legal, key=lambda action: values[action])


def pac_state(obs: PacObservation) -> tuple[State, tuple[int, ...]]:
    cell, known = obs.pacman, obs.known
    passable = lambda p: known[p[0] * 5 + p[1]] != KnownCell.WALL
    distances = bfs(cell, passable)
    target = min(sorted(obs.pellets), key=lambda p: distances[p])
    direction = int(first_step(cell, target, passable))
    legal = tuple(int(d) for d, delta in enumerate(DELTAS)
                  if in_grid((cell[0] + delta[0], cell[1] + delta[1]))
                  and passable((cell[0] + delta[0], cell[1] + delta[1])))
    mask = sum(1 << d for d in legal)
    dr, dc = obs.ghost[0] - cell[0], obs.ghost[1] - cell[1]
    offset = (dr + 4) * 9 + dc + 4
    ghost_distance = min(distances[obs.ghost], 4)
    seen = int(line_of_sight(cell, obs.ghost, passable))
    target_distance = min((distances[target] - 1) // 2, 3)
    clock = min(obs.turns_left // 25, 3)
    return (direction, mask, offset, ghost_distance, seen, len(obs.pellets), target_distance, clock), legal


class PacActor:
    def __init__(self, table: QTable | FrozenTable, *, training=False, alpha=0.1, gamma=1.0):
        if training and not isinstance(table, QTable):
            raise ValueError("training requires a mutable table")
        self.table, self.training = table, training
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = 0.0
        self.pending = None
        self.decisions = self.unseen = 0

    def reset(self, rng):
        self.rng, self.pending = rng, None

    def act(self, obs: PacObservation) -> Direction:
        state, legal = pac_state(obs)
        self.decisions += 1
        self.unseen += int(state not in self.table.rows)
        action = choose(self.table, state, legal, self.rng, self.epsilon if self.training else 0.0)
        self.pending = state, action
        return Direction(action)

    def observe(self, result: TurnResult):
        if self.training:
            state, action = self.pending
            target = result.pac_reward
            if result.outcome is None:
                next_state, legal = pac_state(result.observation)
                target += self.gamma * max(self.table.values(next_state)[a] for a in legal)
            self.table.update(state, action, target, self.alpha)
        self.pending = None


class BaselinePac:
    def __init__(self, name="pathfinding"):
        if name not in ("pathfinding", "random"):
            raise ValueError("unknown Pac-Man baseline")
        self.name = name

    def reset(self, rng):
        self.rng = rng

    def act(self, obs):
        state, legal = pac_state(obs)
        return Direction(legal[pick(self.rng, len(legal))] if self.name == "random" else state[0])

    def observe(self, result):
        pass
