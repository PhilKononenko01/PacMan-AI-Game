import random

from pacman.grid import DELTAS, Direction, first_step, in_grid, neighbours
from pacman.hybrid.learning import FrozenTable, QTable, choose
from pacman.hybrid.models import GHOST_START, GhostObservation, Rules
from pacman.rng import pick


class GhostController:
    def __init__(self, rules: Rules, table: QTable | FrozenTable | None = None,
                 *, training=False, alpha=0.1, gamma=0.97):
        if training and not isinstance(table, QTable):
            raise ValueError("ghost training requires a mutable table")
        self.rules, self.table, self.training = rules, table, training
        self.alpha, self.gamma, self.epsilon = alpha, gamma, 0.0
        self.search_decisions = 0
        self.unseen = 0

    def reset(self, game_map, rng: random.Random):
        self.game_map, self.rng = game_map, rng
        self.last_seen = self.last_visible = self.heading = self.target = None
        self.age = self.choices = self.moves_left = 0
        self.tried = set()
        self.pending = None
        self.ended = False
        self.mode = "wander"

    def _clear_memory(self):
        self.last_seen = self.last_visible = self.heading = self.target = None
        self.age = self.choices = self.moves_left = 0
        self.tried = set()

    def _wander(self, cell):
        legal = [d for d, delta in enumerate(DELTAS)
                 if in_grid((cell[0] + delta[0], cell[1] + delta[1]))
                 and self.game_map.is_open((cell[0] + delta[0], cell[1] + delta[1]))]
        self.mode = "wander"
        return Direction(legal[pick(self.rng, len(legal))])

    def candidates(self, cell):
        targets = {}
        for direction, (dr, dc) in enumerate(DELTAS):
            target = cell
            for _ in range(self.rules.search_distance):
                next_cell = (target[0] + dr, target[1] + dc)
                if not in_grid(next_cell) or not self.game_map.is_open(next_cell):
                    break
                target = next_cell
            if target != cell and target not in self.tried:
                targets[direction] = target
        return targets

    def _state(self, obs, legal):
        heading = self.heading if self.heading is not None else 4
        forward = int(self.heading in legal)
        return (obs.cell[0] * 5 + obs.cell[1], self.last_seen[0] * 5 + self.last_seen[1],
                heading, sum(1 << d for d in legal), min(self.age // 3, 3),
                self.choices, min(obs.turns_left // 25, 3), forward)

    def _close(self, bootstrap=0.0):
        if self.pending is not None:
            state, action, elapsed, total = self.pending
            self.table.update(state, action, total + self.gamma ** elapsed * bootstrap, self.alpha)
            self.pending = None

    def act(self, obs: GhostObservation) -> Direction:
        if self.ended:
            raise RuntimeError("reset ghost before starting another game")
        cell, visible = obs.cell, obs.visible_pacman
        if visible is not None:
            heading = None
            if self.last_visible is not None:
                delta = (visible[0] - self.last_visible[0], visible[1] - self.last_visible[1])
                if delta in DELTAS:
                    heading = DELTAS.index(delta)
            self.last_seen, self.last_visible, self.heading = visible, visible, heading
            self.age = self.choices = self.moves_left = 0
            self.target, self.tried = None, set()
            self.mode = "chase"
            return first_step(cell, visible, self.game_map.is_open)
        self.last_visible = None
        if self.last_seen is None:
            return self._wander(cell)
        self.age += 1
        if self.age >= self.rules.memory_turns:
            self._clear_memory()
            return self._wander(cell)
        if self.target is None and self.choices == 0 and cell != self.last_seen:
            self.mode = "pursue"
            return first_step(cell, self.last_seen, self.game_map.is_open)
        if self.target is not None and cell != self.target and self.moves_left > 0:
            self.moves_left -= 1
            self.mode = "search"
            return first_step(cell, self.target, self.game_map.is_open)
        if self.choices >= self.rules.search_choices:
            self._clear_memory()
            return self._wander(cell)
        targets = self.candidates(cell)
        if not targets:
            self._clear_memory()
            return self._wander(cell)
        legal = tuple(targets)
        state = self._state(obs, legal)
        self.search_decisions += 1
        if self.table is None:
            action = self.heading if self.heading in legal else legal[pick(self.rng, len(legal))]
        else:
            self.unseen += int(state not in self.table.rows)
            if self.training:
                self._close(max(self.table.values(state)[a] for a in legal))
            values = self.table.values(state)
            if not self.epsilon and len({values[a] for a in legal}) == 1:
                action = self.heading if self.heading in legal else legal[pick(self.rng, len(legal))]
            else:
                action = choose(self.table, state, legal, self.rng,
                                self.epsilon if self.training else 0.0)
            if self.training:
                self.pending = (state, action, 0, 0.0)
        self.target = targets[action]
        self.tried.add(self.target)
        self.choices += 1
        self.moves_left = self.rules.search_moves - 1
        self.mode = "search"
        return first_step(cell, self.target, self.game_map.is_open)

    def reward(self, value: float, terminal: bool):
        if self.ended:
            raise RuntimeError("ghost received a second terminal notification")
        if self.training and self.pending is not None:
            state, action, elapsed, total = self.pending
            self.pending = (state, action, elapsed + 1, total + self.gamma ** elapsed * value)
            if terminal:
                self._close()
        if terminal:
            self.ended = True
