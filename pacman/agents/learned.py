"""Tabular Q-learning over the ADR-2 features (§10.5)."""

import random

import numpy

from pacman.agents.features import state_index
from pacman.config import LearnerSettings
from pacman.env import Observation, Outcome
from pacman.grid import Direction
from pacman.policy_io import TABLE_SHAPE
from pacman.rng import pick


class LearnedAgent:
    """Greedy outside training; epsilon-greedy with Q updates in training."""

    name = "learned"

    def __init__(
        self, settings: LearnerSettings, q: numpy.ndarray | None = None, training: bool = False,
    ) -> None:
        if q is None:
            q = numpy.zeros(TABLE_SHAPE, dtype=numpy.float64)
        elif not isinstance(q, numpy.ndarray) or q.dtype != numpy.float64 or q.shape != TABLE_SHAPE:
            raise ValueError(f"q must be a float64 array of shape {TABLE_SHAPE}")
        self.settings = settings
        self.q = q
        self.training = training
        self.epsilon = 0.0  # the training loop sets it before each episode (§11.1)
        self.rng: random.Random | None = None
        self._state: int | None = None      # index of the observation the next act sees
        self._action: Direction | None = None

    def reset(self, rng: random.Random) -> None:
        """New game: fresh generator and empty transition cache; the table is kept."""
        self.rng = rng
        self._state = None
        self._action = None

    def act(self, observation: Observation) -> Direction:
        state = self._state if self._state is not None else state_index(observation)
        if self.training and self.rng.random() < self.epsilon:
            action = Direction(pick(self.rng, 4))
        else:
            action = Direction(int(numpy.argmax(self.q[state])))  # first maximum: direction order
        self._state, self._action = state, action
        return action

    def observe(self, reward: float, observation: Observation, outcome: Outcome | None) -> None:
        if not self.training:
            return
        if self._action is None:
            raise RuntimeError("observe without a pending action; act comes first in every game")
        state, action = self._state, self._action
        settings = self.settings
        bootstrap = outcome is None or (outcome is Outcome.TIE and settings.tie_update == "bootstrap")
        if bootstrap:
            next_state = state_index(observation)
            target = reward + settings.discount * float(numpy.max(self.q[next_state]))
        else:
            target = reward  # WIN, LOSS or a terminal TIE: the final observation is never encoded
        self.q[state, action] += settings.learning_rate * (target - self.q[state, action])
        self._state = next_state if outcome is None else None
        self._action = None
