"""The random baseline: one uniform draw per step (§10.2)."""

import random

from pacman.env import Observation, Outcome
from pacman.grid import Direction
from pacman.rng import pick


class RandomAgent:
    name = "random"

    def reset(self, rng: random.Random) -> None:
        self.rng = rng

    def act(self, observation: Observation) -> Direction:
        return Direction(pick(self.rng, 4))

    def observe(self, reward: float, observation: Observation, outcome: Outcome | None) -> None:
        return None
