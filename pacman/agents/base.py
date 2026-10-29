"""Common agent contract, independent of the environment implementation."""

import random
from typing import Protocol, runtime_checkable

from pacman.env import Observation, Outcome
from pacman.grid import Direction


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, rng: random.Random) -> None:
        """Start a game with fresh transition caches; retain learned policy data."""
        ...

    def act(self, observation: Observation) -> Direction:
        """Choose an action. The runner never calls this after a final outcome."""
        ...

    def observe(
        self, reward: float, observation: Observation, outcome: Outcome | None,
    ) -> None:
        """Receive each transition, including the explicit final referee ruling.

        WIN/LOSS/TIE end action requests. A learned agent's configured tie_update
        independently determines whether the final TIE update bootstraps.
        """
        ...
