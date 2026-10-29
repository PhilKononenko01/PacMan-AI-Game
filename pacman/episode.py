"""Episode record contract. The referee and runner are implemented in issue #9."""

from dataclasses import dataclass

from pacman.env import Outcome


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    seed: int
    agent: str
    outcome: Outcome
    steps: int
    episode_return: float
