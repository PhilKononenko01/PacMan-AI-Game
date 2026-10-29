"""The one episode loop that joins an environment, the referee and an agent (§10.6)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pacman.env import Environment, Observation, Outcome
from pacman.grid import line_of_sight
from pacman.referee import Referee
from pacman.rng import derive

if TYPE_CHECKING:  # the loop never imports a concrete agent at runtime
    from pacman.agents.base import Agent


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    seed: int
    agent: str
    outcome: Outcome
    steps: int
    episode_return: float


@dataclass(frozen=True, slots=True)
class Frame:
    """What watch mode draws after reset and after each step (§14.6)."""

    observation: Observation
    t: int
    outcome: Outcome | None
    ghost_sees: bool
    max_steps: int


def play_episode(
    environment: Environment,
    referee: Referee,
    agent: "Agent",
    seed: int,
    on_frame: Callable[[Frame], None] | None = None,
) -> EpisodeRecord:
    """Play one game to the referee's ruling; nothing here branches on the agent."""
    observation = environment.reset(seed)
    referee.start()
    agent.reset(derive(seed, "agent"))
    episode_return = 0.0
    if on_frame is not None:
        on_frame(_frame(environment, observation, 0, None, referee.max_steps))
    while True:
        t = referee.next_move()
        result = environment.step(agent.act(observation), t)
        outcome = referee.rule(result)
        agent.observe(result.reward, result.observation, outcome)
        episode_return += result.reward
        if on_frame is not None:
            on_frame(_frame(environment, result.observation, t, outcome, referee.max_steps))
        if outcome is not None:
            return EpisodeRecord(seed, agent.name, outcome, t, episode_return)
        observation = result.observation


def _frame(
    environment: Environment,
    observation: Observation,
    t: int,
    outcome: Outcome | None,
    max_steps: int,
) -> Frame:
    assert environment.game_map is not None
    sees = line_of_sight(observation.ghost, observation.pacman, environment.game_map.is_open)
    return Frame(observation, t, outcome, sees, max_steps)
