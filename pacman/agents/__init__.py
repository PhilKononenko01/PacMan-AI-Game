"""Agent names and the factory the CLI builds them with."""

from pacman.config import Config

AGENT_NAMES = ("random", "pathfinding", "learned")


def make_agent(name: str, config: Config, policy=None):
    """Build a named agent; the learned agent needs a policy and always plays greedily here."""
    if name == "random":
        from pacman.agents.random_agent import RandomAgent

        return RandomAgent()
    if name == "pathfinding":
        from pacman.agents.pathfinding import PathfindingAgent

        return PathfindingAgent()
    if name == "learned":
        if policy is None:
            raise ValueError("the learned agent needs a policy file (--policy)")
        from pacman.agents.learned import LearnedAgent

        return LearnedAgent(config.learner, q=policy.q, training=False)
    raise ValueError(f"unknown agent {name!r}; choose one of {', '.join(AGENT_NAMES)}")
