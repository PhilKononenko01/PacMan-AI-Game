"""Standalone random-versus-random experiment; leaves the production ghost unchanged."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess

from pacman.agents.random_agent import RandomAgent
from pacman.config import Config
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.ghost import Ghost
from pacman.grid import Direction, move
from pacman.referee import Referee
from pacman.rng import derive, pick


class UniformRandomGhost(Ghost):
    def act(self, pacman, game_map, t):
        direction = Direction(pick(self.rng, 4))
        self.cell = move(self.cell, direction, game_map.is_open)
        self.last_visit[self.cell] = t


class RandomVersusRandomEnvironment(Environment):
    def reset(self, seed, game_map=None):
        observation = super().reset(seed, game_map)
        self.ghost = UniformRandomGhost(derive(seed, "ghost"))
        return observation


if __name__ == "__main__":
    config = Config()
    environment = RandomVersusRandomEnvironment(config)
    referee = Referee(config.game.max_steps)
    agent = RandomAgent()
    records = [play_episode(environment, referee, agent, seed)
               for seed in range(1_000_000, 1_000_100)]
    assert len(records) == 100
    assert len({record.seed for record in records}) == 100
    counts = Counter(record.outcome.value for record in records)
    assert sum(counts.values()) == 100
    replayed = []
    for outcome in counts:
        expected = next(record for record in records if record.outcome.value == outcome)
        actual = play_episode(RandomVersusRandomEnvironment(config), Referee(100),
                              RandomAgent(), expected.seed)
        assert actual == expected
        replayed.append(expected.seed)
    result = {
        "experiment": "random Pac-Man versus random ghost",
        "games": 100,
        "split": "validation",
        "seeds": [1_000_000, 1_000_099],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "pacman_policy": "uniform among up, right, down, left",
        "ghost_policy": "uniform among up, right, down, left; no chase or search",
        "blocked_moves": "stay in place and consume the turn",
        "outcome_perspective": "Pac-Man",
        "config": json.loads(config.to_json()),
        "wins": counts["win"],
        "losses": counts["loss"],
        "ties": counts["tie"],
        "mean_steps": sum(record.steps for record in records) / 100,
        "verified_replay_seeds": replayed,
        "records": [{**asdict(record), "outcome": record.outcome.value} for record in records],
    }
    out = Path(__file__).resolve().parent
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = (f"100 games: {counts['win']} wins, {counts['loss']} losses, "
               f"{counts['tie']} ties (Pac-Man outcomes).\n"
               "Both players choose uniformly among four directions; blocked moves stay.\n"
               "Validation seeds 1000000 through 1000099; default map rules; 100-turn limit.\n")
    (out / "summary.txt").write_text(summary)
    print(summary, end="")
