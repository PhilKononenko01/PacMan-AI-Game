"""NFR-1: the same seeds give the same games and the same training, in and across processes."""

import json
import os
import subprocess
import sys

import numpy

from pacman import evaluate as evaluate_module
from pacman import train as train_module
from pacman.agents.learned import LearnedAgent
from pacman.agents.pathfinding import PathfindingAgent
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config, LearnerSettings
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.referee import Referee
from pacman.train import train

SEEDS = [0, 1, 2, 1_000_000, 1_000_001]

PLAY_SCRIPT = """
import json, sys
from pacman.agents.pathfinding import PathfindingAgent
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.referee import Referee
config = Config()
out = []
for agent in (RandomAgent(), PathfindingAgent()):
    for seed in %s:
        r = play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
        out.append([r.seed, r.agent, r.outcome.value, r.steps, r.episode_return])
print(json.dumps(out))
"""


def records_in_process(config=Config()):
    out = []
    for agent in (RandomAgent(), PathfindingAgent()):
        for seed in SEEDS:
            r = play_episode(Environment(config), Referee(config.game.max_steps), agent, seed)
            out.append([r.seed, r.agent, r.outcome.value, r.steps, r.episode_return])
    return out


def test_the_same_seeds_give_identical_records_twice_in_one_process():
    assert records_in_process() == records_in_process()
    config = Config()
    learned = LearnedAgent(config.learner)
    learned.q[:, 1] = 1.0  # a fixed, non-trivial greedy policy
    first = [play_episode(Environment(config), Referee(100), learned, seed) for seed in SEEDS]
    second = [play_episode(Environment(config), Referee(100), learned, seed) for seed in SEEDS]
    assert first == second


def test_records_are_identical_in_subprocesses_with_different_hash_seeds():
    expected = records_in_process()
    for hash_seed in ("1", "424242"):
        result = subprocess.run(
            [sys.executable, "-c", PLAY_SCRIPT % SEEDS], check=True, capture_output=True,
            text=True, env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert json.loads(result.stdout) == expected


def test_a_short_training_run_gives_a_byte_identical_table_twice(monkeypatch):
    monkeypatch.setattr(train_module, "CURVE_EVERY", 100)
    monkeypatch.setattr(train_module, "CURVE_GAMES", 5)
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 5)
    config = Config(learner=LearnerSettings(episodes=200))
    first, second = train(config), train(config)
    assert first.policy.q.tobytes() == second.policy.q.tobytes()
    assert numpy.count_nonzero(first.policy.q) > 0
    assert first.curve == second.curve and len(first.curve) == 2
    assert first.validation == second.validation
