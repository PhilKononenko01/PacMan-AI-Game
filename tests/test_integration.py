"""The whole pipeline in one process: train, save, load, evaluate, record, replay."""

import numpy
import pytest

from pacman import evaluate as evaluate_module
from pacman import train as train_module
from pacman.agents import AGENT_NAMES, make_agent
from pacman.agents.learned import LearnedAgent
from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.env import Environment, Outcome
from pacman.episode import play_episode
from pacman.evaluate import compare, read_eval_json, run_split, summarise, write_eval_json
from pacman.mapgen import Split
from pacman.policy_io import load_policy
from pacman.referee import Referee
from pacman.train import train, write_outputs

CONFIG = Config(game=GameKnobs(max_steps=40), learner=LearnerSettings(episodes=30))


@pytest.fixture(autouse=True)
def small_sizes(monkeypatch):
    monkeypatch.setattr(train_module, "CURVE_EVERY", 10)
    monkeypatch.setattr(train_module, "CURVE_GAMES", 3)
    monkeypatch.setattr(evaluate_module, "FULL_GAMES", 6)


def test_train_save_load_evaluate_record_and_replay(tmp_path):
    out = write_outputs(tmp_path / "run", train(CONFIG))
    policy = load_policy(out / "policy.npz")
    assert policy.training_seeds == (0, 29) and len(policy.learning_curve) == 3

    records = {
        name: run_split(policy.config, make_agent(name, policy.config, policy), Split.VAL, 6)
        for name in AGENT_NAMES
    }
    summaries = [summarise(name, agent_records) for name, agent_records in records.items()]
    assert [summary.games for summary in summaries] == [6, 6, 6]
    for name in ("random", "pathfinding"):
        comparison = compare(records["learned"], records[name])
        assert 0.0 <= comparison.p <= 1.0

    path = out / "eval-val.json"
    write_eval_json(path, policy.config, [r for rs in records.values() for r in rs])
    config, loaded = read_eval_json(path)
    assert config == policy.config and len(loaded) == 18

    # Every recorded game replays to the same ruling, step count and return (FR-14).
    for record in loaded:
        frames = []
        agent = make_agent(record.agent, config, policy)
        replay = play_episode(
            Environment(config), Referee(config.game.max_steps), agent, record.seed, on_frame=frames.append,
        )
        assert replay == record
        assert frames[-1].outcome is record.outcome and frames[-1].t == record.steps
        assert len(frames) == record.steps + 1


def test_the_learned_agent_is_never_asked_after_a_ruling_and_starts_each_game_fresh():
    class Spy(LearnedAgent):
        acts_after_end = 0
        ended = False

        def reset(self, rng):
            super().reset(rng)
            self.ended = False

        def act(self, observation):
            if self.ended:
                self.acts_after_end += 1
            return super().act(observation)

        def observe(self, reward, observation, outcome):
            super().observe(reward, observation, outcome)
            if outcome is not None:
                self.ended = True
                assert self._action is None and self._next_state is None

    agent = Spy(CONFIG.learner, training=True)
    agent.epsilon = 0.3
    outcomes = set()
    for seed in range(20):
        record = play_episode(Environment(CONFIG), Referee(CONFIG.game.max_steps), agent, seed)
        outcomes.add(record.outcome)
    assert agent.acts_after_end == 0
    assert outcomes & {Outcome.LOSS, Outcome.TIE, Outcome.WIN}
    assert numpy.count_nonzero(agent.q) > 0
