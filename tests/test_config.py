"""Configuration inputs are public API and may come from hand-edited JSON."""

from dataclasses import FrozenInstanceError
import json

import pytest

from pacman.config import Config, GameKnobs, LearnerSettings


def test_defaults_match_the_approved_game_and_learner_settings():
    config = Config()
    assert json.loads(config.to_json()) == {
        "game": {
            "ghost_move_prob": 1.0, "wall_density": 0.20,
            "max_walls": 6, "min_loops": 4, "max_steps": 100,
        },
        "learner": {
            "reward_win": 100.0, "reward_loss": -100.0,
            "reward_corner": 1.0, "reward_step": -0.05,
            "learning_rate": 0.1, "discount": 0.99,
            "epsilon_start": 1.0, "epsilon_end": 0.05,
            "epsilon_decay_fraction": 0.5, "episodes": 50_000,
            "tie_update": "bootstrap",
        },
    }


def test_json_round_trip_preserves_nondefault_settings():
    config = Config(
        game=GameKnobs(ghost_move_prob=0.75, wall_density=0.3, max_steps=200),
        learner=LearnerSettings(episodes=200, epsilon_start=0.4, tie_update="terminal"),
    )
    assert Config.from_json(config.to_json()) == config
    assert Config.from_json(config.to_json()).to_json() == config.to_json()


def test_missing_json_keys_use_defaults():
    assert Config.from_json("{}") == Config()
    config = Config.from_json('{"game": {"max_walls": 3}, "learner": {}}')
    assert config.game == GameKnobs(max_walls=3)
    assert config.learner == LearnerSettings()


@pytest.mark.parametrize("text", [
    '{"games": {}}', '{"game": {"grid_size": 7}}',
    '{"learner": {"learning_rait": 0.1}}',
])
def test_unknown_keys_are_rejected_instead_of_silently_ignored(text):
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_json(text)


@pytest.mark.parametrize("text", [
    "null", "[]", '"config"', '{"game": null}',
    '{"game": []}', '{"learner": 3}', '{',
])
def test_invalid_json_structure_is_rejected(text):
    with pytest.raises(ValueError):
        Config.from_json(text)


@pytest.mark.parametrize("field,value", [
    ("ghost_move_prob", -0.01), ("ghost_move_prob", 1.01),
    ("wall_density", -1), ("wall_density", 2),
    ("max_walls", -1), ("max_walls", 1.5),
    ("min_loops", -1), ("min_loops", "4"),
    ("max_steps", 0), ("max_steps", True),
    ("ghost_move_prob", False), ("wall_density", None),
])
def test_invalid_game_settings_fail_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        GameKnobs(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("learning_rate", 0), ("learning_rate", 1.1),
    ("discount", -0.01), ("discount", 1.01),
    ("epsilon_start", 1.01), ("epsilon_end", -0.01),
    ("epsilon_decay_fraction", 0), ("epsilon_decay_fraction", 1.01),
    ("episodes", 0), ("episodes", 1_000_001), ("episodes", True),
    ("episodes", 3.5), ("tie_update", "stop"), ("tie_update", []),
    ("reward_win", "100"), ("reward_step", None),
])
def test_invalid_learner_settings_fail_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        LearnerSettings(**{field: value})


def test_exploration_end_cannot_exceed_start():
    with pytest.raises(ValueError, match="epsilon_end"):
        LearnerSettings(epsilon_start=0.1, epsilon_end=0.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_cannot_enter_rewards_or_probabilities(value):
    with pytest.raises(ValueError, match="finite"):
        LearnerSettings(reward_step=value)
    with pytest.raises(ValueError, match="finite"):
        GameKnobs(wall_density=value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json_values_are_rejected(constant):
    with pytest.raises(ValueError):
        Config.from_json('{"learner": {"reward_win": ' + constant + '}}')


def test_valid_boundaries_and_impossible_generator_settings_remain_expressible():
    # Map generation, not configuration, owns bounded rejection of these maps.
    game = GameKnobs(ghost_move_prob=0, wall_density=1, max_walls=0, min_loops=17, max_steps=1)
    learner = LearnerSettings(
        learning_rate=1, discount=0, epsilon_start=0, epsilon_end=0,
        epsilon_decay_fraction=1, episodes=1_000_000, tie_update="terminal",
    )
    assert Config.from_json(Config(game, learner).to_json()) == Config(game, learner)


def test_configuration_cannot_be_changed_during_an_episode():
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.game.max_steps = 1
    with pytest.raises(FrozenInstanceError):
        config.learner.tie_update = "terminal"
    with pytest.raises(FrozenInstanceError):
        config.game = GameKnobs(max_steps=1)


@pytest.mark.parametrize("parts", [{"game": {}}, {"learner": {}}])
def test_constructor_requires_validated_config_sections(parts):
    with pytest.raises(ValueError):
        Config(**parts)
