"""Versioned policy save/load and replay metadata (FR-9, FR-14, FR-15)."""

import json
import subprocess
import sys

import numpy
import pytest

from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.policy_io import (
    FORMAT_VERSION,
    TABLE_SHAPE,
    Policy,
    ValidationSummary,
    load_policy,
    save_policy,
)

CONFIG = Config(
    game=GameKnobs(ghost_move_prob=0.9, max_steps=80),
    learner=LearnerSettings(learning_rate=0.2, episodes=40_000, tie_update="terminal"),
)


def table(scale=0.5):
    return (numpy.arange(TABLE_SHAPE[0] * TABLE_SHAPE[1], dtype=numpy.float64) * scale).reshape(TABLE_SHAPE)


def summary(**overrides):
    fields = dict(seeds=(1_000_000, 1_000_999), games=1000, wins=250, losses=700, ties=50,
                  mean_steps_to_win=13.6)
    fields.update(overrides)
    return ValidationSummary(**fields)


def policy(**overrides):
    fields = dict(
        q=table(), config=CONFIG, training_seeds=(0, 39_999),
        learning_curve=((2500, 0.365), (5000, 0.41)), validation=summary(),
    )
    fields.update(overrides)
    return Policy(**fields)


def meta_of(path):
    with numpy.load(path, allow_pickle=False) as data:
        return json.loads(bytes(data["meta"]).decode("utf-8"))


def write_raw(path, q=None, meta=None, **arrays):
    """Write an .npz the way the format describes, with any field overridden."""
    if q is None:
        q = table()
    if meta is None:
        meta = meta_of_policy(policy())
    arrays = {"q": q, "meta": numpy.frombuffer(json.dumps(meta).encode("utf-8"), dtype=numpy.uint8),
              **arrays}
    with open(path, "wb") as file:
        numpy.savez(file, **arrays)


def meta_of_policy(policy_):
    return {
        "format_version": FORMAT_VERSION,
        "config": json.loads(policy_.config.to_json()),
        "training_seeds": list(policy_.training_seeds),
        "learning_curve": [list(point) for point in policy_.learning_curve],
        "validation": {
            "seeds": list(policy_.validation.seeds),
            "games": policy_.validation.games,
            "wins": policy_.validation.wins,
            "losses": policy_.validation.losses,
            "ties": policy_.validation.ties,
            "mean_steps_to_win": policy_.validation.mean_steps_to_win,
        },
    }


def test_table_shape_and_version_are_the_architecture_constants():
    assert TABLE_SHAPE == (122_880, 4)
    assert FORMAT_VERSION == 1


def test_round_trip_restores_table_config_and_metadata(tmp_path):
    original = policy()
    path = tmp_path / "policy.npz"
    save_policy(path, original)
    loaded = load_policy(path)
    assert numpy.array_equal(loaded.q, original.q)
    assert loaded.q.dtype == numpy.float64 and loaded.q.shape == TABLE_SHAPE
    assert loaded.config == CONFIG
    assert loaded.config.to_json() == CONFIG.to_json()
    assert loaded.training_seeds == (0, 39_999)
    assert loaded.learning_curve == ((2500, 0.365), (5000, 0.41))
    assert loaded.validation == summary()


def test_the_file_holds_exactly_the_two_documented_arrays(tmp_path):
    path = tmp_path / "policy.npz"
    save_policy(path, policy())
    assert path.exists()
    with numpy.load(path, allow_pickle=False) as data:
        assert sorted(data.files) == ["meta", "q"]
        assert data["q"].dtype == numpy.float64 and data["q"].shape == TABLE_SHAPE
        assert data["meta"].dtype == numpy.uint8 and data["meta"].ndim == 1
    meta = meta_of(path)
    assert set(meta) == {"format_version", "config", "training_seeds", "learning_curve", "validation"}
    assert meta["format_version"] == 1
    assert meta["training_seeds"] == [0, 39_999]
    assert meta["learning_curve"] == [[2500, 0.365], [5000, 0.41]]
    assert meta["validation"] == {
        "seeds": [1_000_000, 1_000_999], "games": 1000, "wins": 250, "losses": 700, "ties": 50,
        "mean_steps_to_win": 13.6,
    }
    assert meta["config"] == json.loads(CONFIG.to_json())


def test_zero_wins_store_a_null_mean(tmp_path):
    path = tmp_path / "policy.npz"
    save_policy(path, policy(validation=summary(wins=0, losses=950, mean_steps_to_win=None)))
    assert meta_of(path)["validation"]["mean_steps_to_win"] is None
    assert load_policy(path).validation.mean_steps_to_win is None


def test_an_empty_learning_curve_round_trips(tmp_path):
    path = tmp_path / "policy.npz"
    save_policy(path, policy(learning_curve=()))
    assert load_policy(path).learning_curve == ()


def test_loading_never_allows_pickles(tmp_path):
    path = tmp_path / "policy.npz"
    with open(path, "wb") as file:
        numpy.savez(file, q=table(), meta=numpy.array([{"format_version": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_policy(path)


def test_unknown_format_version_is_refused(tmp_path):
    path = tmp_path / "policy.npz"
    meta = meta_of_policy(policy())
    meta["format_version"] = 2
    write_raw(path, meta=meta)
    with pytest.raises(ValueError, match="version"):
        load_policy(path)


@pytest.mark.parametrize("q", [
    numpy.zeros((10, 4)), numpy.zeros((122_880, 5)), numpy.zeros(122_880 * 4),
    numpy.zeros(TABLE_SHAPE, dtype=numpy.float32),
])
def test_wrong_table_shape_or_dtype_is_refused(tmp_path, q):
    path = tmp_path / "policy.npz"
    write_raw(path, q=q)
    with pytest.raises(ValueError, match="q"):
        load_policy(path)


@pytest.mark.parametrize("missing", [
    "format_version", "config", "training_seeds", "learning_curve", "validation",
])
def test_a_missing_top_level_key_is_refused(tmp_path, missing):
    path = tmp_path / "policy.npz"
    meta = meta_of_policy(policy())
    del meta[missing]
    write_raw(path, meta=meta)
    with pytest.raises(ValueError, match=missing):
        load_policy(path)


@pytest.mark.parametrize("missing", ["seeds", "games", "wins", "losses", "ties", "mean_steps_to_win"])
def test_a_missing_validation_key_is_refused(tmp_path, missing):
    path = tmp_path / "policy.npz"
    meta = meta_of_policy(policy())
    del meta["validation"][missing]
    write_raw(path, meta=meta)
    with pytest.raises(ValueError, match=missing):
        load_policy(path)


def test_a_file_without_the_meta_array_is_refused(tmp_path):
    path = tmp_path / "policy.npz"
    with open(path, "wb") as file:
        numpy.savez(file, q=table())
    with pytest.raises(ValueError, match="meta"):
        load_policy(path)


def test_an_invalid_config_inside_the_file_is_refused(tmp_path):
    path = tmp_path / "policy.npz"
    meta = meta_of_policy(policy())
    meta["config"]["game"]["wall_density"] = 7
    write_raw(path, meta=meta)
    with pytest.raises(ValueError, match="wall_density"):
        load_policy(path)
    meta = meta_of_policy(policy())
    meta["config"]["game"]["ghost_speed"] = 1
    write_raw(path, meta=meta)
    with pytest.raises(ValueError, match="ghost_speed"):
        load_policy(path)


def test_malformed_metadata_json_is_refused(tmp_path):
    path = tmp_path / "policy.npz"
    with open(path, "wb") as file:
        numpy.savez(file, q=table(), meta=numpy.frombuffer(b"{not json", dtype=numpy.uint8))
    with pytest.raises(ValueError):
        load_policy(path)


def test_policy_rejects_a_table_of_the_wrong_shape_or_dtype():
    with pytest.raises(ValueError, match="q"):
        policy(q=numpy.zeros((5, 4)))
    with pytest.raises(ValueError, match="q"):
        policy(q=numpy.zeros(TABLE_SHAPE, dtype=numpy.int64))


def test_policy_rejects_inconsistent_metadata():
    with pytest.raises(ValueError):
        policy(training_seeds=(10, 5))
    with pytest.raises(ValueError):
        policy(validation=summary(games=999))
    with pytest.raises(ValueError):
        policy(validation=summary(wins=0, losses=950, mean_steps_to_win=12.0))
    with pytest.raises(ValueError):
        policy(validation=summary(mean_steps_to_win=None))
    with pytest.raises(ValueError):
        policy(validation=summary(seeds=(1_000_999, 1_000_000)))


def test_non_finite_metadata_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):
        save_policy(tmp_path / "policy.npz", policy(learning_curve=((2500, float("nan")),)))


def test_policy_io_imports_only_configuration_and_numpy():
    code = (
        "import sys, pacman.policy_io;"
        " print(sorted(m for m in sys.modules if m.startswith('pacman')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "['pacman', 'pacman.config', 'pacman.policy_io']"
