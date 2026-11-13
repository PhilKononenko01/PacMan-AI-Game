from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from pacman.hybrid.artifacts import load_policy, save_policy
from pacman.hybrid.learning import FrozenTable
from pacman.hybrid.models import Rules

PAC = (0, 3, 40, 2, 1, 2, 0, 3)
GHOST = (0, 24, 4, 3, 1, 2, 3, 1)


def rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields.update(changes)
    np.savez_compressed(path, **fields)


@pytest.mark.parametrize("role,state", [("pacman", PAC), ("ghost", GHOST)])
def test_roundtrip_and_frozen_aliases(tmp_path, role, state):
    path = tmp_path / "policy.npz"
    source = {state: [1.0, 2.0, 3.0, 4.0]}
    table = FrozenTable(source)
    source[state][0] = 99.0
    metadata = {"opponents": ["abc"], "details": {"games": 5}}
    digest = save_policy(path, role, table, Rules(), metadata)
    metadata["opponents"].append("mutated")
    policy = load_policy(path, expected_role=role, rules=Rules())
    assert digest == policy.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert policy.table.values(state) == (1.0, 2.0, 3.0, 4.0)
    assert policy.metadata["opponents"] == ("abc",)
    with pytest.raises(TypeError):
        policy.table.rows[state] = (0.0,) * 4
    with pytest.raises(TypeError):
        policy.metadata["details"]["games"] = 6
    with pytest.raises(FrozenInstanceError):
        policy.role = "other"
    with pytest.raises(FileExistsError):
        save_policy(path, role, table, Rules(), {})
    assert load_policy(path).digest == digest


@pytest.mark.parametrize("role", ["pacman", "ghost"])
def test_empty_table(tmp_path, role):
    path = tmp_path / "empty.npz"
    save_policy(path, role, FrozenTable({}), Rules(), {})
    assert dict(load_policy(path).table.rows) == {}
    with np.load(path, allow_pickle=False) as archive:
        assert archive["states"].shape == (0, 8)
        assert archive["q"].shape == (0, 4)


def test_mismatch_and_legacy(tmp_path):
    path = tmp_path / "policy.npz"
    save_policy(path, "pacman", FrozenTable({PAC: (0.0,) * 4}), Rules(), {})
    with pytest.raises(ValueError, match="role"):
        load_policy(path, "ghost")
    with pytest.raises(ValueError, match="rules"):
        load_policy(path, rules=Rules(max_steps=99))
    rewrite(path, features_version=np.array("old"))
    with pytest.raises(ValueError, match="feature"):
        load_policy(path)
    np.savez(path, version=np.array(1), q=np.zeros((1, 4)))
    with pytest.raises(ValueError, match="hybrid"):
        load_policy(path)


@pytest.mark.parametrize("changes", [
    {"q": np.full((1, 4), np.nan)},
    {"q": np.full((1, 4), np.inf)},
    {"states": np.array([PAC], dtype=float)},
    {"states": np.array([(-1,) + PAC[1:]], dtype=np.int64)},
    {"states": np.array([(4,) + PAC[1:]], dtype=np.int64)},
    {"states": np.array([PAC, PAC]), "q": np.zeros((2, 4))},
    {"q": np.zeros((1, 3))},
    {"metadata": np.array('{"x":NaN}')},
    {"metadata": np.array('{"x":1e400}')},
    {"metadata": np.array('{"x":1,"x":2}')},
    {"rules": np.array('{"max_steps":true}')},
    {"role": np.array("other")},
    {"namespace": np.array("pacman.policy")},
    {"rules_version": np.array("legacy")},
    {"version": np.array(2)},
    {"states": np.array([[object()] * 8], dtype=object)},
])
def test_reject_malformed_artifact(tmp_path, changes):
    path = tmp_path / "policy.npz"
    save_policy(path, "pacman", FrozenTable({PAC: (0.0,) * 4}), Rules(), {})
    rewrite(path, **changes)
    with pytest.raises(ValueError):
        load_policy(path)


def test_reject_invalid_save_without_leaving_artifact(tmp_path):
    path = tmp_path / "bad.npz"
    for table, metadata in [(FrozenTable({PAC: (float("inf"),) * 4}), {}),
                            (FrozenTable({PAC: (0.0,) * 4}), {"x": float("nan")})]:
        with pytest.raises(ValueError):
            save_policy(path, "pacman", table, Rules(), metadata)
        assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_ghost_bounds(tmp_path):
    path = tmp_path / "ghost.npz"
    save_policy(path, "ghost", FrozenTable({GHOST: (0.0,) * 4}), Rules(), {})
    rewrite(path, states=np.array([(25,) + GHOST[1:]], dtype=np.int64))
    with pytest.raises(ValueError, match="bounds"):
        load_policy(path)


def test_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(ValueError):
        load_policy(path)
