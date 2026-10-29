"""Protect the stream contract used by map generation, actors and replay."""

import json
import os
import random
import subprocess
import sys

import pytest

from pacman.rng import derive, pick


def test_documented_seed_derivation_golden_value():
    assert derive(0, "map").random() == 0.8442537128977315


def test_component_streams_do_not_consume_each_others_draws():
    map_rng = derive(25, "map")
    control = derive(25, "map")
    ghost_rng = derive(25, "ghost")
    for _ in range(100):
        ghost_rng.random()
        assert map_rng.random() == control.random()
    assert derive(25, "map").random() != derive(25, "ghost").random()
    assert derive(25, "map").random() != derive(26, "map").random()


def test_helpers_never_change_global_random_state():
    before = random.getstate()
    pick(derive(3, "agent"), 4)
    assert random.getstate() == before


@pytest.mark.parametrize("draw,expected", [
    (0.0, 0), (0.249999, 0), (0.25, 1), (0.5, 2), (0.9999999999999999, 3),
])
def test_pick_boundaries_use_one_random_draw(draw, expected):
    class CountingRandom(random.Random):
        calls = 0

        def random(self):
            self.calls += 1
            return draw

    rng = CountingRandom(0)
    assert pick(rng, 4) == expected
    assert rng.calls == 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 25])
def test_pick_is_bounded_and_repeatable(n):
    first, second = derive(42, "agent"), derive(42, "agent")
    draws = [pick(first, n) for _ in range(200)]
    assert all(0 <= value < n for value in draws)
    assert draws == [pick(second, n) for _ in range(200)]


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "4"])
def test_invalid_pick_size_consumes_no_randomness(n):
    rng, control = derive(42, "agent"), derive(42, "agent")
    with pytest.raises(ValueError, match="positive integer"):
        pick(rng, n)
    assert rng.random() == control.random()


@pytest.mark.parametrize("seed,component", [(True, "map"), ("1", "map"), (1, ""), (1, "👻")])
def test_invalid_seed_or_component_is_rejected(seed, component):
    with pytest.raises(ValueError):
        derive(seed, component)


def test_stream_is_stable_across_processes_and_python_hash_seeds():
    code = (
        "import json; from pacman.rng import derive; "
        "rng = derive(1234, 'ghost'); "
        "print(json.dumps([rng.random() for _ in range(8)]))"
    )
    outputs = []
    for hash_seed in ("1", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True,
            text=True, env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.append(json.loads(result.stdout))
    rng = derive(1234, "ghost")
    assert outputs[0] == outputs[1] == [rng.random() for _ in range(8)]
