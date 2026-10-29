"""Candidate drawing, structural filters, split hashing and the draw limit (FR-2)."""

from hashlib import sha256
import random

import pytest

from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.grid import CORNERS
from pacman.mapgen import (
    MAX_DRAWS,
    GameMap,
    MapGenerationError,
    Split,
    draw_candidate,
    generate,
    is_connected,
    is_winnable,
    layout_bytes,
    loop_count,
    split_bucket,
    split_of_map,
    split_of_seed,
)
from pacman.rng import derive

DEFAULTS = Config()
EMPTY = GameMap(())
RING_ROUND_BLOCK = GameMap({(row, col) for row in (1, 2, 3) for col in (1, 2, 3)})
SEVEN_WALLS_CONNECTED = GameMap({(1, 1), (1, 2), (1, 3), (2, 2), (3, 1), (3, 2), (3, 3)})

GOLDEN = [
    # (map, layout bytes, SHA-256, bucket, split, seed producing it)
    (
        EMPTY,
        b"5x5:0000000000000000000000000",
        "b7bce985c39409cc0504aa9dd356c00f862dd5e048e4c7df57156e90d2fa3ba4",
        96, Split.TEST, 2_000_000,
    ),
    (
        GameMap({(1, 2), (1, 3)}),
        b"5x5:0000000110000000000000000",
        "3d51ab9a4b367db5950aa90bcf713327b097ed2b46493f6f909d69fbbebef908",
        84, Split.VAL, 1_000_000,
    ),
    (
        GameMap({(3, 4), (4, 1)}),
        b"5x5:0000000000000000000101000",
        "dafd13479351b71d98d6ae1a373db43ff366d020b157b886ed5c314d1a0eb1bc",
        28, Split.TRAIN, 0,
    ),
]


class CountingRandom(random.Random):
    calls = 0

    def random(self):
        self.calls += 1
        return super().random()


@pytest.mark.parametrize("wall_density", [0.0, 0.5, 1.0])
def test_every_candidate_consumes_exactly_21_draws(wall_density):
    rng = CountingRandom(7)
    draw_candidate(rng, wall_density)
    assert rng.calls == 21


def test_candidate_walls_follow_the_draws_in_row_major_non_corner_order():
    class Scripted(random.Random):
        def __init__(self):
            self.values = iter([0.9] * 3 + [0.1] + [0.9] * 17)

        def random(self):
            return next(self.values)

    # The fourth non-corner cell in row-major order is (1, 0).
    assert draw_candidate(Scripted(), 0.2) == GameMap({(1, 0)})


def test_candidates_never_wall_a_corner():
    rng = derive(5, "map")
    for _ in range(200):
        assert not draw_candidate(rng, 0.9).walls & set(CORNERS)


def test_connectivity_is_reachability_from_the_start_corner():
    assert is_connected(EMPTY)
    assert is_connected(RING_ROUND_BLOCK)
    assert not is_connected(GameMap({(0, 1), (1, 0)}))


def test_loop_count_is_edges_minus_cells_plus_one():
    assert loop_count(EMPTY) == 16
    assert loop_count(RING_ROUND_BLOCK) == 1
    assert loop_count(GameMap({(2, 2)})) == 13


def test_wall_cap_rejects_a_connected_seven_wall_map():
    assert is_connected(SEVEN_WALLS_CONNECTED)
    assert not is_winnable(SEVEN_WALLS_CONNECTED, GameKnobs(max_walls=6, min_loops=0))
    assert is_winnable(SEVEN_WALLS_CONNECTED, GameKnobs(max_walls=7, min_loops=0))


def test_loop_floor_rejects_the_ring_round_a_block():
    assert not is_winnable(RING_ROUND_BLOCK, GameKnobs(max_walls=9, min_loops=4))
    assert is_winnable(RING_ROUND_BLOCK, GameKnobs(max_walls=9, min_loops=1))
    assert is_winnable(EMPTY, GameKnobs())


@pytest.mark.parametrize("game_map,layout,digest,bucket,split,seed", GOLDEN)
def test_layout_hash_and_bucket_golden_values(game_map, layout, digest, bucket, split, seed):
    assert layout_bytes(game_map) == layout
    assert sha256(layout).hexdigest() == digest
    assert split_bucket(game_map) == bucket
    assert split_of_map(game_map) is split


@pytest.mark.parametrize("game_map,layout,digest,bucket,split,seed", GOLDEN)
def test_golden_maps_for_the_first_seed_of_each_split(game_map, layout, digest, bucket, split, seed):
    assert generate(seed, split, DEFAULTS) == game_map


@pytest.mark.parametrize("seed,split", [
    (0, Split.TRAIN), (999_999, Split.TRAIN),
    (1_000_000, Split.VAL), (1_000_999, Split.VAL),
    (2_000_000, Split.TEST), (2_000_999, Split.TEST),
])
def test_split_of_seed_follows_the_seed_ranges(seed, split):
    assert split_of_seed(seed) is split


@pytest.mark.parametrize("seed", [-1, 1_001_000, 1_999_999, 2_001_000])
def test_split_of_seed_rejects_seeds_outside_every_range(seed):
    with pytest.raises(ValueError):
        split_of_seed(seed)


@pytest.mark.parametrize("split,first", [
    (Split.TRAIN, 0), (Split.VAL, 1_000_000), (Split.TEST, 2_000_000),
])
def test_generated_maps_belong_to_the_requested_split_and_pass_the_filters(split, first):
    for seed in range(first, first + 50):
        game_map = generate(seed, split, DEFAULTS)
        assert split_of_map(game_map) is split
        assert is_connected(game_map)
        assert is_winnable(game_map, DEFAULTS.game)
        assert not game_map.walls & set(CORNERS)


def test_same_seed_and_split_give_the_same_map():
    assert generate(42, Split.TRAIN, DEFAULTS) == generate(42, Split.TRAIN, DEFAULTS)


def test_non_generator_settings_do_not_change_the_map():
    changed = Config(
        game=GameKnobs(ghost_move_prob=0.5, max_steps=7),
        learner=LearnerSettings(reward_win=1.0, episodes=3, tie_update="terminal"),
    )
    for seed in range(20):
        assert generate(seed, Split.TRAIN, changed) == generate(seed, Split.TRAIN, DEFAULTS)


def test_generator_knobs_do_change_the_map():
    dense = Config(game=GameKnobs(wall_density=0.5, max_walls=25, min_loops=0))
    assert generate(0, Split.TRAIN, dense) != generate(0, Split.TRAIN, DEFAULTS)


def test_default_draw_limit_is_the_architecture_constant():
    assert MAX_DRAWS == 10_000


def test_impossible_loop_floor_fails_with_a_useful_error():
    config = Config(game=GameKnobs(min_loops=17))
    with pytest.raises(MapGenerationError) as info:
        generate(3, Split.TRAIN, config, max_draws=25)
    message = str(info.value)
    for fragment in ("seed 3", "train", "25", "min_loops=17", "max_walls=6", "wall_density=0.2"):
        assert fragment in message


def test_wall_cap_zero_accepts_only_the_empty_map_which_is_a_test_map():
    config = Config(game=GameKnobs(max_walls=0))
    assert generate(2_000_000, Split.TEST, config) == EMPTY
    with pytest.raises(MapGenerationError):
        generate(0, Split.TRAIN, config, max_draws=50)


def test_generate_draws_only_from_the_map_stream():
    before = random.getstate()
    generate(1, Split.TRAIN, DEFAULTS)
    assert random.getstate() == before


@pytest.mark.parametrize("wall", [(0, 0), (0, 4), (4, 0), (4, 4)])
def test_game_map_rejects_a_wall_on_a_corner(wall):
    with pytest.raises(ValueError, match="corner"):
        GameMap({wall})


@pytest.mark.parametrize("wall", [(5, 0), (0, -1), (9, 9)])
def test_game_map_rejects_a_wall_off_the_grid(wall):
    with pytest.raises(ValueError, match="grid"):
        GameMap({wall})


def test_generate_rejects_a_split_that_is_not_the_seeds_own():
    with pytest.raises(ValueError, match="split"):
        generate(0, Split.TEST, DEFAULTS)


def test_is_winnable_requires_a_connected_map():
    sealed_corner = GameMap({(0, 1), (1, 0)})
    assert not is_connected(sealed_corner)
    with pytest.raises(ValueError, match="connected"):
        is_winnable(sealed_corner, DEFAULTS.game)
