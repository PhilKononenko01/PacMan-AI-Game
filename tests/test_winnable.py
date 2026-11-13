"""Which games can be won at all: the exact search against the predictable ghost."""

import pytest

from pacman.config import Config, GameKnobs
from pacman.env import Environment, Outcome
from pacman.winnable import is_winnable, winning_moves

VAL = range(1_000_000, 1_000_020)


@pytest.mark.parametrize("game", [
    GameKnobs(),
    GameKnobs(ghost_move_prob=0.5, max_steps=40),
    GameKnobs(pellets="anywhere"),
    GameKnobs(pellets="anywhere", pellet_count=2),
], ids=["default", "ghost-moves-half-the-time", "pellets-anywhere", "two-pellets"])
def test_every_win_the_search_finds_wins_the_real_game_on_its_last_move(game):
    config = Config(game=game)
    found = 0
    for seed in VAL:
        moves = winning_moves(config, seed)
        if moves is None:
            continue
        found += 1
        env = Environment(config)
        env.reset(seed)
        outcomes = [env.step(action, t).outcome for t, action in enumerate(moves, start=1)]
        assert outcomes == [None] * (len(moves) - 1) + [Outcome.WIN], seed
    assert found > 0


def test_a_ghost_that_never_moves_guards_its_own_corner_so_no_map_can_be_won():
    still = Config(game=GameKnobs(ghost_move_prob=0.0))
    assert not any(is_winnable(still, seed) for seed in range(1_000_000, 1_000_010))


def test_eleven_moves_are_too_few_to_reach_three_corners_on_any_map():
    # (0, 0) to (0, 4) to (4, 4) to (4, 0) is at least 12 moves.
    short = Config(game=GameKnobs(max_steps=11))
    assert not any(is_winnable(short, seed) for seed in range(1_000_000, 1_000_010))


def test_the_first_100_validation_maps_match_the_independent_search():
    # 48 of the first 100 were winnable in the separate search run on 2026-11-13.
    assert sum(is_winnable(Config(), seed) for seed in range(1_000_000, 1_000_100)) == 48


def test_learner_settings_do_not_change_winnability():
    from pacman.config import LearnerSettings

    other = Config(learner=LearnerSettings(learning_rate=0.5, episodes=10))
    assert [is_winnable(other, s) for s in VAL] == [is_winnable(Config(), s) for s in VAL]
