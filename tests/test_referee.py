"""The referee alone counts moves and calls the tie (FR-4, OQ-14)."""

import pytest

from pacman.env import KnownCell, Observation, Outcome, StepResult
from pacman.referee import Referee

OBSERVATION = Observation((0, 0), (4, 4), frozenset({(0, 0)}), (KnownCell.UNKNOWN,) * 25)


def result(outcome=None):
    return StepResult(OBSERVATION, -0.05, outcome)


def test_moves_count_from_one_after_start():
    referee = Referee(100)
    referee.start()
    assert referee.moves == 0
    assert [referee.next_move() for _ in range(3)] == [1, 2, 3]
    assert referee.moves == 3


def test_start_resets_the_count_for_the_next_game():
    referee = Referee(100)
    referee.start()
    referee.next_move()
    referee.next_move()
    referee.start()
    assert referee.next_move() == 1


def test_tie_is_called_exactly_on_move_max_steps():
    referee = Referee(3)
    referee.start()
    rulings = []
    for _ in range(3):
        referee.next_move()
        rulings.append(referee.rule(result()))
    assert rulings == [None, None, Outcome.TIE]


@pytest.mark.parametrize("outcome", [Outcome.WIN, Outcome.LOSS])
def test_a_win_or_loss_passes_through_on_any_move(outcome):
    referee = Referee(2)
    referee.start()
    referee.next_move()
    assert referee.rule(result(outcome)) is outcome
    referee.next_move()
    assert referee.rule(result(outcome)) is outcome


def test_the_referee_reads_max_steps_from_its_own_field_only():
    referee = Referee(1)
    assert referee.max_steps == 1
    referee.start()
    referee.next_move()
    assert referee.rule(result()) is Outcome.TIE


def test_observations_carry_no_move_count():
    assert not hasattr(OBSERVATION, "moves")
    assert not hasattr(OBSERVATION, "t")
