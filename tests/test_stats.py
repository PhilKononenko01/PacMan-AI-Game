"""Wilson intervals and the paired one-sided exact McNemar test (FR-10, FR-11)."""

from fractions import Fraction
from math import comb
import subprocess
import sys

import pytest

from pacman.stats import Z_95, discordant_pairs, mcnemar_one_sided, mean_steps, wilson


def percent(interval):
    return tuple(round(bound * 100, 1) for bound in interval)


def test_z_is_the_architecture_constant():
    assert Z_95 == 1.959963984540054


def test_wilson_matches_the_evaluation_report_examples():
    assert percent(wilson(0, 1000)) == (0.0, 0.4)
    assert percent(wilson(142, 1000)) == (12.2, 16.5)
    assert percent(wilson(858, 1000)) == (83.5, 87.8)
    assert percent(wilson(1000, 1000)) == (99.6, 100.0)


def test_wilson_matches_a_textbook_value():
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=5e-4)
    assert high == pytest.approx(0.7634, abs=5e-4)


def test_wilson_bounds_are_exact_at_zero_and_all():
    assert wilson(0, 7)[0] == 0.0
    assert wilson(7, 7)[1] == 1.0
    assert wilson(0, 1)[0] == 0.0
    assert wilson(1, 1)[1] == 1.0


def test_wilson_is_symmetric():
    low, high = wilson(3, 20)
    mirrored_low, mirrored_high = wilson(17, 20)
    assert low == pytest.approx(1 - mirrored_high)
    assert high == pytest.approx(1 - mirrored_low)


def test_wilson_contains_the_observed_rate_and_lies_in_the_unit_interval():
    for n in (1, 2, 10, 1000):
        for k in range(n + 1):
            low, high = wilson(k, n)
            assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_accepts_another_z():
    low, high = wilson(50, 100, z=0.0)
    assert (low, high) == (0.5, 0.5)


@pytest.mark.parametrize("k,n", [(0, 0), (-1, 10), (11, 10), (1.5, 10), (True, 10), (3, 2.0)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        wilson(k, n)


@pytest.mark.parametrize("b,c,expected", [
    (0, 0, 1.0),
    (1, 0, 0.5),
    (0, 1, 1.0),
    (2, 0, 0.25),
    (0, 2, 1.0),
    (3, 1, 0.3125),
    (1, 3, 0.9375),
    (5, 0, 0.03125),
    (10, 2, 79 / 4096),
    (2, 10, 4083 / 4096),
])
def test_mcnemar_hand_computed_values(b, c, expected):
    assert mcnemar_one_sided(b, c) == pytest.approx(expected, rel=0, abs=1e-15)


def test_mcnemar_is_exact_binomial_arithmetic():
    b, c = 10, 2
    n = b + c
    exact = Fraction(sum(comb(n, k) for k in range(b, n + 1)), 2**n)
    assert mcnemar_one_sided(b, c) == float(exact)


def test_mcnemar_direction_matters():
    assert mcnemar_one_sided(9, 1) < 0.05
    assert mcnemar_one_sided(1, 9) > 0.95
    assert mcnemar_one_sided(5, 5) == pytest.approx(0.623046875)


def test_mcnemar_handles_large_counts_without_overflow():
    assert 0.0 <= mcnemar_one_sided(600, 400) < 1e-9
    assert 1 - 1e-9 < mcnemar_one_sided(400, 600) <= 1.0


@pytest.mark.parametrize("b,c", [(-1, 0), (0, -1), (1.0, 2), (True, 1), (None, 1)])
def test_mcnemar_rejects_bad_counts(b, c):
    with pytest.raises(ValueError):
        mcnemar_one_sided(b, c)


def test_discordant_pairs_count_only_disagreements_by_seed():
    learned = {0: True, 1: True, 2: False, 3: False, 4: True}
    baseline = {0: True, 1: False, 2: True, 3: False, 4: False}
    assert discordant_pairs(learned, baseline) == (2, 1)
    assert discordant_pairs(baseline, learned) == (1, 2)


def test_discordant_pairs_need_the_same_seeds():
    with pytest.raises(ValueError):
        discordant_pairs({0: True, 1: False}, {0: True})
    with pytest.raises(ValueError):
        discordant_pairs({0: True}, {0: True, 2: False})


def test_discordant_pairs_of_empty_evaluations_are_zero():
    assert discordant_pairs({}, {}) == (0, 0)
    assert mcnemar_one_sided(*discordant_pairs({}, {})) == 1.0


def test_mean_steps_to_a_win():
    assert mean_steps([]) is None
    assert mean_steps([13]) == 13.0
    assert mean_steps([10, 20, 12]) == pytest.approx(14.0)


def test_stats_imports_nothing_from_the_game():
    code = (
        "import sys, pacman.stats;"
        " print(sorted(m for m in sys.modules if m.startswith('pacman')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "['pacman', 'pacman.stats']"
