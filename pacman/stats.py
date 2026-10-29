"""Wilson intervals and the paired one-sided exact McNemar test (§13.2).

Pure functions on counts; nothing here imports the game, the CLI or training.
"""

from collections.abc import Iterable, Mapping
from fractions import Fraction
from math import comb, sqrt

Z_95 = 1.959963984540054


def _count(name: str, value: object, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        upper = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a non-negative integer{upper}")
    return value


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials, 95% by default."""
    if _count("n", n) < 1:
        raise ValueError("n must be at least 1")
    _count("k", k, n)
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b of the b + c discordant pairs favour the learned agent) under no difference.

    `b` counts seeds only the learned agent won, `c` seeds only the baseline won.
    The sum is exact binomial arithmetic; `p = 1.0` when there are no discordant pairs.
    """
    n = _count("b", b) + _count("c", c)
    if n == 0:
        return 1.0
    return float(Fraction(sum(comb(n, k) for k in range(b, n + 1)), 2**n))


def discordant_pairs(
    learned_won: Mapping[int, bool], baseline_won: Mapping[int, bool],
) -> tuple[int, int]:
    """(b, c) for McNemar from per-seed win flags of the learned agent and a baseline."""
    if learned_won.keys() != baseline_won.keys():
        raise ValueError("a paired test needs the same seeds for both agents")
    b = sum(learned_won[seed] and not baseline_won[seed] for seed in learned_won)
    c = sum(baseline_won[seed] and not learned_won[seed] for seed in learned_won)
    return b, c


def mean_steps(steps: Iterable[int]) -> float | None:
    """Mean steps over won episodes; None when there is no win to average."""
    values = list(steps)
    return sum(values) / len(values) if values else None
