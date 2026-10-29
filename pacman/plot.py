"""The learning-curve PNG. This is the only module that imports Matplotlib."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot  # noqa: E402  (the backend must be chosen first)


def save_curve(path: str | Path, curve: Iterable[tuple[int, float]]) -> None:
    """Plot (episodes completed, validation win rate) points and write them as a PNG."""
    points = list(curve)
    figure, axes = pyplot.subplots(figsize=(7, 4))
    try:
        if points:
            episodes, rates = zip(*points)
            axes.plot(episodes, [100 * rate for rate in rates], marker="o")
        axes.set_xlabel("training episodes")
        axes.set_ylabel("greedy validation win rate (%)")
        axes.set_ylim(0, 100)
        axes.set_title("Learning curve")
        axes.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(path, dpi=120)
    finally:
        pyplot.close(figure)
