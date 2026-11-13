"""Draw the frozen cross-play results and all five training replicates."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(directory):
    directory = Path(directory)
    data = json.loads((directory / "aggregate.json").read_text())
    names = ["baseline-vs-fixed", "baseline-vs-learned", "adapted-vs-fixed", "adapted-vs-learned"]
    labels = ["Before / fixed ghost", "Before / learned ghost", "After / fixed ghost", "After / learned ghost"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    left = np.zeros(4)
    for outcome, label, color in [("win", "Win", "#2679b4"), ("loss", "Capture", "#c96559"), ("tie", "Timeout", "#b9bec5")]:
        values = np.array([100 * data["test"][name][outcome] / data["test"][name]["games"] for name in names])
        axes[0].barh(labels, values, left=left, label=label, color=color)
        left += values
    for index, name in enumerate(names):
        rate = 100 * data["test"][name]["win_rate"]
        axes[0].text(rate / 2, index, f"{rate:.2f}%", ha="center", va="center", color="white", weight="bold")
    axes[0].invert_yaxis()
    axes[0].axvline(55, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 100), xlabel="Share of all scheduled test games (%)", title="Same rules, matched schedules")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    positions = np.arange(5)
    for offset, name, label, color in [(-0.18, "baseline-vs-learned", "Before additional training", "#a9b6c2"),
                                       (0.18, "adapted-vs-learned", "After additional training", "#2679b4")]:
        rates = np.array(data["test"][name]["replicate_win_rates"]) * 100
        axes[1].bar(positions + offset, rates, 0.36, label=label, color=color)
    axes[1].axhline(55, color="black", linestyle="--", linewidth=1, label="55% target")
    axes[1].set(ylim=(0, 100), xticks=positions, xlabel="Independent training replicate", ylabel="Pac-Man wins (%)",
                title="Against each frozen learned ghost")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Two-pellet experiment: 10,000 test games per matchup", fontsize=13)
    fig.tight_layout()
    fig.savefig(directory / "results.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
