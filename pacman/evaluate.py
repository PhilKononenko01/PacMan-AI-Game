"""Paired evaluation on a split, summaries, the report layout, replay JSON and the
difficulty check. Everything returns data or strings; the CLI prints.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

from pacman.agents.pathfinding import PathfindingAgent
from pacman.agents.random_agent import RandomAgent
from pacman.config import Config, GameKnobs
from pacman.env import Environment, Outcome
from pacman.episode import EpisodeRecord, play_episode
from pacman.mapgen import SPLIT_SEEDS, Split
from pacman.policy_io import Policy, ValidationSummary
from pacman.referee import Referee
from pacman.stats import discordant_pairs, mcnemar_one_sided, mean_steps, wilson

FULL_GAMES = 1_000  # An evaluation plays every seed of its split.

TABLE_HEADER = "agent         wins                  losses                ties                  steps/win"


@dataclass(frozen=True, slots=True)
class Summary:
    agent: str
    games: int
    wins: int
    losses: int
    ties: int
    win_interval: tuple[float, float]
    loss_interval: tuple[float, float]
    tie_interval: tuple[float, float]
    mean_steps_to_win: float | None


@dataclass(frozen=True, slots=True)
class Comparison:
    learned: str
    baseline: str
    b: int
    c: int
    p: float


@dataclass(frozen=True, slots=True)
class DifficultyCheck:
    baselines: tuple[Summary, ...]
    stored: Summary | None
    knobs: dict[str, tuple[Any, Any]]


def run_split(config: Config, agent, split: Split, games: int) -> list[EpisodeRecord]:
    """Play the first `games` seeds of the split, in seed order, with the agent as given."""
    seeds = SPLIT_SEEDS[split]
    if type(games) is not int or games < 1 or games > len(seeds):
        raise ValueError(f"games must be between 1 and {len(seeds)} for the {split.value} split")
    if getattr(agent, "training", False):
        # Evaluation is greedy. A training agent here would learn from held-out seeds.
        raise ValueError(
            f"{agent.name} is in training mode; evaluation plays greedily, so set"
            " training = False or use a separate evaluation agent"
        )
    environment = Environment(config)
    referee = Referee(config.game.max_steps)
    return [play_episode(environment, referee, agent, seed) for seed in seeds[:games]]


def summarise(agent: str, records: Iterable[EpisodeRecord]) -> Summary:
    """Counts, 95% Wilson intervals and mean steps to a win for one agent's records."""
    records = tuple(records)
    outcomes = [record.outcome for record in records]
    games = len(outcomes)
    if games == 0:
        raise ValueError("nothing to summarise")
    wins = outcomes.count(Outcome.WIN)
    losses = outcomes.count(Outcome.LOSS)
    ties = outcomes.count(Outcome.TIE)
    steps = mean_steps(record.steps for record in records if record.outcome is Outcome.WIN)
    return Summary(
        agent, games, wins, losses, ties,
        wilson(wins, games), wilson(losses, games), wilson(ties, games), steps,
    )


def summary_of_validation(validation: ValidationSummary, agent: str = "learned") -> Summary:
    """The summary a policy file stores for its final validation run."""
    games = validation.games
    return Summary(
        agent, games, validation.wins, validation.losses, validation.ties,
        wilson(validation.wins, games), wilson(validation.losses, games),
        wilson(validation.ties, games), validation.mean_steps_to_win,
    )


def _one_agent(records: Sequence[EpisodeRecord]) -> str:
    names = {record.agent for record in records}
    if len(names) != 1:
        raise ValueError("records of exactly one agent are expected")
    return names.pop()


def _wins_by_seed(records: Sequence[EpisodeRecord]) -> dict[int, bool]:
    wins = {record.seed: record.outcome is Outcome.WIN for record in records}
    if len(wins) != len(records):
        raise ValueError("a seed appears more than once")
    return wins


def compare(learned: Sequence[EpisodeRecord], baseline: Sequence[EpisodeRecord]) -> Comparison:
    """The one-sided exact McNemar test of the learned agent against a baseline."""
    b, c = discordant_pairs(_wins_by_seed(learned), _wins_by_seed(baseline))
    return Comparison(_one_agent(learned), _one_agent(baseline), b, c, mcnemar_one_sided(b, c))


def format_header(split: Split, games: int, game: GameKnobs) -> str:
    knobs = " ".join(f"{field.name}={getattr(game, field.name)}" for field in fields(game))
    return f"split {split.value}, {games} games per agent, knobs: {knobs}"


def _cell(count: int, games: int, interval: tuple[float, float], width: int) -> str:
    low, high = interval
    return f"{count:>{width}} {100 * count / games:5.1f}% [{100 * low:4.1f},{100 * high:5.1f}]"


def format_summary_row(summary: Summary) -> str:
    """One summary-table row with fixed column widths."""
    steps = "n/a" if summary.mean_steps_to_win is None else f"{summary.mean_steps_to_win:.1f}"
    return (
        f"{summary.agent:<12}"
        + _cell(summary.wins, summary.games, summary.win_interval, 5)
        + _cell(summary.losses, summary.games, summary.loss_interval, 6)
        + _cell(summary.ties, summary.games, summary.tie_interval, 5)
        + f"   {steps}"
    )


def format_comparison(comparison: Comparison) -> str:
    label = f"{comparison.learned} vs {comparison.baseline}:"
    return f"{label:<25}b={comparison.b} c={comparison.c} one-sided exact McNemar p={comparison.p:.4f}"


def format_report(
    split: Split, games: int, game: GameKnobs,
    summaries: Iterable[Summary], comparisons: Iterable[Comparison] = (),
) -> str:
    lines = [format_header(split, games, game), TABLE_HEADER]
    lines += [format_summary_row(summary) for summary in summaries]
    lines += [format_comparison(comparison) for comparison in comparisons]
    return "\n".join(lines)


def write_eval_json(path: str | Path, config: Config, records: Iterable[EpisodeRecord]) -> None:
    """The configuration and every record, so any evaluated game can be replayed."""
    data = {
        "config": json.loads(config.to_json()),
        "records": [
            {
                "seed": record.seed, "agent": record.agent, "outcome": record.outcome.value,
                "steps": record.steps, "episode_return": record.episode_return,
            }
            for record in records
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_eval_json(path: str | Path) -> tuple[Config, list[EpisodeRecord]]:
    data = json.loads(Path(path).read_text())
    config = Config.from_json(json.dumps(data["config"]))
    records = [
        EpisodeRecord(
            item["seed"], item["agent"], Outcome(item["outcome"]), item["steps"],
            item["episode_return"],
        )
        for item in data["records"]
    ]
    return config, records


def knob_differences(current: GameKnobs, stored: GameKnobs) -> dict[str, tuple[Any, Any]]:
    """Game knobs whose current value differs from the one a policy was trained with."""
    return {
        field.name: (getattr(current, field.name), getattr(stored, field.name))
        for field in fields(current)
        if getattr(current, field.name) != getattr(stored, field.name)
    }


def difficulty_check(config: Config, policy: Policy | None = None) -> DifficultyCheck:
    """The baselines on the validation split under the given knobs; never trains."""
    baselines = tuple(
        summarise(agent.name, run_split(config, agent, Split.VAL, FULL_GAMES))
        for agent in (RandomAgent(), PathfindingAgent())
    )
    if policy is None:
        return DifficultyCheck(baselines, None, {})
    return DifficultyCheck(
        baselines, summary_of_validation(policy.validation),
        knob_differences(config.game, policy.config.game),
    )


def simulate(config: Config, agents: Iterable, split: Split, games: int) -> list[Summary]:
    """The per-agent summary of the first `games` seeds of a split, no graphics."""
    return [summarise(agent.name, run_split(config, agent, split, games)) for agent in agents]
