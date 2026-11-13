"""Q-learning training: seeds, the epsilon schedule, the learning curve and outputs."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy

from pacman import evaluate
from pacman.agents.learned import LearnedAgent
from pacman.config import Config, LearnerSettings
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.evaluate import Summary, run_split, summarise
from pacman.mapgen import SPLIT_SEEDS, Split
from pacman.policy_io import Policy, ValidationSummary, save_policy
from pacman.referee import Referee
from pacman.winnable import winning_moves

# Evaluation sizes control how much is measured, not what is learned.
CURVE_EVERY = 2_500
CURVE_GAMES = 200


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: Policy
    curve: tuple[tuple[int, float], ...]
    validation: Summary
    kept_episodes: int  # episodes behind the saved table: all of them, or the best curve point's


def epsilon_for(episode: int, settings: LearnerSettings) -> float:
    """Linear decay from epsilon_start to epsilon_end over the decay fraction, then held."""
    decay_episodes = settings.episodes * settings.epsilon_decay_fraction
    fall = (settings.epsilon_start - settings.epsilon_end) * episode / decay_episodes
    return max(settings.epsilon_end, settings.epsilon_start - fall)


def learning_rate_for(episode: int, settings: LearnerSettings) -> float:
    """Linear decay from learning_rate to its final fraction over the whole run; constant by default."""
    if settings.final_learning_rate_fraction == 1.0:
        return settings.learning_rate
    share = 1.0 - (1.0 - settings.final_learning_rate_fraction) * episode / settings.episodes
    return settings.learning_rate * share


def training_seeds(config: Config) -> Iterator[int]:
    """Training seeds in order; with training_maps="winnable", only those a move sequence can win."""
    for seed in SPLIT_SEEDS[Split.TRAIN]:
        if config.learner.training_maps == "all" or winning_moves(config, seed) is not None:
            yield seed


def train(
    config: Config, on_progress: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    """Train on the training seeds, record the curve, and validate the table that is kept.

    By default the table after the last episode is kept; with keep_table="best" it is the
    table at the curve point with the highest win rate, the earliest if several tie.
    """
    settings = config.learner
    learner = LearnedAgent(settings, training=True)
    # The curve plays greedily on the learner's own table without writing to it.
    evaluator = LearnedAgent(settings, q=learner.q, training=False)
    environment = Environment(config)
    referee = Referee(config.game.max_steps)
    curve: list[tuple[int, float]] = []
    seeds = training_seeds(config)
    first_trained = last_trained = -1
    best: tuple[float, int, numpy.ndarray] | None = None  # (win rate, episodes, table)
    for episode in range(settings.episodes):
        seed = next(seeds, None)
        if seed is None:
            raise ValueError(
                f"the training split has only {episode} winnable training maps"
                f" for {settings.episodes} episodes"
            )
        if episode == 0:
            first_trained = seed
        last_trained = seed
        learner.epsilon = epsilon_for(episode, settings)
        learner.learning_rate = learning_rate_for(episode, settings)
        play_episode(environment, referee, learner, seed)
        completed = episode + 1
        if completed % CURVE_EVERY == 0:
            summary = summarise("learned", run_split(config, evaluator, Split.VAL, CURVE_GAMES))
            point = (completed, summary.wins / summary.games)
            curve.append(point)
            if settings.keep_table == "best" and (best is None or point[1] > best[0]):
                best = (point[1], completed, learner.q.copy())
            if on_progress is not None:
                on_progress(*point)
    table, kept = (best[2], best[1]) if best is not None else (learner.q, settings.episodes)
    games = evaluate.FULL_GAMES
    final = LearnedAgent(settings, q=table, training=False)
    validation = summarise("learned", run_split(config, final, Split.VAL, games))
    first_seed = SPLIT_SEEDS[Split.VAL].start
    policy = Policy(
        q=table,
        config=config,
        training_seeds=(first_trained, last_trained),
        learning_curve=tuple(curve),
        validation=ValidationSummary(
            seeds=(first_seed, first_seed + games - 1),
            games=validation.games, wins=validation.wins, losses=validation.losses,
            ties=validation.ties, mean_steps_to_win=validation.mean_steps_to_win,
        ),
    )
    return TrainingResult(policy, tuple(curve), validation, kept)


def prepare_output_dir(out_dir: str | Path) -> Path:
    """Create the run directory; refuse one that already holds a policy."""
    out = Path(out_dir)
    if (out / "policy.npz").exists():
        raise FileExistsError(f"{out / 'policy.npz'} already exists; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_outputs(out_dir: str | Path, result: TrainingResult) -> Path:
    """Write policy.npz, curve.csv, curve.png and config.json."""
    out = prepare_output_dir(out_dir)
    save_policy(out / "policy.npz", result.policy)  # first: the run must survive a plotting failure
    from pacman.plot import save_curve  # Matplotlib is loaded only when a curve is drawn

    (out / "curve.csv").write_text(
        "episodes,win_rate\n" + "".join(f"{episodes},{rate}\n" for episodes, rate in result.curve)
    )
    save_curve(out / "curve.png", result.curve)
    (out / "config.json").write_text(result.policy.config.to_json())
    return out
