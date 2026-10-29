"""Q-learning training: seeds, the epsilon schedule, the learning curve and outputs (§11)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pacman import evaluate
from pacman.agents.learned import LearnedAgent
from pacman.config import Config, LearnerSettings
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.evaluate import Summary, run_split, summarise
from pacman.mapgen import SPLIT_SEEDS, Split
from pacman.policy_io import Policy, ValidationSummary, save_policy
from pacman.referee import Referee

# PRD §2.8 evaluation sizes: how much is measured, not what is learned (§11.1).
CURVE_EVERY = 2_500
CURVE_GAMES = 200


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: Policy
    curve: tuple[tuple[int, float], ...]
    validation: Summary


def epsilon_for(episode: int, settings: LearnerSettings) -> float:
    """Linear decay from epsilon_start to epsilon_end over the decay fraction, then held."""
    decay_episodes = settings.episodes * settings.epsilon_decay_fraction
    fall = (settings.epsilon_start - settings.epsilon_end) * episode / decay_episodes
    return max(settings.epsilon_end, settings.epsilon_start - fall)


def train(
    config: Config, on_progress: Callable[[int, float], None] | None = None,
) -> TrainingResult:
    """Train on seeds 0 .. episodes - 1 and validate the final table; no checkpoint is picked."""
    settings = config.learner
    learner = LearnedAgent(settings, training=True)
    # The curve and the final validation play greedily on the same table without writing to it.
    evaluator = LearnedAgent(settings, q=learner.q, training=False)
    environment = Environment(config)
    referee = Referee(config.game.max_steps)
    curve: list[tuple[int, float]] = []
    for episode in range(settings.episodes):
        learner.epsilon = epsilon_for(episode, settings)
        play_episode(environment, referee, learner, episode)
        completed = episode + 1
        if completed % CURVE_EVERY == 0:
            summary = summarise("learned", run_split(config, evaluator, Split.VAL, CURVE_GAMES))
            point = (completed, summary.wins / summary.games)
            curve.append(point)
            if on_progress is not None:
                on_progress(*point)
    games = evaluate.FULL_GAMES
    validation = summarise("learned", run_split(config, evaluator, Split.VAL, games))
    first_seed = SPLIT_SEEDS[Split.VAL].start
    policy = Policy(
        q=learner.q,
        config=config,
        training_seeds=(0, settings.episodes - 1),
        learning_curve=tuple(curve),
        validation=ValidationSummary(
            seeds=(first_seed, first_seed + games - 1),
            games=validation.games, wins=validation.wins, losses=validation.losses,
            ties=validation.ties, mean_steps_to_win=validation.mean_steps_to_win,
        ),
    )
    return TrainingResult(policy, tuple(curve), validation)


def prepare_output_dir(out_dir: str | Path) -> Path:
    """Create the run directory; refuse one that already holds a policy (§14.1)."""
    out = Path(out_dir)
    if (out / "policy.npz").exists():
        raise FileExistsError(f"{out / 'policy.npz'} already exists; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_outputs(out_dir: str | Path, result: TrainingResult) -> Path:
    """Write policy.npz, curve.csv, curve.png and config.json (§11.2)."""
    out = prepare_output_dir(out_dir)
    save_policy(out / "policy.npz", result.policy)  # first: the run must survive a plotting failure
    from pacman.plot import save_curve  # Matplotlib is loaded only when a curve is drawn

    (out / "curve.csv").write_text(
        "episodes,win_rate\n" + "".join(f"{episodes},{rate}\n" for episodes, rate in result.curve)
    )
    save_curve(out / "curve.png", result.curve)
    (out / "config.json").write_text(result.policy.config.to_json())
    return out
