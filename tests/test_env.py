"""Reset, sensing, memory, turn order, collisions and rewards (FR-3, FR-4, FR-7, FR-16)."""

import ast
import subprocess
import sys

import pytest

from pacman.config import Config, GameKnobs, LearnerSettings
from pacman.env import Environment, KnownCell, Outcome
from pacman.grid import COLS, Direction
from pacman.mapgen import GameMap, Split, generate
from pacman.rng import derive, pick

EMPTY = GameMap(())
STILL_GHOST = Config(game=GameKnobs(ghost_move_prob=0.0))
UP, RIGHT, DOWN, LEFT = Direction


def known_at(observation, cell):
    row, col = cell
    return observation.known[row * COLS + col]


def place(env, pacman=None, ghost=None, visited=None):
    """Put a reset environment into a hand-built mid-game state."""
    if pacman is not None:
        env.pacman = pacman
    if ghost is not None:
        env.ghost.cell = ghost
    if visited is not None:
        env.visited = set(visited)


def test_reset_places_the_actors_and_marks_the_start_corner():
    observation = Environment(Config()).reset(0, EMPTY)
    assert observation.pacman == (0, 0)
    assert observation.ghost == (4, 4)
    assert observation.visited == frozenset({(0, 0)})


def test_reset_senses_the_start_neighbours_and_nothing_else():
    observation = Environment(Config()).reset(0, GameMap({(0, 1)}))
    assert known_at(observation, (0, 1)) is KnownCell.WALL
    assert known_at(observation, (1, 0)) is KnownCell.OPEN
    unknown = [cell for cell in observation.known if cell is KnownCell.UNKNOWN]
    assert len(unknown) == 23
    assert known_at(observation, (0, 0)) is KnownCell.UNKNOWN


def test_reset_without_a_map_generates_the_seeds_map():
    env = Environment(Config())
    env.reset(0)
    assert env.game_map == generate(0, Split.TRAIN, Config())
    env.reset(2_000_000)
    assert env.game_map == EMPTY


def test_reset_rejects_a_seed_outside_every_split_when_it_must_generate():
    with pytest.raises(ValueError):
        Environment(Config()).reset(5_000_000)


def test_a_move_senses_the_new_neighbours_and_keeps_old_knowledge():
    env = Environment(STILL_GHOST)
    env.reset(0, GameMap({(1, 1), (0, 3)}))
    result = env.step(RIGHT, 1)
    observation = result.observation
    assert observation.pacman == (0, 1)
    assert known_at(observation, (0, 0)) is KnownCell.OPEN
    assert known_at(observation, (0, 2)) is KnownCell.OPEN
    assert known_at(observation, (1, 1)) is KnownCell.WALL
    assert known_at(observation, (1, 0)) is KnownCell.OPEN     # remembered from reset
    assert known_at(observation, (0, 3)) is KnownCell.UNKNOWN  # not yet sensed


def test_a_blocked_move_leaves_pacman_in_place_and_still_costs_a_step():
    env = Environment(STILL_GHOST)
    env.reset(0, GameMap({(0, 1)}))
    result = env.step(RIGHT, 1)
    assert result.observation.pacman == (0, 0)
    assert result.reward == pytest.approx(-0.05)
    assert result.outcome is None
    result = env.step(UP, 2)
    assert result.observation.pacman == (0, 0)


def test_memory_is_rebuilt_at_reset():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    for action in (RIGHT, RIGHT, DOWN):
        env.step(action, 1)
    observation = env.reset(0, EMPTY)
    assert sum(cell is not KnownCell.UNKNOWN for cell in observation.known) == 2


def test_observations_are_snapshots():
    env = Environment(STILL_GHOST)
    first = env.reset(0, EMPTY)
    second = env.step(RIGHT, 1).observation
    assert first.pacman == (0, 0)
    assert known_at(first, (0, 2)) is KnownCell.UNKNOWN
    assert known_at(second, (0, 2)) is KnownCell.OPEN
    env.step(RIGHT, 2)
    assert second.pacman == (0, 1)


def test_no_observation_shows_a_wall_pacman_has_not_sensed():
    config = Config()
    env = Environment(config)
    rng = derive(77, "test")
    for seed in range(40):
        observation = env.reset(seed)
        walls = env.game_map.walls
        sensed = set()
        outcome = None
        while outcome is None:
            sensed.update(
                (observation.pacman[0] + d_row, observation.pacman[1] + d_col)
                for d_row, d_col in ((-1, 0), (0, 1), (1, 0), (0, -1))
            )
            for index, memory in enumerate(observation.known):
                cell = divmod(index, COLS)
                if memory is KnownCell.WALL:
                    assert cell in walls and cell in sensed
                elif memory is KnownCell.OPEN:
                    assert cell not in walls and cell in sensed
            result = env.step(Direction(pick(rng, 4)), 1)
            observation, outcome = result.observation, result.outcome
            if len(sensed) > 200:  # a tie is the referee's business; stop ourselves
                break


def test_every_step_pays_the_step_cost():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    assert env.step(DOWN, 1).reward == pytest.approx(-0.05)


def test_first_corner_visit_pays_once():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    rewards = [env.step(RIGHT, t).reward for t in range(1, 5)]
    assert rewards[:3] == pytest.approx([-0.05] * 3)
    assert rewards[3] == pytest.approx(0.95)
    assert env.step(LEFT, 5).reward == pytest.approx(-0.05)
    assert env.step(RIGHT, 6).reward == pytest.approx(-0.05)


def test_the_start_corner_pays_nothing_on_return():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    env.step(RIGHT, 1)
    assert env.step(LEFT, 2).reward == pytest.approx(-0.05)


def test_pacman_stepping_onto_the_ghost_loses_before_the_corner_counts():
    env = Environment(Config())
    env.reset(0, EMPTY)
    place(env, pacman=(4, 3), ghost=(4, 4), visited={(0, 0), (0, 4), (4, 0)})
    result = env.step(RIGHT, 9)
    assert result.outcome is Outcome.LOSS
    assert result.reward == pytest.approx(-100.05)
    assert result.observation.pacman == result.observation.ghost == (4, 4)
    assert (4, 4) not in result.observation.visited
    assert env.ghost.last_visit == {(4, 4): 0}  # the ghost never moved


def test_the_last_corner_wins_before_the_ghost_moves():
    env = Environment(Config())
    env.reset(0, EMPTY)
    place(env, pacman=(4, 3), ghost=(3, 4), visited={(0, 0), (0, 4), (4, 0)})
    result = env.step(RIGHT, 9)
    assert result.outcome is Outcome.WIN
    assert result.reward == pytest.approx(100.95)
    assert result.observation.ghost == (3, 4)
    assert result.observation.visited == frozenset({(0, 0), (0, 4), (4, 0), (4, 4)})


def test_the_ghosts_move_onto_pacman_loses():
    env = Environment(Config())
    env.reset(0, EMPTY)
    place(env, ghost=(0, 1))
    result = env.step(UP, 1)  # off the grid: Pac-Man stays at (0, 0)
    assert result.outcome is Outcome.LOSS
    assert result.reward == pytest.approx(-100.05)
    assert result.observation.ghost == (0, 0)


def test_a_corner_visit_and_a_loss_on_the_same_step_both_pay():
    env = Environment(Config())
    env.reset(0, EMPTY)
    place(env, pacman=(0, 3), ghost=(1, 4))
    result = env.step(RIGHT, 1)
    assert result.outcome is Outcome.LOSS
    assert result.reward == pytest.approx(-99.05)
    assert (0, 4) in result.observation.visited


def test_sensing_happens_before_the_ghost_moves_and_after_a_win_it_does_not():
    env = Environment(Config())
    env.reset(0, GameMap({(3, 3)}))
    place(env, pacman=(4, 3), ghost=(3, 4), visited={(0, 0), (0, 4), (4, 0)})
    result = env.step(RIGHT, 9)
    assert result.outcome is Outcome.WIN
    assert known_at(result.observation, (3, 4)) is KnownCell.UNKNOWN


def test_the_ghost_moves_with_the_configured_probability():
    still = Environment(STILL_GHOST)
    still.reset(0, EMPTY)
    for t in range(1, 21):
        assert still.step(UP, t).observation.ghost == (4, 4)
    moving = Environment(Config())
    moving.reset(0, EMPTY)
    assert moving.step(UP, 1).observation.ghost != (4, 4)


def test_the_ghost_is_seeded_by_the_episode_and_stamps_visits_with_t():
    env = Environment(Config())
    env.reset(3, EMPTY)
    env.step(UP, 7)
    assert env.ghost.last_visit[env.ghost.cell] == 7
    again = Environment(Config())
    again.reset(3, EMPTY)
    assert again.step(UP, 7).observation.ghost == env.ghost.cell


def test_the_environment_never_calls_a_tie_and_ignores_t_otherwise():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    for t in (0, 1, 100, 101, 10_000, -5):
        assert env.step(UP, t).outcome is None


def test_step_after_the_episode_ends_is_an_error():
    env = Environment(Config())
    env.reset(0, EMPTY)
    place(env, ghost=(0, 1))
    assert env.step(UP, 1).outcome is Outcome.LOSS
    with pytest.raises(RuntimeError):
        env.step(UP, 2)


def test_step_before_reset_is_an_error():
    with pytest.raises(RuntimeError, match="reset"):
        Environment(Config()).step(UP, 1)


def test_reward_totals_for_a_full_win():
    env = Environment(STILL_GHOST)
    env.reset(0, EMPTY)
    place(env, ghost=(2, 2))
    path = [RIGHT] * 4 + [DOWN] * 4 + [LEFT] * 4
    results = [env.step(action, t) for t, action in enumerate(path, start=1)]
    assert [result.outcome for result in results[:-1]] == [None] * 11
    assert results[-1].outcome is Outcome.WIN
    assert sum(result.reward for result in results) == pytest.approx(12 * -0.05 + 3 + 100)


def test_rewards_come_from_the_learner_settings():
    config = Config(
        game=GameKnobs(ghost_move_prob=0.0),
        learner=LearnerSettings(reward_step=-1.0, reward_corner=5.0, reward_win=50.0),
    )
    env = Environment(config)
    env.reset(0, EMPTY)
    place(env, pacman=(4, 3), ghost=(2, 2), visited={(0, 0), (0, 4), (4, 0)})
    assert env.step(RIGHT, 9).reward == pytest.approx(54.0)


def test_env_imports_only_the_game_modules():
    code = (
        "import sys, pacman.env;"
        " print(sorted(m for m in sys.modules if m.startswith('pacman')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    loaded = ast.literal_eval(result.stdout)
    assert set(loaded) <= {
        "pacman", "pacman.config", "pacman.rng", "pacman.grid", "pacman.mapgen",
        "pacman.ghost", "pacman.env",
    }
