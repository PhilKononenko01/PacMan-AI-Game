import random

import pytest

from pacman.env import KnownCell, Outcome
from pacman.grid import Direction, move
from pacman.hybrid.game import Game
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import PacActor, QTable, FrozenTable, pac_state
from pacman.hybrid.models import GhostObservation, Rules
from pacman.mapgen import GameMap


EMPTY = GameMap(())


def test_rule_limits_match_saved_search_feature_schema():
    with pytest.raises(ValueError, match="at most 3"):
        Rules(search_choices=4)


def test_game_rejects_ghost_with_different_rules():
    with pytest.raises(ValueError, match="must match"):
        Game(Rules(), GhostController(Rules(max_steps=10)))


def make_game(rules=Rules(), ghost=None):
    ghost = ghost if ghost is not None else GhostController(rules)
    game = Game(rules, ghost)
    game.reset(0, 42, EMPTY)
    return game


def test_two_pellets_are_open_distinct_repeatable_and_exclude_both_spawns():
    placements = set()
    for key in range(30):
        game = Game(Rules(), GhostController(Rules()))
        obs = game.reset(0, key, GameMap({(2, 2), (1, 1)}))
        assert len(obs.pellets) == 2
        assert not obs.pellets.intersection({(0, 0), (4, 4), (2, 2), (1, 1)})
        assert obs.pacman == (0, 0) and obs.ghost == (4, 4)
        assert obs.pellets == game.reset(0, key, game.game_map).pellets
        placements.add(obs.pellets)
    assert len(placements) > 1


def test_pacman_knows_positions_but_only_senses_adjacent_walls():
    game = make_game()
    obs = game.observation()
    assert obs.ghost == (4, 4)
    assert obs.known[0] is KnownCell.OPEN
    assert sum(c is not KnownCell.UNKNOWN for c in obs.known) == 3
    assert obs.known[24] is KnownCell.UNKNOWN


def test_hidden_position_does_not_change_ghost_observation_or_action():
    ghosts = [GhostController(Rules()), GhostController(Rules())]
    games = [make_game(ghost=g) for g in ghosts]
    for game, pacman in zip(games, [(0, 0), (1, 1)]):
        game.pacman = pacman
    observations = [g.ghost_observation() for g in games]
    assert observations[0] == observations[1]
    assert observations[0].visible_pacman is None
    assert ghosts[0].act(observations[0]) == ghosts[1].act(observations[1])
    assert ghosts[0].last_seen is None and ghosts[1].last_seen is None


def test_hidden_movement_does_not_create_distance_rewards_for_ghost():
    results = []
    for cell in ((0, 0), (1, 1)):
        game = make_game()
        game.pacman = cell
        game.pellets = {(0, 3), (0, 4)}
        results.append(game.step(Direction.RIGHT))
    assert all(r.outcome is None and r.ghost_reward == 0.0 for r in results)


@pytest.mark.parametrize("pacman,walls,visible", [
    ((4, 0), (), True), ((0, 4), (), True),
    ((4, 0), ((4, 2),), False), ((0, 4), ((2, 4),), False),
    ((0, 0), (), False),
])
def test_visibility_requires_clear_row_or_column(pacman, walls, visible):
    game = make_game()
    game.game_map = GameMap(walls)
    game.pacman = pacman
    assert (game.ghost_observation().visible_pacman is not None) is visible


def test_direction_comes_only_from_consecutive_visible_observations():
    ghost = GhostController(Rules())
    ghost.reset(EMPTY, random.Random(3))
    ghost.act(GhostObservation((0, 0), (0, 2), 99))
    assert ghost.heading is None
    ghost.act(GhostObservation((0, 0), (0, 3), 98))
    assert ghost.heading == Direction.RIGHT
    ghost.act(GhostObservation((1, 0), None, 97))
    assert ghost.heading == Direction.RIGHT
    ghost.act(GhostObservation((0, 0), (0, 4), 96))
    assert ghost.heading is None


def test_lost_sighting_uses_bfs_around_wall_to_old_position():
    ghost = GhostController(Rules())
    game_map = GameMap({(2, 2)})
    ghost.reset(game_map, random.Random(1))
    ghost.last_seen = (2, 3)
    action = ghost.act(GhostObservation((2, 1), None, 90))
    assert ghost.mode == "pursue"
    assert move((2, 1), action, game_map.is_open) == (1, 1)
    assert ghost.last_seen == (2, 3)


def test_hidden_memory_expires_and_returns_to_wandering():
    rules = Rules(memory_turns=2)
    ghost = GhostController(rules)
    ghost.reset(EMPTY, random.Random(1))
    ghost.last_seen = (0, 4)
    ghost.act(GhostObservation((4, 0), None, 90))
    ghost.act(GhostObservation((3, 0), None, 89))
    assert ghost.mode == "wander" and ghost.last_seen is None


def test_baseline_and_zero_learning_policy_have_same_nonlearning_behavior():
    ghosts = [GhostController(Rules()), GhostController(Rules(), FrozenTable({}))]
    for ghost in ghosts:
        ghost.reset(EMPTY, random.Random(42))
    observations = [GhostObservation((4, 4), None, 99),
                    GhostObservation((4, 3), (4, 0), 98),
                    GhostObservation((4, 2), None, 97),
                    GhostObservation((4, 0), None, 96)]
    for obs in observations:
        assert ghosts[0].act(obs) == ghosts[1].act(obs)


class StillGhost:
    def reset(self, game_map, rng):
        self.rewards = []

    def act(self, obs):
        return Direction.RIGHT

    def reward(self, value, terminal):
        self.rewards.append((value, terminal))


def test_collect_once_and_win_before_ghost_reply():
    ghost = StillGhost()
    game = make_game(ghost=ghost)
    game.pellets = {(0, 1), (0, 2)}
    first = game.step(Direction.RIGHT)
    assert first.pac_reward == pytest.approx(0.95)
    game.step(Direction.LEFT)
    revisit = game.step(Direction.RIGHT)
    assert revisit.pac_reward == -0.05
    final = game.step(Direction.RIGHT)
    assert final.outcome is Outcome.WIN
    assert final.pac_reward == pytest.approx(100.95)
    assert ghost.rewards.count((-1.0, True)) == 1
    with pytest.raises(RuntimeError):
        game.step(Direction.RIGHT)


def test_capture_on_pacman_move_notifies_ghost_even_without_ghost_action():
    ghost = StillGhost()
    game = make_game(ghost=ghost)
    game.pacman = (4, 3)
    game.pellets = {(4, 4)}
    result = game.step(Direction.RIGHT)
    assert result.outcome is Outcome.LOSS
    assert ghost.rewards == [(1.0, True)]


def test_capture_on_ghost_move():
    game = make_game()
    game.pacman, game.ghost_cell = (4, 1), (4, 3)
    game.pellets = {(0, 1), (0, 2)}
    result = game.step(Direction.RIGHT)
    assert result.outcome is Outcome.LOSS
    assert result.ghost_reward == 1.0


def test_timeout_has_terminal_failure_reward_and_zero_ghost_success():
    rules = Rules(max_steps=1)
    ghost = StillGhost()
    game = make_game(rules, ghost)
    game.pellets = {(2, 2), (2, 3)}
    result = game.step(Direction.RIGHT)
    assert result.outcome is Outcome.TIE
    assert result.pac_reward == -100.05
    assert result.ghost_reward == 0.0
    assert ghost.rewards == [(0.0, True)]


def test_search_credit_survives_reacquisition_and_discounts_elapsed_game_turns():
    table = QTable()
    ghost = GhostController(Rules(), table, training=True, alpha=1.0, gamma=0.5)
    ghost.reset(EMPTY, random.Random(42))
    ghost.last_seen = (2, 2)
    ghost.act(GhostObservation((2, 2), None, 95))
    state, action, _, _ = ghost.pending
    ghost.reward(0.0, False)
    ghost.act(GhostObservation((2, 3), (2, 4), 94))
    assert ghost.pending is not None
    ghost.reward(1.0, True)
    assert table.values(state)[action] == 0.5
    assert table.updates == 1 and ghost.pending is None
    with pytest.raises(RuntimeError):
        ghost.reward(1.0, True)


def test_pending_search_gets_capture_credit_when_pacman_walks_into_ghost():
    table = QTable()
    ghost = GhostController(Rules(), table, training=True, alpha=1.0)
    game = make_game(ghost=ghost)
    ghost.last_seen = (2, 2)
    ghost.act(GhostObservation((2, 2), None, 90))
    state, action, _, _ = ghost.pending
    game.pacman, game.ghost_cell = (4, 3), (4, 4)
    result = game.step(Direction.RIGHT)
    assert result.outcome is Outcome.LOSS
    assert table.values(state)[action] == 1.0
    assert table.updates == 1


def test_search_budget_ends_in_wandering_without_more_choices():
    ghost = GhostController(Rules(memory_turns=100))
    ghost.reset(EMPTY, random.Random(7))
    ghost.last_seen = (2, 2)
    cell = (2, 2)
    for turn in range(20):
        action = ghost.act(GhostObservation(cell, None, 99 - turn))
        cell = move(cell, action, EMPTY.is_open)
        if ghost.mode == "wander":
            break
    assert ghost.mode == "wander" and ghost.last_seen is None
    assert ghost.search_decisions == 3


def test_exhausted_search_targets_fall_back_to_legal_wandering():
    ghost = GhostController(Rules())
    ghost.reset(EMPTY, random.Random(7))
    ghost.last_seen = (2, 2)
    ghost.tried.update(ghost.candidates((2, 2)).values())
    action = ghost.act(GhostObservation((2, 2), None, 90))
    assert ghost.mode == "wander"
    assert move((2, 2), action, EMPTY.is_open) != (2, 2)


def test_pacman_timeout_update_does_not_bootstrap():
    game = make_game(Rules(max_steps=1), StillGhost())
    game.pellets = {(2, 2), (2, 3)}
    table = QTable()
    pac = PacActor(table, training=True, alpha=1.0)
    pac.reset(random.Random(1))
    action = pac.act(game.observation())
    state, selected = pac.pending
    table.rows[state] = [500.0] * 4
    result = game.step(action)
    pac.observe(result)
    assert table.values(state)[selected] == pytest.approx(result.pac_reward)
    assert table.updates == 1


def test_pacman_greedy_choice_and_bootstrap_ignore_illegal_actions():
    game = make_game(ghost=StillGhost())
    game.pellets = {(2, 2), (2, 3)}
    table = QTable()
    pac = PacActor(table, training=True, alpha=1.0)
    pac.reset(random.Random(1))
    state, legal = pac_state(game.observation())
    table.rows[state] = [1e9, 2.0, 1.0, 1e9]
    action = pac.act(game.observation())
    assert action is Direction.RIGHT
    result = game.step(action)
    next_state, next_legal = pac_state(result.observation)
    table.rows[next_state] = [1e9 if d not in next_legal else 4.0 for d in range(4)]
    pac.observe(result)
    assert table.values(state)[action] == pytest.approx(result.pac_reward + 4.0)


def test_search_bootstrap_uses_elapsed_discount_and_only_legal_candidates():
    table = QTable()
    ghost = GhostController(Rules(), table, training=True, alpha=1.0, gamma=0.5)
    ghost.reset(EMPTY, random.Random(7))
    ghost.last_seen = (2, 2)
    ghost.act(GhostObservation((2, 2), None, 95))
    state, action, _, _ = ghost.pending
    ghost.reward(0.0, False)
    ghost.reward(0.0, False)
    ghost.target = (0, 0)
    ghost.moves_left = 0
    obs = GhostObservation((0, 0), None, 94)
    targets = ghost.candidates(obs.cell)
    next_state = (0, 12, 4, sum(1 << d for d in targets), 0, 1, 3, 0)
    table.rows[next_state] = [4.0 if d in targets else 1e9 for d in range(4)]
    ghost.act(obs)
    assert table.values(state)[action] == 1.0


def test_frozen_table_is_detached_and_cannot_be_changed():
    table = QTable()
    table.update((0,), 0, 10, 1)
    frozen = table.freeze()
    digest = frozen.digest()
    table.update((0,), 0, -10, 1)
    assert frozen.values((0,))[0] == 10
    assert frozen.digest() == digest
    with pytest.raises(TypeError):
        frozen.rows[(0,)] = (0.0,) * 4
