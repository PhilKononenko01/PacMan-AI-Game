from pacman.config import Config
from pacman.env import KnownCell, Outcome
from pacman.grid import Direction, line_of_sight, move, neighbours
from pacman.hybrid.models import PAC_START, GHOST_START, GhostObservation, PacObservation, Rules, TurnResult
from pacman.mapgen import generate, split_of_seed
from pacman.rng import derive, pick


class Game:
    def __init__(self, rules: Rules, ghost):
        if hasattr(ghost, "rules") and ghost.rules != rules:
            raise ValueError("ghost and game rules must match")
        self.rules, self.ghost = rules, ghost

    def reset(self, layout_seed: int, episode_key: int, game_map=None):
        self.game_map = game_map if game_map is not None else generate(layout_seed, split_of_seed(layout_seed), Config())
        self.pacman, self.ghost_cell = PAC_START, GHOST_START
        candidates = [(r, c) for r in range(5) for c in range(5)
                      if (r, c) not in (PAC_START, GHOST_START) and self.game_map.is_open((r, c))]
        rng = derive(episode_key, "hybrid-pellets")
        self.pellets = {candidates.pop(pick(rng, len(candidates))) for _ in range(2)}
        self.initial_pellets = tuple(sorted(self.pellets))
        self.known = [KnownCell.UNKNOWN] * 25
        self.turn = 0
        self.outcome = None
        self.ghost.reset(self.game_map, derive(episode_key, "hybrid-ghost"))
        self._sense()
        return self.observation()

    def _sense(self):
        for r, c in (self.pacman, *neighbours(self.pacman)):
            self.known[r * 5 + c] = KnownCell.OPEN if self.game_map.is_open((r, c)) else KnownCell.WALL

    def observation(self):
        return PacObservation(self.pacman, self.ghost_cell, frozenset(self.pellets),
                              tuple(self.known), self.rules.max_steps - self.turn)

    def ghost_observation(self):
        visible = self.pacman if line_of_sight(self.ghost_cell, self.pacman, self.game_map.is_open) else None
        return GhostObservation(self.ghost_cell, visible, self.rules.max_steps - self.turn)

    def step(self, action: Direction):
        if self.outcome is not None:
            raise RuntimeError("the game has already ended")
        self.turn += 1
        reward = -0.05
        self.pacman = move(self.pacman, action, self.game_map.is_open)
        self._sense()
        if self.pacman == self.ghost_cell:
            return self._finish(Outcome.LOSS, reward - 100.0)
        if self.pacman in self.pellets:
            self.pellets.remove(self.pacman)
            reward += 1.0
            if not self.pellets:
                return self._finish(Outcome.WIN, reward + 100.0)
        action = self.ghost.act(self.ghost_observation())
        self.ghost_cell = move(self.ghost_cell, action, self.game_map.is_open)
        if self.pacman == self.ghost_cell:
            return self._finish(Outcome.LOSS, reward - 100.0)
        if self.turn >= self.rules.max_steps:
            return self._finish(Outcome.TIE, reward - 100.0)
        self.ghost.reward(0.0, False)
        return TurnResult(self.observation(), reward, 0.0, None)

    def _finish(self, outcome, reward):
        self.outcome = outcome
        ghost_reward = 1.0 if outcome is Outcome.LOSS else -1.0 if outcome is Outcome.WIN else 0.0
        self.ghost.reward(ghost_reward, True)
        return TurnResult(self.observation(), reward, ghost_reward, outcome)
