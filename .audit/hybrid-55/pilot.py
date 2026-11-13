import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pacman.hybrid.game import Game
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import BaselinePac, PacActor, QTable
from pacman.hybrid.models import Rules
from pacman.rng import derive


def play(seed, key, pac):
    game = Game(Rules(), GhostController(Rules()))
    obs = game.reset(seed, key)
    pac.reset(derive(key, "hybrid-agent"))
    while True:
        result = game.step(pac.act(obs))
        pac.observe(result)
        if result.outcome is not None:
            return result.outcome.value
        obs = result.observation


table = QTable()
pac = PacActor(table, training=True)
started = time.monotonic()
for episode in range(5000):
    pac.epsilon = max(0.05, 1 - 0.95 * episode / 2500)
    play(episode, episode, pac)
    if (episode + 1) % 1000 == 0:
        print(json.dumps({"episodes": episode + 1, "rows": len(table.rows)}), flush=True)
rows = []
for name, actor in [("pathfinding", BaselinePac()), ("learned", PacActor(table.freeze()))]:
    counts = {"win": 0, "loss": 0, "tie": 0}
    for seed in range(1000000, 1000200):
        counts[play(seed, seed, actor)] += 1
    rows.append({"agent": name, **counts})
print(json.dumps({"all_games": 200, "results": rows, "seconds": time.monotonic() - started}))
