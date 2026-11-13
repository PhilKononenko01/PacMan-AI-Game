"""Opt-in two-pellet experiment: python -m pacman.hybrid."""

import argparse
import json
import math
from pathlib import Path
import time

from pacman.env import KnownCell
from pacman.hybrid.artifacts import load_policy
from pacman.hybrid.experiment import evaluate, run_episode, train_run, write_json
from pacman.hybrid.ghost import GhostController
from pacman.hybrid.learning import BaselinePac, PacActor
from pacman.hybrid.models import Rules
from pacman.mapgen import split_of_seed


def _parser():
    parser = argparse.ArgumentParser(description="Two-pellet Pac-Man versus a sight-memory search ghost")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train baseline, ghost, then adapted Pac-Man; validation only")
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--replicate", type=int, default=0)
    train.add_argument("--base-episodes", type=int, default=30_000)
    train.add_argument("--ghost-episodes", type=int, default=30_000)
    train.add_argument("--adapt-episodes", type=int, default=50_000)
    train.add_argument("--validation-games", type=int, default=200)
    train.add_argument("--games", type=int, default=1000, help="Final validation layout seed slots (1..1000)")
    train.add_argument("--repeats", type=int, default=1, help="Episode keys per layout seed (1..10)")
    for name in ("evaluate", "watch"):
        command = commands.add_parser(name)
        pac = command.add_mutually_exclusive_group(required=True)
        pac.add_argument("--pac-policy", type=Path)
        pac.add_argument("--pac-baseline", choices=("random", "bfs"))
        command.add_argument("--ghost-policy", type=Path, help="Omit to use the fixed sight-memory BFS ghost")
        if name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), default="val")
            command.add_argument("--games", type=int, default=1000, help="First N split layout seed slots")
            command.add_argument("--repeats", type=int, default=1)
            command.add_argument("--replicate", type=int, default=0)
            command.add_argument("--out", type=Path, help="Write full JSON report; existing files are refused")
        else:
            command.add_argument("--seed", type=int, default=1_000_000)
            command.add_argument("--key", type=int, default=0)
            command.add_argument("--delay", type=float, default=0.15)
            command.add_argument("--plain", action="store_true", help="Print frames without clearing the terminal")
    return parser


def _actors(args):
    pac_policy = load_policy(args.pac_policy, expected_role="pacman") if args.pac_policy else None
    ghost_policy = load_policy(args.ghost_policy, expected_role="ghost",
                               rules=pac_policy.rules if pac_policy else None) if args.ghost_policy else None
    rules = pac_policy.rules if pac_policy else ghost_policy.rules if ghost_policy else Rules()
    pac = PacActor(pac_policy.table) if pac_policy else BaselinePac(
        "pathfinding" if args.pac_baseline == "bfs" else "random")
    ghost = GhostController(rules, ghost_policy.table if ghost_policy else None)
    return rules, pac, ghost


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            report = train_run(args.out, args.replicate, base_episodes=args.base_episodes,
                               ghost_episodes=args.ghost_episodes, adapt_episodes=args.adapt_episodes,
                               validation_games=args.validation_games, finaleval_games=args.games,
                               repeats=args.repeats)
            print(json.dumps({"out": str(args.out), "replicate": args.replicate,
                              "stages": report["stages"]}, indent=2))
        else:
            if args.command == "evaluate" and args.out is not None and args.out.exists():
                raise FileExistsError(f"refusing to overwrite {args.out}")
            if args.command == "watch":
                split_of_seed(args.seed)
                if not math.isfinite(args.delay) or not 0 <= args.delay <= 10:
                    raise ValueError("delay must be a finite number in 0..10")
                if not 0 <= args.key < 2**63:
                    raise ValueError("key must be in 0..2**63-1")
            rules, pac, ghost = _actors(args)
            if args.command == "evaluate":
                report = evaluate(pac, ghost, rules, split=args.split, games=args.games,
                                  repeats=args.repeats, replicate=args.replicate)
                if args.out is not None:
                    write_json(args.out, report)
                print(json.dumps(report["summary"], indent=2, sort_keys=True))
            else:
                def draw(game, result):
                    observation = game.observation()
                    if not args.plain:
                        print("\033[2J\033[H", end="")
                    for row in range(5):
                        cells = []
                        for col in range(5):
                            cell = row, col
                            known = observation.known[row * 5 + col]
                            glyph = "#" if known == KnownCell.WALL else "." if known == KnownCell.OPEN else "?"
                            if cell in observation.pellets:
                                glyph = "o"
                            if cell == observation.ghost:
                                glyph = "G"
                            if cell == observation.pacman:
                                glyph = "X" if cell == observation.ghost else "P"
                            cells.append(glyph)
                        print(" ".join(cells))
                    outcome = game.outcome.value if game.outcome is not None else "playing"
                    print(f"turn={game.turn}/{rules.max_steps} pellets={len(observation.pellets)}/2 "
                          f"ghost={ghost.mode} outcome={outcome}", flush=True)
                    if args.delay:
                        time.sleep(args.delay)
                record = run_episode(args.seed, args.key, rules, pac, ghost, draw)
                print(json.dumps(record, sort_keys=True))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
