"""The five commands: train, evaluate, check, simulate and watch (§14).

Only this module parses arguments or prints; the services return data and strings.
Every error ends the command with a message on standard error and a non-zero exit.
"""

import argparse
from dataclasses import fields, replace
import os
from pathlib import Path
import sys
import time

from pacman import evaluate
from pacman import train as training
from pacman.agents import AGENT_NAMES, make_agent
from pacman.config import Config
from pacman.env import Environment
from pacman.episode import play_episode
from pacman.evaluate import (
    TABLE_HEADER,
    compare,
    difficulty_check,
    format_header,
    format_report,
    format_summary_row,
    knob_differences,
    run_split,
    summarise,
    write_eval_json,
)
from pacman.mapgen import Split, split_of_seed
from pacman.policy_io import Policy, load_policy
from pacman.referee import Referee
from pacman.render import render_frame, use_plain

CLEAR_SCREEN = "\x1b[H\x1b[2J"  # cursor home, then clear


def apply_sets(config: Config, sets: list[str]) -> Config:
    """Apply `--set name=value` overrides by bare field name, with type-checked values."""
    for setting in sets:
        name, separator, raw = setting.partition("=")
        if not separator or not name:
            raise ValueError(f"--set needs NAME=VALUE, got {setting!r}")
        config = _apply_set(config, name, raw)
    return config


def _apply_set(config: Config, name: str, raw: str) -> Config:
    for section in ("game", "learner"):
        part = getattr(config, section)
        for field in fields(part):
            if field.name == name:
                value = _parse_value(field.type, raw, name)
                return replace(config, **{section: replace(part, **{name: value})})
    raise ValueError(f"unknown setting {name!r}")


def _parse_value(kind, raw: str, name: str):
    try:
        if kind is int:
            return int(raw)
        if kind is float:
            return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be {'an integer' if kind is int else 'a number'}, got {raw!r}") from None
    return raw


def load_config(path: str | Path | None, sets: list[str]) -> Config:
    """The configuration of a command: defaults or a JSON file, then the `--set` overrides."""
    config = Config() if path is None else Config.from_json(Path(path).read_text())
    return apply_sets(config, sets)


def policy_config(args: argparse.Namespace, policy: Policy) -> Config:
    """A policy's configuration with overrides; a differing game knob is an error (§14.2)."""
    base = policy.config if args.config is None else Config.from_json(Path(args.config).read_text())
    config = apply_sets(base, args.set)
    differences = knob_differences(config.game, policy.config.game)
    if differences:
        listed = ", ".join(f"{name}={now} (policy: {stored})" for name, (now, stored) in differences.items())
        raise ValueError(f"game knobs differ from the policy's: {listed}")
    return config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pacman", description="Grid-world Pac-Man against a ghost.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", type=Path, metavar="FILE", help="JSON configuration file")
        command.add_argument(
            "--set", action="append", default=[], metavar="NAME=VALUE",
            help="override one game knob or learner setting by its bare name; repeatable",
        )

    train = commands.add_parser("train", help="train the learned agent and write a run directory")
    train.add_argument("--out", type=Path, required=True, metavar="DIR")
    common(train)

    evaluate_ = commands.add_parser("evaluate", help="play all agents on a split and report")
    evaluate_.add_argument("--split", choices=("val", "test"), required=True)
    evaluate_.add_argument("--policy", type=Path, required=True, metavar="FILE")
    common(evaluate_)

    check = commands.add_parser("check", help="play the baselines on validation under the given knobs")
    check.add_argument("--policy", type=Path, metavar="FILE")
    common(check)

    simulate = commands.add_parser("simulate", help="play games without graphics and summarise")
    simulate.add_argument("--agent", action="append", choices=AGENT_NAMES, required=True, metavar="NAME")
    simulate.add_argument("--split", choices=("train", "val", "test"), required=True)
    simulate.add_argument("--games", type=int, required=True, metavar="N")
    simulate.add_argument("--policy", type=Path, metavar="FILE")
    common(simulate)

    watch = commands.add_parser("watch", help="play one game and draw it")
    watch.add_argument("--agent", choices=AGENT_NAMES, required=True, metavar="NAME")
    watch.add_argument("--seed", type=int, required=True)
    watch.add_argument("--policy", type=Path, metavar="FILE")
    watch.add_argument("--delay", type=float, default=0.25, metavar="SECONDS")
    watch.add_argument("--plain", action="store_true", help="plain-text frames instead of pixel art")
    common(watch)
    return parser.parse_args(argv)


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    out = training.prepare_output_dir(args.out)  # refuse an existing run before training

    def progress(episodes: int, win_rate: float) -> None:
        print(
            f"episodes {episodes}: validation win rate {100 * win_rate:.1f}%"
            f" ({training.CURVE_GAMES} games)", flush=True,
        )

    result = training.train(config, on_progress=progress)
    print(TABLE_HEADER)
    print(format_summary_row(result.validation))
    training.write_outputs(out, result)
    print(f"wrote {out / 'policy.npz'}, curve.csv, curve.png and config.json")


def _evaluate(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    config = policy_config(args, policy)
    split = Split(args.split)
    games = evaluate.FULL_GAMES
    records = {
        name: run_split(config, make_agent(name, config, policy), split, games) for name in AGENT_NAMES
    }
    summaries = [summarise(name, agent_records) for name, agent_records in records.items()]
    comparisons = [compare(records["learned"], records[name]) for name in ("random", "pathfinding")]
    print(format_report(split, games, config.game, summaries, comparisons))
    path = Path(args.policy).parent / f"eval-{split.value}.json"
    write_eval_json(path, config, [record for agent_records in records.values() for record in agent_records])
    print(f"wrote {path}")


def _check(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    policy = load_policy(args.policy) if args.policy is not None else None
    result = difficulty_check(config, policy)
    print(format_header(Split.VAL, evaluate.FULL_GAMES, config.game))
    print(TABLE_HEADER)
    for summary in result.baselines:
        print(format_summary_row(summary))
    if result.stored is not None:
        print(format_summary_row(result.stored))
        knobs = []
        for field in fields(config.game):
            text = f"{field.name}={getattr(config.game, field.name)}"
            if field.name in result.knobs:
                text += f"* (policy: {result.knobs[field.name][1]})"
            knobs.append(text)
        print("policy knobs: " + " ".join(knobs) + ("  (* differs from the policy)" if result.knobs else ""))


def _simulate(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy) if args.policy is not None else None
    config = policy_config(args, policy) if policy is not None else load_config(args.config, args.set)
    split = Split(args.split)
    agents = [make_agent(name, config, policy) for name in args.agent]
    summaries = evaluate.simulate(config, agents, split, args.games)
    print(format_report(split, args.games, config.game, summaries))


def _watch(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy) if args.policy is not None else None
    config = policy_config(args, policy) if policy is not None else load_config(args.config, args.set)
    split_of_seed(args.seed)  # the split comes from the seed; an unknown seed is an error
    agent = make_agent(args.agent, config, policy)
    plain = use_plain(args.plain, sys.stdout.isatty(), os.environ)
    shown = False

    def show(frame) -> None:
        nonlocal shown
        if shown:
            time.sleep(args.delay)
            if plain:
                print()  # plain frames scroll, which suits logs and tests
        if not plain:
            print(CLEAR_SCREEN, end="")  # pixel art redraws in place, so the game animates
        print(render_frame(frame, plain), flush=True)
        shown = True

    play_episode(Environment(config), Referee(config.game.max_steps), agent, args.seed, on_frame=show)


COMMANDS = {
    "train": _train, "evaluate": _evaluate, "check": _check, "simulate": _simulate, "watch": _watch,
}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except Exception as error:  # the command boundary: every failure is a message and exit 1
        if os.environ.get("PACMAN_DEBUG"):
            raise
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
