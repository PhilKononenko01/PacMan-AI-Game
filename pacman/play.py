import argparse
from pathlib import Path
import secrets

from pacman.hybrid import __main__ as hybrid


POLICIES = Path(__file__).resolve().parents[1] / "reports/hybrid-55/replicate-0"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="./play", allow_abbrev=False,
        description="Watch a fresh game using the saved learned Pac-Man and ghost policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Watch options forwarded to pacman.hybrid watch:
  --delay SECONDS     Frame delay, default 0.15 (0..10)
  --plain             Print frames without clearing the terminal

Batch options forwarded to pacman.hybrid evaluate:
  --split {val,test}  Default val
  --games N          First N layout seed slots, default 1000 (1..1000)
  --repeats N        Episode keys per layout, default 1 (1..10)
  --replicate N      Matched schedule replicate, default 0 (0..4)
  --out FILE         Save full JSON report; existing files are refused

Examples:
  ./play
  ./play --seed 1000000 --key 42 --delay 0 --plain
  ./play --pac-baseline bfs --fixed-ghost
  ./play --evaluate --games 10 --repeats 2 --out /tmp/games.json

Native command help: python -m pacman.hybrid watch --help
                     python -m pacman.hybrid evaluate --help
""",
    )
    parser.add_argument("--evaluate", action="store_true", help="run a reproducible batch instead of watching")
    pac = parser.add_mutually_exclusive_group()
    pac.add_argument("--pac-policy", type=Path, metavar="FILE", help="custom learned Pac-Man policy")
    pac.add_argument("--pac-baseline", choices=("random", "bfs"), help="replace learned Pac-Man with a baseline")
    ghost = parser.add_mutually_exclusive_group()
    ghost.add_argument("--ghost-policy", type=Path, metavar="FILE", help="custom learned ghost policy")
    ghost.add_argument("--fixed-ghost", action="store_true", help="use the fixed sight-memory BFS ghost")
    parser.add_argument("--seed", type=int, help="watch layout seed; omitted chooses validation seed 1000000..1000999")
    parser.add_argument("--key", type=int, help="watch episode key (0..2**63-1); omitted chooses a fresh random key")
    args, forwarded = parser.parse_known_args(argv)
    if args.evaluate and (args.seed is not None or args.key is not None):
        parser.error("--seed and --key are watch options; evaluation uses its matched schedule")

    command = ["evaluate" if args.evaluate else "watch"]
    if args.pac_baseline is not None:
        command += ["--pac-baseline", args.pac_baseline]
    else:
        command += ["--pac-policy", str(args.pac_policy if args.pac_policy is not None else POLICIES / "pacman.npz")]
    if not args.fixed_ghost:
        command += ["--ghost-policy", str(args.ghost_policy if args.ghost_policy is not None else POLICIES / "ghost.npz")]
    if not args.evaluate:
        seed = args.seed if args.seed is not None else 1_000_000 + secrets.randbelow(1000)
        key = args.key if args.key is not None else secrets.randbits(63)
        command += ["--seed", str(seed), "--key", str(key)]
    return hybrid.main(command + forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
