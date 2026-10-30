"""Episode-local random streams; never access the global random state."""

from hashlib import sha256
import random


def derive(seed: int, component: str) -> random.Random:
    """Derive an independent stream using a fixed encoding."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(component, str) or not component or not component.isascii():
        raise ValueError("component must be a non-empty ASCII string")
    digest = sha256(f"{seed}:{component}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def pick(rng: random.Random, n: int) -> int:
    """Choose an index in range(n), consuming exactly one random() draw."""
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    return min(int(rng.random() * n), n - 1)
