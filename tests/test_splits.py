"""Held-out splits never share a layout (NFR-6)."""

from pacman.config import Config
from pacman.mapgen import Split, generate, layout_bytes

SAMPLE = 1_000


def test_no_layout_appears_in_two_splits():
    config = Config()
    layouts = {
        Split.TRAIN: {layout_bytes(generate(seed, Split.TRAIN, config)) for seed in range(SAMPLE)},
        Split.VAL: {
            layout_bytes(generate(seed, Split.VAL, config))
            for seed in range(1_000_000, 1_000_000 + SAMPLE)
        },
        Split.TEST: {
            layout_bytes(generate(seed, Split.TEST, config))
            for seed in range(2_000_000, 2_000_000 + SAMPLE)
        },
    }
    assert not layouts[Split.TRAIN] & layouts[Split.VAL]
    assert not layouts[Split.TRAIN] & layouts[Split.TEST]
    assert not layouts[Split.VAL] & layouts[Split.TEST]
