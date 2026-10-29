"""The learning-curve PNG, the only Matplotlib user, on the Agg backend."""

import subprocess
import sys

from pacman.plot import save_curve

PNG = b"\x89PNG\r\n\x1a\n"


def test_save_curve_writes_a_png(tmp_path):
    path = tmp_path / "curve.png"
    save_curve(path, [(2500, 0.365), (5000, 0.41), (7500, 0.5)])
    assert path.read_bytes()[:8] == PNG


def test_an_empty_curve_still_writes_a_png(tmp_path):
    path = tmp_path / "curve.png"
    save_curve(path, [])
    assert path.read_bytes()[:8] == PNG


def test_only_plot_imports_matplotlib_and_uses_agg():
    code = (
        "import sys, pacman.train, pacman.evaluate, pacman.policy_io;"
        " print('matplotlib' in sys.modules);"
        " import pacman.plot, matplotlib;"
        " print(matplotlib.get_backend().lower())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert result.stdout.split() == ["False", "agg"]
