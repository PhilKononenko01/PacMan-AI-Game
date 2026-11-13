# Final project submission

Submit **CSCI_3202_Final_Report.ipynb** for Option 1. Everything needed for the
report is inside that file, including application code, saved policies, test
results, a working example, three conclusion paragraphs, and the cited
extra-credit AI explanation. A two-page PDF copy of that explanation is embedded
in the notebook's data and restored when it runs.

## Watch the game

From the repository root:

```sh
./play
```

Use `./play --delay 0 --plain` for immediate text output or `./play --key 43` for
another episode. Both characters act automatically using their saved policies.
The default example ends in a Pac-Man win after 13 turns. Playback does not train.

The launcher uses `.venv/bin/python` if available, otherwise `python3`. Set up
Python 3.11+ and the project dependencies before first use:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run the notebook

Use a Python 3.11+ Jupyter kernel with NumPy and Matplotlib. Install missing
packages in that environment before running, for example with
`python -m pip install jupyterlab numpy matplotlib`. Select **Restart Kernel and
Run All**. No repository checkout, Git, or internet connection is needed while
it runs. The notebook creates a temporary workspace and prints its location.

Run All checks the embedded source and evidence, recomputes recorded outcome
tables, displays the chart, replays the 13-turn example, and runs ten validation
games. The full 125,000-game replay and 545-test application suite are recorded
results. Those full checks are not repeated by Run All. The ten-game example is
a smoke check, not a new estimate of test performance.

Training is disabled by default. The optional small training cell demonstrates
the stages with tiny budgets and does not replace the reported policies.

The report explains the changes from the intermediate four-corner game to the
final two-pellet game. The intermediate report, notebook, and original builder
remain separate historical files. Temporary final-report builders have been
removed. Edit report text directly in the final notebook's Markdown cells.
