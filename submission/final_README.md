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

Each run chooses a fresh validation seed and episode key. Both characters use
their saved policies without training. Use `./play --delay 0 --plain` for
immediate text output. The final JSON records the seed and key.

To run other configurations:

```sh
./play --seed 1000000 --key 42
./play --seed 1000007 --key 43
./play --pac-baseline bfs
./play --pac-baseline random
./play --fixed-ghost
./play --evaluate --games 100 --repeats 2
./play --evaluate --pac-baseline bfs --fixed-ghost --games 100 --repeats 2
./play --help
```

The first command repeats the recorded 13-turn win. Without both identifiers,
a watched game uses fresh random inputs. `--seed` changes the wall-layout input;
`--key` changes pellets and episode randomness. Different inputs can still
produce the same wall layout. `--fixed-ghost` uses fixed search instead of learned
search. Batch evaluation uses a repeatable matched schedule and counts every
game. A 100-seed schedule with two keys per seed contains 200 games.

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
tables, displays the chart, runs a fresh game and checks its exact replay, and runs ten validation
games. The full 125,000-game replay and 545-test application suite are recorded
results. Those full checks are not repeated by Run All. The ten-game example is
a smoke check, not a new estimate of test performance.

Running the notebook uses the saved policies and does not start training.

The report explains the changes from the intermediate four-corner game to the
final two-pellet game. The intermediate report, notebook, and original builder
remain separate historical files. Temporary final-report builders have been
removed. Edit report text directly in the final notebook's Markdown cells.
