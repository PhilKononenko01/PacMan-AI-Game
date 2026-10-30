"""Build the portable course notebook from the application source and saved evidence."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
import zlib


def build(project: Path, experiment: Path, output: Path) -> None:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    files = {p.relative_to(project).as_posix(): p.read_bytes()
             for p in sorted((project / "pacman").rglob("*.py"))}
    application_paths = list(files)
    for name in ("pyproject.toml", "README.md"):
        files[name] = (project / name).read_bytes()
    original_experiment = (experiment / "run.py").read_text()
    git_lookup = 'subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()'
    if original_experiment.count(git_lookup) != 1:
        raise ValueError("Expected exactly one experiment Git provenance lookup")
    portable_experiment = original_experiment.replace(git_lookup, repr(revision)).replace("import subprocess\n", "")
    files["experiments/random_vs_random.py"] = portable_experiment.encode()
    artifact_paths = ["policy.npz", "config.json", "curve.csv", "curve.png", "eval-test.json", "eval-val.json"]
    artifacts = {f"reports/results/{name}": (project / "reports/results" / name).read_bytes()
                 for name in artifact_paths}
    for name in ("results.json", "summary.txt"):
        artifacts[f"evidence/random-vs-random/{name}"] = (experiment / name).read_bytes()
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    artifact_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}
    cells = []

    def cell(kind: str, source: str, metadata=None) -> None:
        if kind == "markdown":
            source = textwrap.dedent(source).strip("\n") + "\n"
        elif not source.endswith("\n"):
            source += "\n"
        value = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": metadata or {},
                 "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """
    # CSCI 3202: Intermediate Project Report
    ## Pac-Man AI, Option 1: Your Choice
    **Phillip Kononenko and Christian Gore**  
    **October 30, 2026**

    This notebook contains the report, all application source, the saved trained policy,
    recorded experimental data and runnable demonstrations. The automated test suite's
    source is intentionally excluded. Only this `.ipynb` file is needed for the submission.

    Use a **Python 3.11 or later Jupyter kernel**, then choose **Restart Kernel and Run All**.
    NumPy and Matplotlib must be available in that kernel. If they are missing, run
    `%pip install numpy matplotlib` in a new cell, restart the kernel and run all cells.

    Source cells reconstruct normal Python files in a fresh temporary workspace. They do
    not overwrite files beside this notebook or require Git or the original repository.
    Run All displays recorded results, runs 100 random-versus-random games and replays
    one recorded learned-agent win. It does not retrain the policy or repeat the final
    held-out evaluation. The optional training cell is disabled by default.
    """)
    cell("markdown", """
    ## Project and current implementation

    Our project asks whether reinforcement learning can outperform random movement and
    basic pathfinding in a Pac-Man-style game. A 5x5 grid contains randomly generated
    walls. Pac-Man starts at `(0, 0)` and must visit all four corners, with his starting
    corner already counted. The ghost starts at `(4, 4)`. Capture is a loss, and reaching
    100 turns without winning or losing is a tie. A blocked move leaves an actor in place.

    Pac-Man always knows the ghost's position but only senses adjacent walls. Discovered
    walls remain in memory for the current game. Unknown cells are treated as open when
    planning. Map generation keeps corners open, connects all open cells, allows at most
    six walls and requires at least four independent loops.

    The implementation includes the game, wall sensing and memory, rewards, a scripted
    ghost, three agents, training, evaluation, policy saving and terminal visualization.
    The **random agent** chooses among four directions. The **pathfinding agent** uses
    breadth-first search to move toward the nearest unvisited corner. The **learned
    agent** uses tabular Q-learning with BFS-derived features and can depart from that
    shortest route to avoid danger. The normal ghost does not learn. It chases visible
    Pac-Man, pursues his last observed position, and otherwise searches by least-recent visits.

    The learned state describes goal direction, neighboring open moves, relative ghost
    position, ghost distance, estimated visibility and remaining corners. This gives
    **122,880 states with four actions each**. Coordinates alone would be ambiguous across
    different wall layouts. The compact features permit reuse across maps but omit detail.

    Training uses **50,000 episodes**, learning rate **0.1** and discount **0.99**.
    Epsilon-greedy exploration decreases from **1.0 to 0.05** during the first half.
    Rewards are **+100** for winning, **-100** for capture, **+1** per new corner and
    **-0.05** per turn. The Q update learns from immediate reward and the estimated value
    of the next state. Wins and losses are terminal. A referee tie stops the episode but
    bootstraps the learning target under the approved default.

    ## Correctness and progress

    The recorded project acceptance suite passed **476 automated tests**. Tests cover
    movement, map constraints, sensing privacy, visibility, collision order, rewards,
    timeout precedence, numeric Q updates and per-game cache resets. Integration tests
    train, save, load, evaluate and replay a policy. Determinism checks reproduce games
    across processes. Those are historical verification results; no test source or
    automated test runner is included in this submission notebook.

    The core implementation and planned evaluation are complete. Remaining course work
    includes the final write-up and interview preparation. The learned agent still loses
    or ties some games. Its feature summary omits the ghost's internal history, so the
    results demonstrate improvement over baselines rather than guaranteed success on
    every game. Structural generator constraints are not a formal proof of a winning
    strategy for every encounter. Further improvement studies would use validation data.

    ## Libraries and sources

    | Tool | Role | Reference |
    |---|---|---|
    | Python standard library | Game, seeded randomness, configuration and CLI. | https://docs.python.org/3/ |
    | NumPy | Q-table storage and updates. | https://numpy.org/doc/stable/ |
    | Matplotlib | Learning-curve plots. | https://matplotlib.org/stable/users/index.html |
    | pytest | Original automated test suite, excluded here. | https://docs.pytest.org/en/stable/ |
    | Jupyter and IPython | Notebook execution and readable file-writing cells. | https://docs.jupyter.org/ and https://ipython.readthedocs.io/ |

    Dependencies are declared in the included `pyproject.toml`. Only NumPy and Matplotlib
    are required for the application demonstrations. Game rules, BFS, the ghost and the
    Q-learning update are implemented in the source below, without a neural-network framework.
    """)
    cell("markdown", "## Setup\nRun the following cells in order. The workspace location is printed so generated files can be inspected or saved.")
    parents = sorted({str(Path(name).parent) for name in files | artifacts})
    cell("code", textwrap.dedent(f"""
    import base64
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import tempfile
    import zlib
    from collections import Counter
    from IPython.display import Image, Markdown, display

    if sys.version_info < (3, 11):
        raise RuntimeError("Select a Python 3.11 or later kernel.")
    missing = [name for name in ("numpy", "matplotlib") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing) + ". Run %pip install numpy matplotlib, then restart.")

    import numpy
    import matplotlib
    SOURCE_COMMIT = {revision!r}
    WORKSPACE = Path(tempfile.mkdtemp(prefix="pacman-course-"))
    RUN_OUTPUTS = WORKSPACE / "notebook_runs"
    RUN_OUTPUTS.mkdir()
    for parent in {parents!r}:
        (WORKSPACE / parent).mkdir(parents=True, exist_ok=True)
    os.chdir(WORKSPACE)
    RUN_ENV = dict(os.environ, PYTHONPATH=str(WORKSPACE), PYTHONNOUSERSITE="1", MPLBACKEND="Agg")
    RUN_ENV["MPLCONFIGDIR"] = str(WORKSPACE / "matplotlib-cache")
    print("Python", sys.version.split()[0], "NumPy", numpy.__version__, "Matplotlib", matplotlib.__version__)
    print("Source snapshot:", SOURCE_COMMIT)
    print("WORKSPACE =", WORKSPACE)
    """))
    cell("markdown", """
    ## Complete application source

    Each cell below writes the named file and otherwise contains the original application
    source. These cells do not execute the module bodies. The modules are loaded normally
    by the demonstration subprocesses after reconstruction. The production application
    modules are unchanged from the recorded source snapshot.
    """)
    for name, data in files.items():
        if name == "experiments/random_vs_random.py":
            cell("markdown", """
            ### Separate random-versus-random experiment
            This copy replaces only the original Git revision lookup with the fixed source
            snapshot identifier above, and removes its unused Git subprocess import. That
            provenance-only adaptation makes it portable outside a repository. The random
            policies, game rules, seeds and outcome collection are unchanged. Its live copy
            runs in a fresh output folder rather than beside the recorded evidence.
            """)
        cell("markdown", f"### `{name}`")
        cell("code", f"%%writefile {name}\n" + data.decode(), {"tags": ["application-source"], "source_path": name})

    packed = [{"path": name, "sha256": artifact_hashes[name],
               "compressed_base64": base64.b64encode(zlib.compress(data, 9)).decode()}
              for name, data in artifacts.items()]
    cell("markdown", """
    ## Restore the saved policy and recorded evidence
    The next cell contains compressed **data**, not additional source or test code.
    Its embedded policy and records make this notebook self-contained. Every restored
    file is checked against its recorded SHA-256 before use. `eval-test.json` is the
    final experiment's outcome data, not automated test source.
    """)
    cell("code", "ARTIFACTS = " + repr(packed) + "\n\n" + textwrap.dedent("""
    for item in ARTIFACTS:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid embedded artifact path")
        data = zlib.decompress(base64.b64decode(item["compressed_base64"]))
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Embedded artifact checksum mismatch: " + item["path"])
        destination = WORKSPACE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    FROZEN_POLICY = WORKSPACE / "reports/results/policy.npz"
    print("Restored", len(ARTIFACTS), "recorded artifacts, including the frozen policy.")
    """), {"tags": ["embedded-data"], "jupyter": {"source_hidden": True}})
    cell("code", textwrap.dedent("""
    def run_pacman(*arguments):
        completed = subprocess.run([sys.executable, "-m", "pacman", *arguments],
                                   cwd=WORKSPACE, env=RUN_ENV, capture_output=True, text=True)
        if completed.returncode:
            print(completed.stdout, completed.stderr)
            completed.check_returncode()
        return completed.stdout

    provenance = subprocess.run([sys.executable, "-c", "import pacman; print(pacman.__file__)"],
                               cwd=WORKSPACE, env=RUN_ENV, capture_output=True, text=True, check=True)
    print("Application imported from:", provenance.stdout.strip())
    print(run_pacman("--help"))
    """))
    cell("markdown", """
    ## Recorded results against the scripted ghost

    All three Pac-Man agents played the same **1,000 held-out test seeds** after freezing
    the policy and settings. The following table is calculated from the included recorded
    data. It does not run another held-out evaluation. Repeated layouts within a split
    are permitted; those seeds generated 491 distinct layouts.

    The learned agent's 95% Wilson win-rate interval was **70.8% to 76.2%**. Both one-sided
    exact paired McNemar comparisons passed `p < 0.05`. The 73.6% learned rate meets our
    proposed 50% to 90% difficulty guideline. Losses and ties remain separate from wins.
    """)
    cell("code", textwrap.dedent("""
    recorded = json.loads((WORKSPACE / "reports/results/eval-test.json").read_text())
    rows = ["| Pac-Man agent | Wins | Losses | Ties | Win rate |", "|---|---:|---:|---:|---:|"]
    for name in ("random", "pathfinding", "learned"):
        games = [r for r in recorded["records"] if r["agent"] == name]
        counts = Counter(r["outcome"] for r in games)
        rows.append(f"| {name} | {counts['win']} | {counts['loss']} | {counts['tie']} | {counts['win'] / len(games):.1%} |")
    display(Markdown("\\n".join(rows)))
    display(Image(filename=str(WORKSPACE / "reports/results/curve.png"), width=700))
    """))
    cell("markdown", """
    ## Run the separate 100-game random-versus-random experiment

    Both characters choose uniformly among up, right, down and left. Blocked moves stay
    in place and consume a turn. The four-corner goal, collision order and 100-turn limit
    are unchanged. The recorded result was **2 Pac-Man wins, 89 losses and 9 ties**, with
    **42.7 Pac-Man turns per game** on average. A turn includes Pac-Man's action and,
    if the game continues, the ghost's action. This opponent differs from the scripted
    ghost above, so the two experiments are not a matched comparison of agent performance.

    This cell reruns the experiment in a fresh output folder. Original recorded results
    stay under `evidence/random-vs-random/`.
    """)
    cell("code", textwrap.dedent("""
    experiment_run = Path(tempfile.mkdtemp(prefix="random-vs-random-", dir=RUN_OUTPUTS))
    script = experiment_run / "run.py"
    shutil.copyfile(WORKSPACE / "experiments/random_vs_random.py", script)
    completed = subprocess.run([sys.executable, str(script)], cwd=WORKSPACE, env=RUN_ENV,
                               capture_output=True, text=True)
    if completed.returncode:
        print(completed.stdout, completed.stderr)
        completed.check_returncode()
    print(completed.stdout)
    fresh = json.loads((experiment_run / "results.json").read_text())
    print("Average Pac-Man turns:", fresh["mean_steps"])
    print("New experiment output:", experiment_run)
    """))
    cell("markdown", """
    ## Replay a learned-agent game
    This is the already recorded win on seed **2,000,000**. It should end at **move 33**.
    The command uses the saved policy greedily and does not train it. The final frame is
    displayed below; the full plain-text replay is saved in the temporary output folder.
    """)
    cell("code", textwrap.dedent("""
    replay = run_pacman("watch", "--agent", "learned", "--seed", "2000000",
                        "--policy", str(FROZEN_POLICY), "--delay", "0", "--plain")
    replay_path = RUN_OUTPUTS / "learned-replay.txt"
    replay_path.write_text(replay)
    print(replay.strip().split("\\n\\n")[-1])
    print("Full replay:", replay_path)
    """))
    cell("markdown", """
    ## Optional training demonstration
    Leave the switch below `False` for ordinary Run All. Setting it to `True` trains a
    new, small 100-episode policy in a fresh folder. The normal training command also
    runs **1,000 final validation games**, even for this short run, so it takes longer
    than the 100 training episodes alone. It never changes the included frozen policy.

    A full training run uses the same command without the `--set episodes=100` override.
    """)
    cell("code", textwrap.dedent("""
    RUN_TRAINING_DEMO = False
    if RUN_TRAINING_DEMO:
        training_run = Path(tempfile.mkdtemp(prefix="training-demo-", dir=RUN_OUTPUTS))
        print(run_pacman("train", "--out", str(training_run), "--set", "episodes=100"))
        print("New policy and learning-curve files:", training_run)
    else:
        print("Optional training is disabled. The submitted policy has not been retrained.")
    """))
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11", "file_extension": ".py", "mimetype": "text/x-python"},
        "pacman_submission": {"source_commit": revision, "application_paths": application_paths,
            "source_sha256": source_hashes, "artifact_sha256": artifact_hashes,
            "experiment_original_sha256": hashlib.sha256(original_experiment.encode()).hexdigest(),
            "experiment_adaptation": "Git revision subprocess replaced with fixed source snapshot; unused import removed.",
            "test_source_included": False}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes, {len(cells)} cells, {len(application_paths)} application files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment-root", type=Path,
                        default=Path(__file__).with_name("random_vs_random"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.project.resolve(), args.experiment_root.resolve(),
          args.output or Path(__file__).with_name("CSCI_3202_Intermediate_Report.ipynb"))
