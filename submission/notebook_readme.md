# Pac-Man AI Project

A Python 3.11+ grid-world game comparing random movement, BFS pathfinding and tabular
Q-learning. Pac-Man must visit all four corners while avoiding a scripted ghost. The
referee declares a tie after 100 turns when neither a win nor a loss has occurred.

The notebook includes the full application source, saved policy and experimental data.
Run its cells in order to reconstruct the project in a temporary directory. NumPy and
Matplotlib are the required application dependencies.

To replay the recorded learned-policy win from the reconstructed project directory:

```sh
python -m pacman watch --agent learned --seed 2000000 --policy reports/results/policy.npz --delay 0.25
```

The notebook also provides a separate 100-game random-versus-random experiment and an
optional short training demonstration. The proposed power-up system is future work.
