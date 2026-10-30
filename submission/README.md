# Intermediate report submission

Upload `CSCI_3202_Intermediate_Report.ipynb` as the single-file submission. It contains
the report, all 23 application modules, the saved policy, recorded results and runnable
demonstrations. Automated test source is excluded.

Open it with a Python 3.11 or later Jupyter kernel and select **Restart Kernel and Run All**.
If NumPy or Matplotlib is missing, run `%pip install numpy matplotlib` in that kernel first.
The notebook reconstructs files in a new temporary directory. It does not need this
repository beside it. Training is disabled by default.

The saved notebook includes outputs from a clean-directory execution. That run reproduced
the 100-game random-versus-random result of 2 wins, 89 losses and 9 ties, then replayed the
learned policy's recorded win at move 33. The recorded full evaluation against the scripted
ghost is shown from embedded data and is not repeated by Run All.

To regenerate an unexecuted notebook from this checkout:

```sh
python submission/build_notebook.py --output /tmp/CSCI_3202_Intermediate_Report.ipynb
```

The builder uses the tracked experiment inputs in `random_vs_random/`. Its notebook copy
replaces the experiment's Git provenance lookup with the recorded source revision so it
can run without Git. Game rules and random policies are unchanged.
