# Final project submission

Review `final_report.md` for the final report and optional extra-credit AI explanation.
The conclusion is written in the team's voice; both authors should review it before submission.

`CSCI_3202_Final_Report.ipynb` combines that report with all application source, the saved
policy, recorded experimental data, and runnable demonstrations. It preserves the
intermediate submission as a separate file. No gameplay or policy changes were made.

Open the final notebook with a Python 3.11+ Jupyter kernel containing NumPy and Matplotlib,
then select **Restart Kernel and Run All**. It reconstructs the application in a new
temporary directory, displays the recorded final results, runs 100 random-versus-random
games, and replays the recorded learned-policy win on seed 2,000,000. Training is disabled
by default. The full held-out evaluation is not repeated. Automated test source is not
embedded; the report explains the recorded suite and points to its repository evidence.

The extra-credit appendix is approximately 700 words plus references. Its final printed
page count depends on export formatting; check the instructor's 1–2 page limit if submitting
it as a separate document. The main report's conclusion contains three paragraphs.

To regenerate the notebook after editing the report, from the repository root:

```sh
python submission/build_notebook.py \
  --report submission/final_report.md \
  --output submission/CSCI_3202_Final_Report.ipynb
```

Regeneration clears cell outputs. Run all cells again before submitting the notebook.
