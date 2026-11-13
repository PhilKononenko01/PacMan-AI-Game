# Pac-Man AI: Final project report

**Christian Gore and Phillip Kononenko**  
**CSCI 3202: Introduction to Artificial Intelligence**

## Project overview

For our final project, we built a Pac-Man-style game to investigate whether reinforcement learning could outperform random movement and a basic pathfinding strategy. Pac-Man must visit all four corners of a randomly generated 5×5 grid while avoiding a single ghost. The project combines a game environment, a scripted opponent, three Pac-Man agents, a training system, and tools for testing and replaying games.

Our learned agent uses tabular Q-learning. We wanted it to win roughly 50% to 90% of held-out games, which we used as a difficulty guideline. We also wanted to establish whether it won more often than both baselines. The frozen policy won 736 of 1,000 test games, or 73.6%, compared with 149 wins for pathfinding and zero for random movement.

## Game rules and implementation

Pac-Man starts at the top-left corner, and the ghost starts at the bottom-right corner. The starting corner counts as visited, leaving three corners for Pac-Man to reach. Both characters move at most one cell per turn. An action into a wall or outside the grid leaves the character in place. Pac-Man moves first. A collision ends the game in a loss, visiting the last corner ends it in a win, and reaching 100 turns without either outcome produces a tie. Completing the final corner wins before the ghost can reply. A win or loss on turn 100 takes priority over a timeout.

Pac-Man always knows the ghost's position but discovers walls only in the four adjacent cells. Discovered cells remain in memory until the next game. Unknown cells are treated as open when planning. The ghost knows the true wall layout, but sees Pac-Man only when they share a row or column with no wall between them. It chases a visible Pac-Man, pursues the last sighting after losing visibility, and otherwise searches neighboring cells it visited least recently. The main ghost is scripted; it does not learn.

The map generator samples walls in non-corner cells with probability 0.20, then rejects unsuitable candidates. Accepted maps have connected open cells, at most six walls, and at least four independent loops. These loops provide alternative routes. These structural checks do not prove that an agent can win every accepted game.

The implementation separates responsibilities. `grid.py` supplies movement and search; `mapgen.py` generates maps; `env.py` manages game state, observations, and rewards; and `ghost.py` controls the opponent. `referee.py` counts turns and declares ties. Every player uses the same `reset`, `act`, and `observe` interface through the common loop in `episode.py`. Training, evaluation, rendering, and policy storage use separate modules. The notebook includes the complete application source below the report and reconstructs it as ordinary Python files.

## Libraries and tools

| Tool | How we used it |
|---|---|
| Python 3.11+ and its standard library | Implemented game rules, BFS, seeded randomness, configuration, statistics, file handling, and command-line commands. See the [Python standard library reference](https://docs.python.org/3/library/). |
| NumPy | Stored and indexed the numerical Q-table and saved the policy array. See [What is NumPy?](https://numpy.org/doc/stable/user/whatisnumpy.html). |
| Matplotlib | Plotted validation win rate over training episodes. See the [Matplotlib quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html). |
| pytest | Ran automated checks of individual modules and complete workflows. See the [pytest documentation](https://docs.pytest.org/en/stable/). |
| Jupyter and IPython | Combined the report, source, recorded evidence, and executable demonstrations in one notebook. See [Project Jupyter](https://jupyter.org/). |

We implemented BFS, ghost behavior, and the Q-learning update directly. The project does not use a game engine or a neural-network framework.

## Player strategies and training

The random player chooses uniformly among four directions, including blocked directions. The pathfinding player uses breadth-first search to find the nearest unvisited corner on the remembered map and takes a shortest-path step toward it. It replans each turn but ignores the ghost. This provides a navigation baseline that can reach corners efficiently when its route is safe.

The learned player uses six features: the BFS direction toward the nearest unvisited corner, an adjacent-open-cell mask, the ghost's relative position, a bucketed path distance to the ghost, estimated line of sight, and the number of remaining corners. These produce 122,880 possible table rows, each holding four action values. The learner can choose a different action from the BFS recommendation. Our comparison therefore tests whether learning when to deviate from a shortest known route improves success.

We trained the saved policy for 50,000 episodes using learning rate 0.1 and discount factor 0.99. Exploration probability decreased from 1.0 to 0.05 over the first half of training, then remained at 0.05. Rewards were +100 for winning, −100 for losing, +1 for a newly visited corner, and −0.05 per turn. These rewards add together. Evaluation used greedy action selection and disabled Q-table updates.

Every 2,500 training episodes, a greedy evaluator played 200 validation games to produce the learning curve. Its win rate rose from 0.5% at 2,500 episodes to 59.0% at 25,000 and 72.5% at 50,000. The final policy then won 69.1% across the full 1,000-game validation split. The checkpoint curve and full validation use different sample sizes. We saved the final table rather than selecting the checkpoint with the highest score.

## Testing and experimental method

We tested software correctness separately from player performance. The recorded final acceptance run passed all **476 automated tests**. The suite checks blocked movement, connected maps, wall sensing and memory reset, ghost vision and pursuit, collision ordering, corner rewards, timeout handling, feature encoding, Q updates, configuration validation, and policy saving and loading. Integration tests cover training, saving, loading, evaluation, and replay. Determinism tests check repeatability within the supported runtime. The recorded test output is preserved in `reports/results/tests-final.log` in the repository; automated test source is not embedded in the submission notebook.

For performance, each agent played the same 1,000 test seeds, from 2,000,000 through 2,000,999. Training used seeds 0 through 49,999, while validation used 1,000,000 through 1,000,999. Separate seed ranges alone would not prevent repeated layouts from crossing splits, so we also assigned each wall layout to one split using SHA-256. A layout can repeat within a split but cannot belong to multiple splits. The test games contained 491 distinct layouts.

Separate random streams control maps, the ghost, and the player. This makes games reproducible without one component directly consuming another component's random draws. All players start from matched seeds and maps, although their different actions lead to different ghost trajectories. The policy and settings were frozen before the reported final evaluation. This report uses the saved results rather than a newly selected run.

We report wins, losses, and ties separately, with 95% Wilson confidence intervals. We also use a one-sided exact McNemar test for each learned-versus-baseline comparison. It pairs results by seed and compares cases where only one player wins. Losses and ties both count as not-wins for that comparison. The implementation and recorded calculations are in `pacman/stats.py` and `reports/results/verification-test.json`.

## Results and analysis

| Player | Wins | Losses | Ties | Win rate and 95% Wilson interval | Mean turns per win |
|---|---:|---:|---:|---|---:|
| Random | 0 | 1,000 | 0 | 0.0% [0.0%, 0.4%] | Not applicable |
| Pathfinding | 149 | 851 | 0 | 14.9% [12.8%, 17.2%] | 13.7 |
| Q-learning | 736 | 220 | 44 | 73.6% [70.8%, 76.2%] | 36.5 |

The learned policy exceeded pathfinding by 58.7 percentage points and random movement by 73.6 percentage points. In the paired comparison against pathfinding, 612 seeds were wins only for the learned player, while 25 were wins only for pathfinding. The exact one-sided p-value was approximately 9.28 × 10⁻¹⁴⁸. Against random movement, the corresponding counts were 736 and zero, with p approximately 2.77 × 10⁻²²². Both comparisons passed the specified p < 0.05 criterion. The CLI displays these very small values as `0.0000`; they are not zero.

The learned rate falls within our difficulty guideline. Its longer successful games are consistent with taking detours to avoid capture, while the baseline always follows a shortest route to its chosen corner. However, the turn averages cover different successful subsets and do not establish a paired efficiency advantage. The learned player still lost 22.0% and tied 4.4% of test games, so the result does not show universal or optimal play.

The results are specific to this policy, ghost, feature representation, and accepted-map distribution. Partial wall knowledge can make route estimates inaccurate, and different situations can share one feature row. The features also omit the ghost's internal memory and the time remaining. These limitations help explain why useful learned behavior does not imply a complete solution to every game.

## Example of the application working

The notebook replays the recorded learned-policy game on seed 2,000,000 using the saved policy. It ends in a win on turn 33. From a configured project directory, the same game can be run with:

```sh
python -m pacman watch --agent learned --seed 2000000 \
  --policy reports/results/policy.npz --delay 0 --plain
```

Its final frame is:

```text
*...*
.....
.....
?....
P.G.*
Move 33/100  Corners 4/4  Ghost sees Pac-Man: yes
Outcome: win
```

Here `P` is Pac-Man, `G` is the ghost, `*` marks a visited corner, and `?` is an unknown cell. Pac-Man has visited all corners without needing to reveal every cell. The episode return is 101.35: the +100 win reward, three +1 corner rewards, and 33 step costs of −0.05.

The notebook also reruns a separate 100-game experiment in which both characters move randomly. Its recorded outcome is two Pac-Man wins, 89 losses, and nine ties, averaging 42.7 turns per game. This uses a different ghost from the main experiment, so it is not a matched comparison with the results above. Ordinary notebook execution displays the recorded final evaluation and learning curve, runs this separate experiment, and replays the learned win. It does not retrain the policy or repeat the full held-out evaluation.

## Conclusion

This project showed us how reinforcement learning can improve decisions when reaching a goal also requires avoiding danger. The pathfinding baseline already knew how to move toward corners, but its shortest routes often led to capture. Combining that navigation information with ghost-related features allowed the learned policy to win substantially more often. The 73.6% test win rate met our difficulty guideline and exceeded both baselines under the paired comparisons.

We also learned that the environment and evaluation design matter as much as the update equation. Movement order, observations, rewards, and timeout handling all affect the task being learned. Automated tests helped check those rules, while matched seeds and layout-based splits made the performance comparison more credible. Keeping recorded evidence separate from live demonstrations also made the notebook easier to reproduce and explain.

The remaining losses and ties show the limitations of our compact state representation. Further work could examine failed games and state coverage before adding information to the learner. A power-up system remains a possible extension from our intermediate report, but it is not part of this implementation. It would change collision rules and learning inputs and require new training and evaluation. Our completed project focuses on the four-corner objective and demonstrates that learned decisions can improve on a basic navigation strategy within that setting.

## Appendix: How the AI works — optional extra-credit explanation

### Learning action values

Reinforcement learning uses interaction and rewards to improve an agent's decisions. A policy specifies how the agent chooses actions, while an action value estimates the future return associated with a choice. Exploration allows the learner to experience alternatives instead of always repeating its current preference. These concepts are described in Sutton and Barto's *Reinforcement Learning: An Introduction*, especially Chapters 2, 3, and 6. [Sutton and Barto, authors' second-edition draft](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

Our implementation represents Q(s, a) as one table entry for encoded state s and direction a. The table begins at zero. During training, taking an action produces a reward and another observation. The learner adjusts the selected entry toward the observed reward plus an estimate of the best future value. This is tabular Q-learning, a model-free temporal-difference method. [Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

For a continuing transition, the update is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right].
$$

The target combines immediate reward r with the discounted best next-state estimate. The learning rate α controls the adjustment size. Q-learning is off-policy because the maximizing action in its target can differ from the exploratory action actually chosen during training. [Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

In our code, α is 0.1 and γ is 0.99. For example, if the current value is 2, an ordinary move earns −0.05, and the best next-state value is 10, the target is 9.85. The updated value becomes 2 + 0.1 × (9.85 − 2), or 2.785. A terminal win or loss uses the immediate reward alone. The default timeout update still bootstraps from the next state, treating the external clock cutoff as truncation. The referee ends the actual game regardless. This behavior is implemented in `pacman/agents/learned.py`.

### Representing the situation

Our state encoder summarizes an observation rather than storing a separate table for every map. It combines the direction toward the nearest corner, four local openness bits, the ghost's relative offset, estimated ghost distance, estimated visibility, and remaining-corner count. Their product is 4 × 16 × 80 × 4 × 2 × 3 = 122,880 rows. Four `float64` action values per row occupy 3,932,160 bytes before file metadata. `pacman/agents/features.py` defines this encoding, and `pacman/policy_io.py` defines the array shape.

The representation lets a table entry apply across layouts with similar local conditions. It also loses information: full wall memory, exact remaining-corner identities, ghost memory, and the clock are not preserved in the row. Distinct situations can therefore receive identical action values. We do not claim that this compressed state satisfies all assumptions needed for an optimality guarantee. Watkins and Dayan's convergence result concerns appropriate Markovian conditions and repeated sampling; our empirical success does not establish those conditions. [Watkins and Dayan, 1992](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)

### Search, exploration, and the opponent

The navigation feature comes from our BFS implementation. Since each grid move has equal cost, BFS finds shortest routes on the remembered map. Unknown cells count as open, and planning repeats after new walls are sensed. The pathfinding baseline always follows this recommendation. The Q-learning player uses it as one feature and can choose another direction. The resulting system combines handcrafted navigation with learned choices about progress and danger; it does not learn navigation entirely from raw observations.

Our training code uses epsilon-greedy action selection. Epsilon starts at 1.0, falls to 0.05 by the halfway point, and remains there for the rest of 50,000 episodes. Random choices sample all four directions, so blocked actions are possible. The greedy evaluator selects a maximum-valued action and never changes the table. This separates the training behavior from measurement. The exploration-versus-exploitation distinction and greedy action selection are discussed by Sutton and Barto. [Sutton and Barto, authors' second-edition draft](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

The ghost supplies a separate form of AI through search and memory. `pacman/ghost.py` uses true-map line of sight, BFS pursuit, and least-recently-visited neighbor selection. It updates its last sighting even on a turn when its movement gate makes it stay still. Pac-Man's visibility feature, by contrast, uses remembered walls and can be inaccurate. This information difference is part of the challenge. The reported test results show that our frozen learned policy handles that challenge more successfully than the two baselines, while leaving substantial room for better decisions.

### Sources and implementation evidence

- Watkins, C. J. C. H., and Dayan, P. (1992). [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf). *Machine Learning*, 8, 279–292. Algorithm and convergence conditions.
- Sutton, R. S., and Barto, A. G. (2018). [Reinforcement Learning: An Introduction, second edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/). MIT Press. Reinforcement-learning terminology, exploration, and action-value methods. An accessible [authors' draft from 2014–2015](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) supplies the linked chapter discussions.
- The official Python, NumPy, Matplotlib, pytest, and Jupyter references are linked in the libraries table above.
- Project source: `pacman/agents/features.py`, `pacman/agents/learned.py`, `pacman/agents/pathfinding.py`, `pacman/ghost.py`, and `pacman/train.py`. These files are included below in the notebook.
- Experimental evidence: `reports/results/config.json`, `curve.csv`, `eval-test.json`, and `eval-val.json`, embedded in the notebook. The repository's `reports/results.md`, `verification-test.json`, and `tests-final.log` preserve further verification details.
