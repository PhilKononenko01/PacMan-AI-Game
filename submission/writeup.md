# Pac-Man AI Project - Intermediate Report

### Current Progress:

**Christian Gore & Phillip Kononenko**

For our project, we created a Pac-Man style game on a 5x5 grid with randomly generated walls. Pac-Man's goal is to visit all four corners without being caught by the ghost. The ghost has limited vision, while Pac-Man always knows where the ghost is.

So far, we have implemented the main game mechanics along with three different strategies: random, pathfinding, and Q-learning. We wanted to see how well a Q-learning player could perform compared to simpler approaches. Our original goal was to get the Q-learning player's win rate between 50% and 90%.

We built the game in Python, using NumPy to help manage the Q-learning data, Matplotlib to visualize training results, and pytest to check that our code works correctly. These libraries made it easier to test and compare the different players.

### Testing and Results

We tested our game using automated tests and by running 1,000 games for each player. All 476 automated tests passed, and we got the following results:

| **Player**   | **Win Rate** |
| ----------- | ------------ |
| Random      | 0.0%         |
| Pathfinding | 14.9%        |
| Q-learning  | 73.6%        |

Based on these results, our Q-learning player performed significantly better than the other two players. It also reached our original goal of having a win rate between 50% and 90%.

We used automated tests to check that the game mechanics and players were working correctly. We also tested the players using the same set of random seeds so that the comparisons would be fair.

### What We Have Left

At this point, most of the main functionality is finished. We still want to work toward a power-up system that allows an alternate win condition of eating the ghost but we haven't started working toward that yet. We expect that to significantly alter the outcomes, as it offers a much easier win condition but is only possible once, adding a new dynamic for the Q-learning player to process and adapt its strategy. We still want to look more closely at how the Q-learning player makes decisions and see if there are areas where it could improve. We also need to prepare our final presentation and explain our approach and results.

Overall, the game is running correctly, and we have been able to show that our Q-learning player performs better than the random and pathfinding players.
