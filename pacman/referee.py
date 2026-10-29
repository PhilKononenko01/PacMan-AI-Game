"""The referee: the only holder of the move count and the only reader of max_steps (§7.4)."""

from pacman.env import Outcome, StepResult


class Referee:
    """Counts moves, stops the episode at max_steps and calls the tie."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.moves = 0

    def start(self) -> None:
        self.moves = 0

    def next_move(self) -> int:
        """Number the next move; the episode loop passes it to the environment."""
        self.moves += 1
        return self.moves

    def rule(self, result: StepResult) -> Outcome | None:
        """The ruling the agent receives: the step's win or loss, else a tie on the clock."""
        if result.outcome is not None:
            return result.outcome
        if self.moves >= self.max_steps:
            return Outcome.TIE
        return None
