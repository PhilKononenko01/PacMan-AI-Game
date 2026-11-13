# A separate, versioned hybrid game

The goal is a trained Pac-Man that wins at least 55% of all scheduled games against a learned search ghost, under the approved two-pellet rules. This design makes that claim reproducible without reinterpreting historical results. The target remains an empirical outcome, not a guarantee.

Use immutable observations and explicit episode state first:

```python
@dataclass(frozen=True)
class PacObservation:
    pacman: Cell
    ghost: Cell
    pellets: frozenset[Cell]
    known: tuple[KnownCell, ...]
    turns_left: int

@dataclass(frozen=True)
class GhostObservation:
    cell: Cell
    visible_pacman: Cell | None
    turns_left: int

@dataclass(frozen=True)
class Sighting:
    cell: Cell
    direction: Direction | None
    age: int

@dataclass(frozen=True)
class SearchState:
    sighting: Sighting
    target: Cell | None
    moves_left: int
    choices_left: int

@dataclass(frozen=True)
class TurnResult:
    pacman: PacObservation
    pac_reward: float
    outcome: Outcome | None
```

The ghost controller owns either no memory or `SearchState`, plus the preceding visible observation. It receives the static map at construction, because its BFS may know walls. It never receives true current Pac-Man coordinates through an environment object, callback, or reward function. `Model the Domain` changes the design here: bounded search is explicit state rather than several loosely synchronized flags.

I recommend `pacman/hybrid/` with its own `game.py`, `ghost.py`, `learning.py`, `artifacts.py`, and `runner.py`. Reuse `grid.py`, `mapgen.py`, `rng.py`, and `stats.py`. Add `python -m pacman hybrid train|evaluate|watch` through a small CLI dispatch. Existing commands retain their rules. Give watch a small renderer that draws remaining pellets, remembered walls, positions, turns, and outcome.

A full core migration would replace `env.Observation`, corner-specific pathfinding, feature indexing, learner terminal handling, renderer assumptions, and policy loaders together. It ultimately removes duplicate episode mechanics, but preserving replay of version-1 artifacts then requires a legacy engine or pervasive rules branches. The opt-in package is the smaller coherent change now. Its boundary represents two actual games, not a temporary implementation toggle. Historical submission files remain untouched.

The existing `Environment.step` establishes Pac-Man movement, collision, collection and early win, ghost movement, collision. Preserve that order. A hybrid environment owns its 100-turn horizon so every consumer receives the same terminal outcome. Reset samples exactly two unique open cells, excluding both spawns, with the pellet RNG. Pac-Man senses adjacent walls and remembers them. Mark its own cell open too. At turn 100, resolve collisions and collection first, then report a tie if play remains active. Every terminal result immediately finalizes both learners, including a win before the ghost acts.

The gameplay owner's contract is:

```python
class HybridGame:
    def __init__(self, rules: Rules, ghost: GhostController): ...
    def reset(self, layout: GameMap, streams: EpisodeStreams) -> PacObservation: ...
    def step(self, action: Direction) -> TurnResult: ...

class GhostController:
    def reset(self, layout: GameMap, rng: Random) -> None: ...
    def act(self, observation: GhostObservation) -> Direction: ...
    def reward(self, value: float, *, terminal: bool) -> None: ...
```

The environment constructs one ghost observation immediately before each ghost movement. Consecutive visible observations can infer direction only when their positions differ by one legal step. An intervening invisible observation breaks that inference. Reacquisition sets direction to unknown until the next consecutive visible observation. Last reliably observed direction may remain in the old sighting during lost sight. No hidden displacement updates it.

While visible, chase with BFS. Otherwise pursue the last seen cell. Upon reaching it, choose one of four directional investigation targets derived solely from that sighting and the map. A target is the furthest open cell up to three cells along that direction, with blocked or already exhausted choices masked. BFS navigates to it. Commit rules constants of memory age 12 turns, at most three investigation choices, and at most four moves per target. Expiry clears memory and uses random legal wandering. These proposed constants need fixing before experiments; never adjust them to inflate the target rate. Baseline and learned ghosts share this controller. Only the search-choice function differs, uniform baseline versus epsilon-greedy Q.

Search learning uses a variable-duration transition. A chosen search action stays pending until the next search decision or episode termination, including any intervening visible pursuit. Thus reacquisition does not discard later capture credit. Accumulate discounted rewards each game turn; update with `R + gamma**duration * max(Q[next])`. Terminal transitions have zero bootstrap. Capture gives +1 exactly once, Pac-Man completion gives -1, timeout gives 0. There is no hidden-distance shaping or reward for timeout survival. Clear pending state after each update; no pending search means no ghost Q update.

The learning owner's contract is:

```python
def pac_features(obs: PacObservation) -> tuple[int, ...]: ...
def search_features(obs: GhostObservation, state: SearchState,
                    layout: GameMap) -> tuple[int, ...]: ...
class PacLearner:
    def act(self, obs: PacObservation) -> Direction: ...
    def observe(self, result: TurnResult) -> None: ...
    def freeze(self) -> FrozenPolicy: ...
class SearchLearner:
    def choose(self, state: tuple[int, ...], legal: tuple[int, ...]) -> int: ...
    def reward(self, value: float, *, terminal: bool) -> None: ...
    def freeze(self) -> FrozenPolicy: ...
```

Pac-Man starts with existing compressed features adapted to nearest pellet, pellet count, known wall mask, relative ghost position, remembered-map distance, and remaining-time bucket. Sparse tuple-keyed Q tables avoid accidental table-size coupling. Version feature definitions explicitly. Frozen policies copy their arrays and expose no update method. Loaders reject nonfinite values, invalid dimensions, unsupported schema or rules IDs, mismatched role, and incompatible rules hashes.

The runner owns reproducible training stages and artifacts:

```python
def play_episode(rules, key: EpisodeKey, pac, ghost, on_frame=None) -> EpisodeRecord: ...
def train_run(spec: TrainingSpec, out: Path) -> RunManifest: ...
def evaluate_pair(pac: FrozenPolicy, ghost: FrozenPolicy,
                  schedule: tuple[EpisodeKey, ...]) -> Evaluation: ...
```

`EpisodeKey` records layout seed, replicate, stage, and episode index. Derive separate layout-selection, pellets, ghost, exploration, and opponent-selection streams. Preserve map partitions. Train Pac-Man against baseline, freeze several checkpoints, train ghost against that pool, freeze ghost, then train Pac-Man against a fixed mixture including that ghost. Store policy hashes, opponent hashes and weights, seeds, rules, features, and schedules in the manifest. Evaluations require frozen opponents and verify hashes remain unchanged.

Run every scheduled validation and test game. Never call `winnable.is_winnable`. Select checkpoints on validation only. Report wins, losses, ties, denominators, Wilson intervals over scheduled maps, and ranges across independent training seeds. Prefer five replicates and their cross-play matrix if runtime permits. A test rate below 55% is a recorded failure, not grounds to change rules or omit maps.

Before training, tests must prove hidden-position invariance of ghost actions and nonterminal rewards, broken direction inference across lost sight, bounded memory expiry, unique pellets, collection once, Pac-Man-first terminal credit, timeout finalization, variable-duration discounting, frozen opponent integrity, schema rejection, and CLI artifact replay. Run the unchanged legacy suite as the compatibility check.
