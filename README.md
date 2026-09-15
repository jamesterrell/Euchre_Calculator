
# Euchre Calculator

A Euchre engine that answers a hand two ways: **God Mode**, where all four
players see all 24 cards and play the true optimum, and the **Perfect
Information Monte Carlo (PIMC) sim**, where each player sees only its own cards
and works the rest out by guessing.

Both run on the same exact solver. The difference is entirely in what the
players are allowed to look at, which is the point -- the gap between the two
numbers is the price of not being able to see.

## The two modes

### God Mode

Every seat sees all four hands, the kitty included, and bids and plays the
genuinely best move. `bidding.solve_bidding` searches the whole auction and
`fast_search` solves the trick play exactly, with alpha-beta that prunes only
branches that provably cannot change the value.

This is the *correct* answer to "was this hand worth ordering up", and it is
not a model of a real table -- nobody at a table knows any of that. It is the
baseline everything else is measured against.

```python
import random, game, bidding as b

deal = game.deal_random(rng=random.Random(0), dealer=3)
out = b.solve_bidding(deal, allow_loners=True)

print(out)                      # seat 1 ordered up hearts -> +2 to team 0
print(b.first_bid_options(deal))  # {'pass': .., 'order': .., 'order alone': ..}
```

### The PIMC sim

Each player sees only its own five cards, the up-card, and the cards already
played. At every decision it samples layouts of the unseen cards consistent
with everything it has watched happen -- the counts, the suits people have
shown out of, where the up-card went -- solves each of those in God Mode, and
takes the option with the best average.

It is genuinely not omniscient, and it fails in recognisable ways: it cannot
signal to its partner, and it is optimistic about plans that depend on knowing
which layout it is really in.

```python
import random, game, table, players

deal = game.deal_random(rng=random.Random(0), dealer=3)

honest = [players.PIMCPlayer(samples=20, rng=random.Random(s)) for s in range(4)]
result = table.play_deal(deal, honest, allow_loners=True)

print(result)          # seat 2 ordered up diamonds -> 3 tricks, +1 to the caller
print(result.auction)  # ('seat 0 pass', 'seat 1 pass', 'seat 2 orders up diamonds')
```

Swap in `players.PerfectPlayer` for God Mode, or mix them at one table.

### What the two say about each other

Measured over 60 deals at 20 play samples and 10 bid samples, loners allowed.
Sampling error is large at this size -- these are shapes, not constants.

|                           | PIMC sim     | God Mode |
| ------------------------- | ------------ | -------- |
| passed out                | 0%           | 0%       |
| called alone              | 10%          | 1.7%     |
| ordered up in round one   | 57/60        | 41/60    |
| named a suit in round two | 3/60         | 19/60    |
| euchred                   | 43% of calls | 10%      |
| marched                   | 10% of calls | 28%      |
| mean tricks to the caller | 2.75         | 3.55     |
| mean points to the caller | -0.10        | +1.02    |

Head to head with the teams swapped on every deal so seat and dealer advantages
cancel exactly: **God Mode beats the PIMC sim by 1.26 +/- 0.36 points a deal**.
A euchre is worth 2, for scale.

```bash
python pimc_sweep.py 60                 # both tables, profiled side by side
python pimc_sweep.py 60 --head-to-head  # the two against each other
python pimc_example.py                  # one deal, every decision narrated
python pimc_example.py --seed 8         # a loner, made
python pimc_example.py --perfect        # the same deal, in God Mode
```

`pimc_example.py` is the one to read first. It plays a single pinned deal and
prints what every seat could see, what each option was worth, and which it took.

## Installation

```bash
git clone https://github.com/yourusername/euchre-calculator.git
cd euchre-calculator
pip install numpy numba jupyter
jupyter notebook interface.ipynb
```

## Card representation

Cards are 2D integer vectors. Suit is direction, strength is magnitude, and
**spades is always trump** -- the whole engine is written for one fixed trump
suit.

```python
# Hearts:   negative x  [-14, 0] .. [-9, 0]     (Ace to 9)
# Diamonds: positive x  [9, 0] .. [14, 0]       (9 to Ace)
# Clubs:    negative y  [0, -14] .. [0, -9]     (Ace to 9)
# Spades:   positive y  [0, 90] .. [0, 140]     (trump, scaled 10x)

[0, 140]   # right bower (jack of spades)
[0, 135]   # left bower (jack of clubs -- inside the trump axis)
[-14, 0]   # ace of hearts
```

Trump ranks are scaled up by 10x, so trump and plain strengths never overlap
and no cross-suit comparison can go wrong. Suit membership is positional: never
infer a card's suit from its raw numbers.

### Calling a different suit

Any other call is rotated into the canonical frame first. `rotation.py` takes
natural `Card(suit, rank)` cards and does it:

```python
from rotation import parse_hand, deal_to_engine, HEARTS
from fast_search import definitive_winner

hands = [parse_hand("JH JD AH KH QH"),   # both bowers, hearts called
         parse_hand("JS JC AS KS QS"),
         parse_hand("AD KD QD TD 9D"),
         parse_hand("AC KC QC TC 9C")]

score = definitive_winner(deal_to_engine(hands, HEARTS), starting_player=0, caller=0)
```

Rotation is not a plain suit relabel: the left bower follows the *colour* of
trump, so hearts called makes the jack of diamonds trump, not the jack of clubs.
For every trump suit the mapping is a bijection onto the full deck, which is
what keeps the solver from silently solving the wrong position.

## Scoring

Score is always from the calling team's perspective. Even seats `(0, 2)` are one
team, odd `(1, 3)` the other.

- **+2** all 5 tricks (march)
- **+1** 3-4 tricks
- **-2** 0-2 tricks (euchred)
- **+4** all 5 tricks playing alone

### Going alone

`alone=True` sits the caller's partner down: its cards are dealt but never
played, tricks are three cards instead of four, and taking all five pays 4.
Three or four tricks is still 1 and a euchre still costs 2, so a loner is worth
`-2`, `1` or `4` -- never 2.

```python
alone = definitive_winner(hands, starting_player=0, caller=0, alone=True)
```

In the auction loners are opt-in, because in God Mode they change the result on
only ~1% of deals and leaving them off keeps four-handed numbers comparable.
The PIMC sim calls them far more often -- 10% against 1.7% -- partly from honest
optimism and partly because averaging over sampled worlds destroys the exact
ties that made God Mode decline them.

Defending alone is not modelled.

## The solver

`fast_search.py` walks the game tree once, depth first:

1. **Move generation enforces legality** -- follow the led suit if you can, and
   nothing else is restricted. All strategy comes out of the minimax.
2. **Minimax by parity** -- a player maximises when on the calling team.
3. **Alpha-beta pruning** -- exact, not approximate. Verified against an
   exhaustive minimax with every cutoff removed: identical value on 60/60 hands
   while visiting 0.31% of the nodes.
4. **Forced-outcome cutoffs** at trick boundaries.
5. **Loners** run the same search over three seats.

Key entry points:

- `definitive_winner()` -- the calling team's score
- `solve()` -- the score plus nodes visited
- `solve_line()` -- the score plus one optimal line of play
- `solve_position()` / `position_moves()` -- a hand part-way through, which is
  what the PIMC sim needs and a fresh-deal solver cannot give

Solving costs ~0.5 ms/hand four-handed and ~0.23 ms alone. The `@njit`
functions compile on first call in a fresh process, about 15 s.

## Project structure

```
├── README.md
├── deck.py                   # Card constants, in canonical spades-trump form
├── rotation.py               # Natural cards <-> the solver's canonical frame
├── game.py                   # Deal: hands, up-card, kitty, dealer seat
├── bidding.py                # The auction, solved in God Mode
├── dealer.py                 # Card dealing and hand management
├── n_game_sim.py             # Batch hand generation
├── fast_search.py            # The solver: depth-first alpha-beta
├── reference_solver.py       # Independent pure-Python solver, used by tests
├── observation.py            # What one seat knows; sampling worlds from it
├── table.py                  # The referee: play a deal out with four players
├── players.py                # Decision rules: God Mode, PIMC sim, random
├── pimc_sweep.py             # God Mode against the PIMC sim, in bulk
├── pimc_example.py           # One deal, every decision printed
├── interface.ipynb           # One worked example, notebook form
├── tests/                    # Test suite
└── archive/                  # Superseded implementations, kept for reference
```

## Testing

Standard library `unittest`; nothing extra to install.

```bash
python -m unittest discover             # the full suite, ~70s with JIT warmup
python -m unittest tests.test_solver    # one module
python tests/test_fast_search.py 2000   # randomised regression sweep
```

Run these from the repo root. The suite is layered deliberately:
`test_solver.py` checks `fast_search` against `reference_solver.py`, and
`test_reference_solver.py` checks *that* against an exhaustive minimax that
prunes nothing and so cannot be wrong the way an alpha-beta search can.

## Dependencies

NumPy, Numba, and Jupyter (optional, for the notebook).

## License

MIT License - see LICENSE file for details
