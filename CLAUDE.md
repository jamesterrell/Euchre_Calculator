# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

There is no build system, package manifest, or linter. Dependencies are installed directly:

```bash
pip install numpy numba jupyter
```

Tests. The unit suite is stdlib `unittest`, so it needs nothing beyond numpy
and numba:

```bash
python -m unittest discover               # whole suite, ~55s including JIT warmup
python -m unittest tests.test_solver      # one module
python -m unittest tests.test_solver.TestLeftBower -v
python tests/test_solver.py               # or run a file directly
```

Plus the randomised regression sweep, which is a script rather than a test case:

```bash
python tests/test_fast_search.py          # 400 hands, ~20s including JIT warmup
python tests/test_fast_search.py 2000     # more hands
```

Run everything from the repo root. `tests/__init__.py` puts the root and the
tests directory on `sys.path`, which is why bare-name imports work inside the
suite and why `discover` needs no `-t` flag.

Run both after any solver change.

Primary workflow is the notebook:

```bash
jupyter notebook interface.ipynb
```

Anything can also be driven from a plain Python session at the repo root (live
modules are top-level and imported by bare name):

```bash
python -c "from fast_search import definitive_winner; ..."
```

Sizing batches: solving is ~0.5 ms/hand and `generate_hands` costs ~0.49 ms/hand,
so hand generation is now roughly half the cost of a sweep. 10,000 hands is
about 10 s end to end. Sampling error dominates search error -- there is no
search error -- so push the sample count when you want a tighter interval.

The solver's `@njit` functions compile on first call in a fresh process, ~15 s.
The archived pipeline's warmup is ~197 s; budget for it if you run a comparison.

## Architecture

### Card representation

A card is a 2-element `int64` vector, not a (suit, rank) pair. `deck.py` holds all 24. Suit is direction, strength is magnitude:

- Hearts `[-9..-14, 0]`, diamonds `[9..14, 0]`, clubs `[0, -9..-14]`, spades `[0, 90..140]`.
- **Spades is always trump.** The whole engine is written for a single fixed trump suit; a hand with a different trump must be rotated into spades by the caller.
- Trump ranks are scaled up by 10x, so `norm(card) > 80` is a trump test.
- The left bower (jack of clubs) is stored as `[0, 135]` -- inside the trump axis, above the ace of spades and below the right bower `[0, 140]`. Clubs therefore has no jack in its own range. Suit membership is positional, so never infer a card's suit from its raw numbers.

This vector form is now only the **I/O format**. `fast_search.encode_hands`
decomposes each card into `(suit, strength)` integers on the way in and the
search never touches a vector again -- no `np.linalg.norm`, no `arccos`, no
`norm > 80`. That is why `deck.py` and `dealer.py` still slot in unchanged.

### Module layering

```
deck.py            card constants
dealer.py          Dealer dataclass: shuffle, stack specific cards, deal 4x5
n_game_sim.py      generate_hands() -> (n_games, 4, 5, 2) batch of dealt hands
fast_search.py     the solver: depth-first alpha-beta over the game tree
reference_solver.py independent pure-Python solver, used only by the tests
tests/             the suite, see "Testing" below
  euchre_testkit.py  fixtures: named cards, line replay, and an exhaustive
                     no-pruning minimax used as the ground-truth oracle
  test_*.py          unit tests
  test_fast_search.py randomised regression sweep for fast_search
archive/           superseded code, see "Archived approaches" below
```

`tests/euchre_testkit.py` is deliberately named so `unittest discover` does not
collect it. Its `full_minimax` prunes nothing at all, and its cost varies far
more than the average suggests: over 15 random deals at ~1M nodes/s, 0.18 s at
the fastest, 2.7 s median, **55 s at the slowest**. Follow-suit is the only
thing bounding the branching factor, so the ceiling is a hand where nobody can
ever follow -- (5!)^4 = 207M sequences, about 3.5 minutes.

So never point it at an arbitrary hand. The tests use `BRUTE_FORCEABLE`, three
pinned deals -- one per outcome -- whose node counts are recorded next to them
and total about a third of a second. Add to that tuple rather than reaching for
a seed: a seeded pick makes the suite's runtime luck, and one unlucky reseed
turns a fast suite into a multi-minute one with nothing looking wrong. For
exploratory calls, `node_limit=` raises `SearchTooLarge` instead of hanging.

`fast_search.py` imports only numpy and numba -- nothing from this repo.

### Teams and scoring

Even players `(0, 2)` are one team, odd `(1, 3)` the other. Score is always
**from the calling team's perspective**: `+2` march (all 5 tricks), `+1` win
(3-4 tricks), `-2` euchred (0-2 tricks). Optimal play is minimax by parity: a
player maximizes when `player % 2 == caller % 2` and minimizes otherwise.

Player indices are absolute (0-3), never relative to the lead.

### How the solver works

`fast_search._search` is a recursive alpha-beta search that plays one card per
call. State is carried in mutable arrays that are mutated and restored around
each recursive call:

- hands as `(4, 5)` `suits` / `strs` planes plus a per-player count `n`. Playing
  card `i` swaps it to the end of the hand and decrements `n`; the identical
  swap on the way back out restores it.
- the trick under construction in `t_suit` / `t_str` / `t_player`, shaped
  `(tricks, 4)` and indexed **by `trick_no`**. That row indexing is load-bearing:
  a single shared 4-element buffer gets clobbered when the recursion enters the
  next trick, which silently corrupts `_resolve`. That was a real bug, caught
  only by the independent cross-check.

**Legality is enforced at move generation, never checked afterwards.** If
`n_in_trick == 0` every card is legal; otherwise the player's remaining cards
are scanned for the led suit and, if any is found, every other card is skipped.
Void players skip nothing. The Euchre-specific part lives in the encoding, not
the check: the left bower encodes as trump, so it cannot follow clubs and must
follow trump. Legality is the *only* restriction on moves -- all strategy comes
from minimax.

`_resolve` picks the winner: highest trump if any trump was played, else highest
card of the led suit. Trump strengths (90-140) and plain strengths (9-14) never
overlap, so no cross-suit comparison can go wrong.

Two forced-outcome cutoffs fire at trick boundaries: the calling team can no
longer reach 3 tricks (`-2`), or it has 3 and a march is already impossible
(`+1`).

Alpha-beta here is exact, not approximate -- it skips only branches that
provably cannot change the value. Verified against an exhaustive minimax with
every cutoff removed: identical value on 60/60 hands while visiting 0.31% of the
nodes (789k vs 253M). On `test_hand.txt` that is 14.5k nodes vs 2.16M.

### Testing

`tests/test_deck.py`, `test_dealer.py`, `test_n_game_sim.py`,
`test_reference_solver.py` and `test_solver.py` are the unit suite. The layering matters: `test_solver.py`
checks `fast_search` against `reference_solver`, and `test_reference_solver.py`
checks *that* against `euchre_testkit.full_minimax`, which cannot prune and so
cannot be wrong the way an alpha-beta search can. Do not collapse those layers.

`tests/test_fast_search.py` is the broad randomised sweep and validates two ways,
neither of which trusts the archived pipeline:

1. against `reference_solver.py`, a pure-Python solver written straight from the
   rules with different move ordering and without the forced-outcome cutoffs;
2. by replaying the returned line and re-deriving everything -- seat order, that
   each card was still held, the revoke check against the hand as it stood, the
   trick winner, and the final score against the trick count.

Run it after any solver change. An earlier version agreed with the old pipeline
on 30/41 hands but failed the independent cross-check, which is what surfaced
the trick-buffer bug above. Agreement with the archived code is not evidence.

**Do not add `cache=True`** to the solver's `@njit` functions. Numba 0.60
segfaults (SIGSEGV, reliably) when loading a cached *recursive* njit function,
so the 15 s warmup cannot currently be cached away. Making `_search` iterative
with an explicit stack would unblock that if the warmup ever matters.

## Archived approaches

`archive/beta_approach/` holds the previous solver: `tree_search.py`,
`n_play_round.py`, `n_branches.py`. They import each other by bare name, so to
run them put that directory on the path rather than treating it as a package:

```python
import sys; sys.path.insert(0, "archive/beta_approach")
from tree_search import definitive_winner
```

`interface.ipynb`'s last cell does exactly this to compare the two solvers on
`hands_test`.

**It disagrees with `fast_search` on roughly 20% of hands, in both directions,**
because it does not compute a minimax value. It scores a move by the **mean**
outcome over branches surviving its heuristic filters, then prunes players
greedily one at a time. On `test_hand.txt` it returns 1 where the true value is
2 -- the calling team can force all five tricks. Treat its numbers as historical.

How it worked, briefly: `n_play_round` built the full Cartesian product of every
player's remaining cards (`n**4` rows) and narrowed it with a filter chain --
`n_ap_filter` to pin the opener, then for each responder in seating order
`nfb_by_hand` (follow suit), `common_sense`, `smart_loss`, `trump_or_dump`. Only
the first is a legality rule; the other three are strategy heuristics, and
mixing them into the same chain is precisely why legal-but-optimal plays got
discarded. `tree_search` then expanded that tree once per candidate opening card
and again once per responder prune, for each of 5 tricks -- roughly 40
expansions of the same tree per hand.

Two hazards if you ever edit it: `next_round` writes each trick's winner at
column `game_round - 1` of an always-5-wide score array, so short simulations
need padded score arrays and shifted `game_round` values -- hand-unrolled and
duplicated across `n_trick_sim`, `player_best_prune`, and `definitive_winner`
with different offsets in each. And buffers are worst-case preallocated
(`round1` 625 branches, `next_round` `len(leads) * 4**4`), which is what drove
the ~533 MB resident footprint.

`archive/legacy_approach/` is older still, superseded by both. Do not extend it.

## Repo notes

- `fast_search.solve` / `solve_line` validate their arguments through `_validate`. They used to take anything: `starting_player=6` indexed straight past the end of the `(4,)` and `(4, 5)` state arrays and **segfaulted the interpreter** (njit has no bounds checking), while a hand with a card count other than 5 returned `-1000` from `solve` and wrote past the end of `solve_line`'s trick buffers. `solve_line` now sizes those buffers with `_ALL` rather than `hands.shape[1]`. `_final` and the forced-outcome cutoffs still hardcode 5 tricks and 3 needed, which is why anything but a 5-card hand is rejected outright rather than solved.
- `Dealer.stack_deck` appends to a player's hand instead of replacing it. Replacing meant the second call dropped the first call's cards while leaving them removed from the deck, so they were dealt to nobody -- `generate_hands` hit this whenever `stack` and `up_card` named the same player, and the hand then played out with 17 live cards. It also now validates the seat, the card shape (accepting a bare `(2,)` card), and that the stack cannot overfill a 5-card hand; the last used to surface as `ValueError: Negative dimensions are not allowed` from inside `np.random.choice`.
- `Dealer.__post_init__` honours `players` rather than always building four hands, and rejects a table the deck cannot seat.
- `Dealer.stack_deck` used `np.isin(self.deck, stack_cards).all(axis=1)`, which compared each *coordinate* against every value in the stack rather than matching whole cards, so it silently deleted extra cards from the deck (stacking 9d `[9,0]` and Ac `[0,-14]` also removed Ah `[-14,0]`). Those cards then could not be dealt to anyone. It now matches rows and raises if it does not match exactly `len(stack_cards)` cards. Measured effect on one affected stack: EV moved from +1.530 to +1.608 over 5000 deals, non-overlapping CIs.
- `test_hand.txt` is the canonical fixture, also inlined in `tests/test_fast_search.py`, `tests/test_solver.py` and `interface.ipynb`. Its true value is 2 with `starting_player=2, caller=0`.
