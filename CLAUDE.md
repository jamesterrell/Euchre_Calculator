# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Terminology

Two names, used consistently throughout the repo:

- **God Mode** -- all four players see all 24 cards and play the true optimum.
  `fast_search`, `bidding` and `players.GodModePlayer` are God Mode. This is
  the exact baseline, not a model of a real table.
- **Perfect Information Monte Carlo (PIMC) sim** -- a player sees only its own
  cards and the play so far, samples layouts consistent with that, and solves
  each sampled layout in God Mode. `players.PIMCPlayer`, fed by
  `observation.py`, driven by `table.py`.

The code uses the same names: `GodModePlayer`, `god_mode_table()`,
`PASS_GOD_MODE` (`"god"`), and `pimc_example.py --god-mode`.

## Commands

There is no build system, package manifest, or linter. Dependencies are installed directly:

```bash
pip install numpy numba jupyter
```

Tests. The unit suite is stdlib `unittest`, so it needs nothing beyond numpy
and numba:

```bash
python -m unittest discover               # whole suite, ~70s including JIT warmup
python -m unittest tests.test_solver      # one module
python -m unittest tests.test_solver.TestLeftBower -v
python tests/test_solver.py               # or run a file directly
```

Plus the randomised regression sweep, which is a script rather than a test case:

```bash
python tests/test_fast_search.py          # 400 hands twice, ~27s with JIT warmup
python tests/test_fast_search.py 2000     # more hands
```

Every hand in that sweep is solved twice, four-handed and as a loner, and the
JIT warmup covers both.

Run everything from the repo root. `tests/__init__.py` puts the root and the
tests directory on `sys.path`, which is why bare-name imports work inside the
suite and why `discover` needs no `-t` flag.

Run both after any solver change.

And the PIMC sim sweep, a measurement rather than a test -- no pass/fail, it
reports what honest players did:

```bash
python pimc_example.py                    # ONE deal, every decision narrated
python pimc_example.py --pass-model zero  # ...with passing priced at nothing
python pimc_sweep.py                      # 40 deals, PIMC sim vs God Mode
python pimc_sweep.py 60 --head-to-head    # what does seeing actually buy?
python pimc_sweep.py 100 --pass-model zero --bid-samples 16
```

And the EV of one pinned hand, which is the question the calculator exists to
answer -- same PIMC sim table, but every sampled layout is played out rather
than solved:

```bash
python hand_ev.py "JS AS 9H 9D TC" --up 9S --seat 0 --dealer 3
python hand_ev.py "JS AS 9H 9D TC" --up 9S --deals 2000 --player-eval-sims 50
python hand_ev.py "JS AS 9H 9D TC" --up 9S --both --deals 500
python hand_ev.py "JS AS 9H 9D TC" --up 9S --deals 5000 --workers 8
```

`pimc_example.py` is the one to read first. It plays a single pinned deal and
prints what every seat could see, what each of its options was worth, and which
it took -- the working behind "the player takes the highest-EV move". Its
docstring lists other seeds worth looking at: a euchre, a hostile discard, a
second-round call, a made loner.

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
- **Spades is always trump.** The whole engine is written for a single fixed trump suit. `rotation.py` does the rotation: `deal_to_engine(hands, trump)` takes natural `(suit, rank)` cards and any called suit and returns a solver-ready `(4, 5, 2)` array.
- Trump ranks are scaled up by 10x, so `norm(card) > 80` is a trump test.
- The left bower (jack of clubs) is stored as `[0, 135]` -- inside the trump axis, above the ace of spades and below the right bower `[0, 140]`. Clubs therefore has no jack in its own range. Suit membership is positional, so never infer a card's suit from its raw numbers.

This vector form is now only the **I/O format**. `fast_search.encode_hands`
decomposes each card into `(suit, strength)` integers on the way in and the
search never touches a vector again -- no `np.linalg.norm`, no `arccos`, no
`norm > 80`. That is why `deck.py` and `dealer.py` still slot in unchanged.

### Module layering

```
deck.py            card constants (already in canonical spades-trump form)
rotation.py        natural (suit, rank) cards <-> the canonical frame
game.py            Deal: 4x5 natural cards + up-card + kitty + dealer seat
bidding.py         the auction, solved in God Mode
dealer.py          Dealer dataclass: shuffle, stack specific cards, deal 4x5
n_game_sim.py      generate_hands() -> (n_games, 4, 5, 2) batch of dealt hands
fast_search.py     the solver: depth-first alpha-beta over the game tree
reference_solver.py independent pure-Python solver, used only by the tests
observation.py     one seat's information set, and sampling worlds from it
table.py           the referee: play a deal out with four player objects
players.py         decision rules: GodModePlayer, PIMCPlayer, RandomPlayer
pimc_sweep.py      measures a PIMC sim table against the God Mode one
pimc_example.py    one deal, every decision narrated -- read this one first
hand_ev.py         EV of one pinned hand, played out by a PIMC sim table
tests/             the suite, see "Testing" below
  euchre_testkit.py  fixtures: named cards, line replay, and an exhaustive
                     no-pruning minimax used as the ground-truth oracle
  test_*.py          unit tests
  test_loners.py     loners, solver and bidding both
  test_position.py   partially played positions; replay along an optimal line
  test_observation.py what a seat knows, and that sampled worlds respect it
  test_table.py      the referee and the players, incl. the God Mode pin
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

`BRUTE_FORCEABLE_ALONE` is the same idea for loners (`full_minimax(...,
alone=True)`), one deal per lone outcome, ~37k nodes total. Two of the three are
`BRUTE_FORCEABLE`'s own deals solved alone instead, which keeps the fixture
small and shows the same layout taking a different value once a hand leaves the
game.

`fast_search.py` imports only numpy and numba -- nothing from this repo.

### Teams and scoring

Even players `(0, 2)` are one team, odd `(1, 3)` the other. Score is always
**from the calling team's perspective**: `+2` march (all 5 tricks), `+1` win
(3-4 tricks), `-2` euchred (0-2 tricks). Optimal play is minimax by parity: a
player maximizes when `player % 2 == caller % 2` and minimizes otherwise.

Called alone, a march pays `+4` instead of `+2`. Nothing else changes: three or
four tricks is still `+1`, and being euchred alone still hands over only `2`. So
a loner's value is one of `{-2, 1, 4}` and **never 2** -- a useful invariant,
and one the tests assert. Defending alone is not modelled.

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

### Loners

`solve`, `solve_line` and `definitive_winner` all take `alone=`. It changes
three things and nothing else: the caller's partner `(caller + 2) % 4` is given
a card count of zero so its dealt hand is as out of play as the kitty, turn
order steps over that seat, and a trick completes at three cards. If the sitting
seat would have led trick one, the lead passes to the next live seat -- that
case is real, since play always starts left of the dealer whoever called.
`solve_line` returns `(5, 3)` play arrays instead of `(5, 4)`; read the width
off `ps.shape[1]` rather than assuming four.

**`_search_alone` is a hand copy of `_search`, deliberately.** The obvious
implementation threads the trick width and the sitting seat through the one
recursion, and that was written and measured first: identical node counts, but
**1.7x-2.5x slower per node** on the four-handed path, purely from the per-node
width test and the skip-the-sitting-seat step. Four-handed play is the hot path
-- an EV sweep is tens of thousands of those solves -- so it keeps its literal
`3`s and `4`s and loners get their own function. The shared pieces are the ones
where duplication would actually be dangerous: `_resolve` (the trick-winner
rule, which takes a width) and `_final` (the scoring). What guards the copy is
that `tests/test_loners.py` runs it against both independent oracles, and
`tests/test_fast_search.py` sweeps every hand twice.

Measured: loner solves are ~3x cheaper in aggregate (0.23 ms vs 0.75 ms/hand
over the same 400 hands). Not deal by deal, though -- a four-handed position
often trips a forced-outcome cutoff that the same layout alone does not.

Alpha-beta here is exact, not approximate -- it skips only branches that
provably cannot change the value. Verified against an exhaustive minimax with
every cutoff removed: identical value on 60/60 hands while visiting 0.31% of the
nodes (789k vs 253M). On `test_hand.txt` that is 14.5k nodes vs 2.16M.

### Testing

`tests/test_deck.py`, `test_dealer.py`, `test_n_game_sim.py`,
`test_reference_solver.py`, `test_solver.py` and `test_loners.py` are the unit
suite. The layering matters: `test_solver.py`
checks `fast_search` against `reference_solver`, and `test_reference_solver.py`
checks *that* against `euchre_testkit.full_minimax`, which cannot prune and so
cannot be wrong the way an alpha-beta search can. Do not collapse those layers.

`test_loners.py` covers both halves of the feature -- the solver's sitting seat,
three-card tricks and `+4` march, and bidding's `allow_loners` -- because they
are one change and splitting it across two files hides the seam where sign
conventions get lost. It uses the same three-layer cross-check.

`tests/test_fast_search.py` is the broad randomised sweep. It puts every hand
through twice -- four-handed, then alone -- and validates two ways, neither of
which trusts the archived pipeline:

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

## Playing without God Mode

Everything above this section answers a deal all at once, with every hand
visible. `table.py`, `observation.py` and `players.py` are the other mode: four
independent players, each seeing only its own cards, deciding one at a time.

```
observation.py     one seat's information set; sampling layouts consistent with it
table.py           the referee: drives a Deal through the auction and the play
players.py         decision rules -- GodModePlayer, PIMCPlayer, RandomPlayer
pimc_sweep.py      measures a PIMC sim table against the God Mode one
pimc_example.py    one deal, narrated decision by decision
hand_ev.py         EV of one pinned hand, played out by a PIMC sim table
```

`PIMCPlayer.last_scores` holds the averaged value of every option from its most
recent decision, which is what `pimc_example.py` prints and what a front end
would want. It is empty when there was nothing to decide.

The split between referee and strategy is strict. `table.py` holds no strategy
at all: it offers the legal options, checks the answer is one of them, and
writes down what happened. Every decision comes from a player object, so the
same loop gives a God Mode table, a PIMC sim table, or a table of coin flips
depending only on who is sitting at it.

**Players are handed both the truth and their own view.** `turn.deal` is the
whole table; `turn.observation` is that seat's information set. A player that
reads `turn.deal` is cheating by definition -- `GodModePlayer` does, on purpose.
`tests/test_table.py` wraps `turn.deal` in a tripwire and asserts `PIMCPlayer`
never touches it, because a peeking PIMC player would otherwise just look like
an unusually strong one.

**A table of four `GodModePlayer`s reproduces `solve_bidding` exactly** -- same
contract, same discard, same score, on every deal tested. That is the pin: the
referee and the God Mode player re-derive the existing baseline through
entirely new code, so the machinery is anchored to the old answer before the
PIMC sim rides on it. It matches because the option ordering and the tie rule
are `bidding._best`'s own: pass, call, call-alone, first of equals wins.

### What the solver needed for this

`solve` and `solve_line` both start from a fresh deal -- five cards each,
nothing played. That is the one position a player choosing a card is never in
after the opening lead. `fast_search.solve_position` and `position_moves` take
the position as it actually stands: per-seat card counts, a trick part-built,
tricks already won.

Nothing in the recursion had to change. `trick_no` and `caller_tricks` were
already absolute rather than relative, and `n` was already a per-seat count, so
entering at trick 3 with two cards each is just a different set of arguments;
the hardcoded 5 tricks and 3 needed stay true. What is new is `_check_position`,
and it is long on purpose -- njit has no bounds checking, so a card count that
disagrees with the trick number reads past the end of the state arrays and
takes the interpreter with it, the same failure `_validate` was written for.

`position_moves` returns a value per *legal card* rather than one value for the
position, which is what a player choosing a card needs and what `solve` does not
give. `solve_line`'s per-ply loop was rerouted through it, so every existing
`solve_line` test now exercises the new code as well.

The invariant that tests all of it: **along a God Mode optimal line the position
value never changes** -- both sides are already playing their best, so nothing
either does moves the number. `tests/test_position.py` walks
`solve_line`'s own line and re-solves from scratch at all 20 plies, and every
one of them has a known answer.

### What a seat knows

`observation.Observation` is one seat's view, and `sample_worlds` turns it into
concrete deals the solver can take. Three kinds of inference go into a sampled
world, all of them real Euchre rather than bookkeeping:

- **counts** -- every seat has played the same number of cards, so how many each
  still holds is public;
- **voids** -- a seat that failed to follow a led suit holds none of it, for the
  rest of the hand;
- **the up-card** -- turned down means buried and nobody holds it; ordered up
  means the dealer took it, and it is still in their hand unless they have since
  shown out of trump, in which case it must have been what they buried.

Voids are read in **effective** suits, so the left bower counts as trump. A
player who ruffs a club lead with the left bower has not shown a club void, and
a player who follows a club lead cannot have done it with the left bower. A
sampler working off the printed suit gets both backwards and deals people cards
they have already shown they cannot hold.

The kitty's capacity is **derived** from card conservation rather than counted,
which is what lets the one awkward moment -- the dealer holding six cards,
picked up and not yet thrown -- be an ordinary `Observation` with
`pending_discard=True` instead of a special case bolted onto the player.

**Bidding inference is deliberately not modelled.** Worlds are drawn as though
the auction said nothing about anybody's cards, so a seat that ordered up is not
assumed to hold trump. That makes PIMC sim players weaker than they could be, in
a specific direction: they under-rate the caller. Closing it needs a bidding
model, which is the thing this project is trying to produce, so the circularity
is left open on purpose rather than closed with a guess.

### The PIMC sim, and where the time goes

`PIMCPlayer` samples N layouts consistent with what its seat has seen, solves
each one in God Mode, and takes the option with the best average. It is not a
search over information sets and it does not know that it does not know, which
shows up as two well-known distortions -- **strategy fusion** (it credits itself
with plans that depend on knowing which world it is in) and **non-locality** (it
expects opponents to find defences they cannot see). Both make it optimistic.
Neither makes it weak.

**Pricing a pass is nearly the whole cost of bidding.** A pass is worth whatever
the rest of the auction does, so `pass_model="god"` (the default) runs the rest
of the auction in God Mode inside each sampled world -- up to 36 solves per
sample. It is inconsistent in an obvious way, since inside the sample the other
seats can see the hand this player is hiding, and it is still the best available
answer to "what happens if I decline". `pass_model="zero"` prices a pass at 0
instead: much faster, and a markedly more aggressive bidder.

Card play is cheap by comparison -- a mid-hand position solve is far smaller
than a whole hand, and a seat with one legal card skips the search entirely
rather than spending a few hundred solves confirming it has no choice.

Tie-breaks matter more here than they look. Averaging over worlds mostly
*destroys* the exact ties the God Mode auction resolved against calling, which
is part of why a PIMC sim table calls loners so much more often. Among cards the
search rates identically, `tie_break` defaults to playing the cheapest; that
only ever chooses between moves of equal expected value, so it cannot cost
anything the model can see, but pass `tie_break="first"` when measuring the sim
rather than trying to win with it.

### What honest players actually do

Measured by `pimc_sweep.py` over 60 deals at 20 play samples and 10 bid samples
per decision, loners allowed, dealer rotating. Sampling error is large at this
size -- these are shapes, not constants.

|                            | PIMC sim     | God Mode |
| -------------------------- | ------------ | -------- |
| passed out                 |  0%          |  0%      |
| called alone               | 10%          |  1.7%    |
| ordered up in round one    | 57/60        | 41/60    |
| named a suit in round two  |  3/60        | 19/60    |
| euchred                    | 43% of calls | 10%      |
| marched                    | 10% of calls | 28%      |
| mean tricks to the caller  | 2.75         | 3.55     |
| mean points to the caller  | -0.10        | +1.02    |

The two auctions land on the same trump suit 67% of the time and the same
caller 55% of the time.

**Seeing the other hands is worth about 1.26 points a deal.** `--head-to-head`
puts the PIMC sim on one team and God Mode on the other, and plays each deal
twice with the teams swapped so seat and dealer advantages cancel exactly rather
than statistically. Over 50 deals: **-1.26 +/- 0.36 points per deal** to the
sim. A euchre is worth 2, for scale.

**The sim over-calls.** 43% of its contracts are euchred against God Mode's
10%. How much of that is the optimizer's curse -- "take the best of several
noisy averages" -- rather than the pass model below is **not settled**, and how
far the sample count per decision has to go before a PIMC player's bidding
stops moving is unmeasured. Do not assume it is flat.

**Most of the over-calling is the pass model.** Same 40 deals with
`pass_model="zero"`: the euchre rate falls to 20-28% and the average call goes
from -0.17 to +0.65 points. Pricing a pass by running the rest of the auction in
God Mode makes declining look worse than it is, because God Mode essentially
always finds a call -- so the pass branch nearly always reads "an opponent ends
up calling this", and never "it comes back around to me".

**But the better-behaved bidder is not a better player, and this is a trap
worth knowing about.** Head to head against God Mode, teams swapped on every
deal: `"god"` scores -1.26 +/- 0.36 and `"zero"` scores -1.34 +/- 0.36 over the
same 50 deals. Indistinguishable. Mean points *per call* flattered `"zero"`
only because it is averaged over the deals a player chose to call, and silently
drops what the deals it passed on cost it. That is `bidding.py`'s "passing is
not free" turning up as a measurement trap rather than a bidding one -- do not
read a per-call average as a strength number. What `"zero"` reliably is, is
about 4x faster.

**Nothing ever passes out, under either model** -- 0 of 60 in the main sweep and
0 of 40 in all five diagnostic configurations. That was the one prediction in
this file that did not come true, and the pass model does not explain it:
`"zero"` prices a pass at exactly nothing and still never throws a hand in. The
reason is the number of chances. Eight seats bid in turn, and each round-two
seat is choosing among three suits -- six options once loners are on. Somebody
almost always finds something that looks positive, especially since the sim's
estimates are optimistic to begin with. Getting a table to pass a deal out
needs a model of what the *other* seats will do with it, which is exactly what
neither pass model has. That is the next thing worth building.

**Loners move the way real tables move** -- 1.7% to 10%. Some of that is honest
optimism about hands that might run. Some is mechanical: averaging over sampled
worlds destroys the exact ties that made the God Mode auction decline a loner
worth no more than the same call four-handed.

### The EV of one hand

`hand_ev.py` is the other shape of the question. `pimc_sweep.py` asks what a
PIMC sim table does in general; this asks what **one pinned hand** is worth to
the seat holding it. `game.deal_around` pins your five cards, the up-card, your
seat and the dealer, deals the other three seats at random, and `table.play_deal`
plays every sampled layout out with four `PIMCPlayer`s.

The difference from the God Mode answer is that the auction is *walked* rather
than solved, so the reported EV includes the deals where the hand gets passed
out, ordered up by somebody else, or over-called -- which is why the report
breaks the mean down by who ended up with the contract.

**Two nested sim counts, and they do different jobs.** `--deals` is the outer
loop, the total hand sims, and it is the only one the error bar is on: outcomes
run -4..+4 with a standard deviation near 2, so the 95% interval is about
4/sqrt(deals). `--player-eval-sims` is how many worlds each player imagines per
decision; it is never averaged into the reported mean, so it moves the mean
itself rather than shrinking its interval. How far it has to go before PIMC
decisions settle is **unmeasured** -- do not assume flatness in either
direction.

Cost is `deals x player_eval_sims` and nothing else. Measured on
`JS AS 9H 9D TC` with `9S` up, pass model `"zero"`, loners on: 0.39 s per deal
at 10 eval sims, so ~39 ms per sim, linear in both axes. That hand is the
expensive case, since it gets ordered up nearly every deal and so plays all
five tricks. `--workers` spreads deals over processes at a measured ~3x on a
12-thread machine, not the 10x the core count suggests, and each worker pays the
~20 s JIT warmup once. **The answer does not depend on the worker count** --
every deal seeds itself from its own index, so serial and parallel runs agree
exactly, which is the cheapest available check that the parallel path is sound.

`--both` solves the same layouts in God Mode as well and reports the **paired**
difference. Paired because both tables play identical layouts, so deal luck
cancels deal by deal rather than statistically, and a few hundred deals separate
the two where a few thousand would be needed unpaired.

## Archived approaches

`archive/beta_approach/` holds the previous solver: `tree_search.py`,
`n_play_round.py`, `n_branches.py`. They import each other by bare name, so to
run them put that directory on the path rather than treating it as a package:

```python
import sys; sys.path.insert(0, "archive/beta_approach")
from tree_search import definitive_winner
```

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

### Rotating a called suit into the canonical frame

`rotation.py` owns the *natural* card form -- a `Card(suit, rank)` with rank
9-14 and `J = 11` -- which is a different thing from `deck.py`'s vectors. The
vectors already encode a trump call; natural cards do not.

Rotation is **not** a plain suit relabel. The left bower is the jack of the suit
the same colour as trump, so which jack leaves its own suit changes with the
call: spades trump takes JC, but hearts trump takes JD. So the same-colour suit
maps onto the canonical clubs axis (the one with only five cards, because its
jack left), trump maps onto the trump axis, and the two off-colour suits map
onto the hearts and diamonds axes, which keep all six ranks.

The safety property, and what `tests/test_rotation.py` leans on: for every trump
suit the mapping is a **bijection onto `full_euchre_deck`**. If two real cards
ever collided, the solver would solve the wrong position without complaint --
it has no way to know it was handed nonsense. Rotating into spades is the
identity, which keeps `deck.py` and `rotation.py` honest about each other.

`rotation.py` imports numpy and nothing from this repo, deliberately: a front
end can parse, validate and rotate cards without pulling in numba. Compose it
with the solver at the call site:

```python
from rotation import parse_hand, deal_to_engine, HEARTS
from fast_search import definitive_winner
definitive_winner(deal_to_engine(hands, HEARTS), starting_player=0, caller=0)
```

### The dealt state

`game.Deal` is the deal as a real game has it: four hands of natural cards, the
up-card, the buried kitty, and the dealer seat. `dealer.py` deals 20 canonical
vectors and drops the other four -- fine for solving trick-play with trump
already fixed, but it cannot support bidding, because there is no named up-card
to order and no dealer to pick up and discard. `game.py` is what bidding will be
built on; `dealer.py` still backs the existing God Mode sweep.

`Deal` is frozen, and every transition returns a new one that has been through
`check()`. The invariant is that all 24 cards are always accounted for --
hands + buried + the up-card if it is still on the kitty. `dealer.py` shipped a
card-losing bug twice, so this is asserted rather than assumed.

`deal_around(known_hand=..., seat=..., up_card=...)` is the shape of the
question the calculator answers: pin what you can see, deal the rest uniformly.
It is the natural-card replacement for `generate_hands`' `stack` / `up_card`.

Note that `pick_up` keeps `up_card` on record after the dealer takes it, since
every seat saw it and the bidding depends on that; `picked_up` says whether it
is in a hand or on the kitty, and `all_cards` reads it accordingly.

### Bidding

`bidding.solve_bidding(deal)` solves the whole auction with every seat seeing
every hand. It is the baseline against which heuristic bidders get measured,
not the destination -- replace the decision rule, keep the machinery.

The tree is a chain, not an exponential: round one is four order-or-pass
decisions and an order ends it, round two is four name-or-pass decisions. At
most 36 God Mode solves per deal (24 in round one, since ordering up makes
the dealer choose among six discards, plus 12 in round two), so about 10-25 ms.

Everything is scored as **net points to team 0**, so calls by different seats
can be compared on one scale. `net_to_team0` converts from
`definitive_winner`'s caller's-perspective answer; `value_to(seat, v)` converts
back for display. A sign error here hides on any deal where the teams agree, so
it is tested directly.

Three things that are easy to get wrong and are deliberate here:

- **The dealer chooses the discard, not the caller.** When the opposition orders
  it up, the dealer is picking up for a contract they want to fail and pitches
  accordingly. `tests/test_bidding.py` pins a deal where that costs the caller a
  march -- +1 instead of +2.
- **Passing is not free.** Its value is whatever the rest of the auction
  produces, which may be worse than the call you declined.
- **Ties resolve to passing.** Otherwise God Mode cheerfully orders up a hand
  it knows will be euchred whenever declining is equally bad; the value is the
  same but the reported line is nonsense.

Two measured facts worth knowing: God Mode essentially **never passes out** (0
of 1600 auctions), because somebody can nearly always find a call that is at
worst harmless. And `stick_the_dealer=True` changes the result on about 4% of
deals -- it bites through the *threat*, by changing what earlier seats do.

#### Loners in the auction

`solve_bidding(deal, allow_loners=True)` adds "and alone" as a separate option
beside every call; `order_up` and `name_suit` take `alone=` directly. It is
**off by default**, so every existing four-handed measurement stays comparable.
Options are listed pass, call, call-alone, and `_best` keeps the first of
equals, so a loner worth no more than the same call four-handed is declined --
same reasoning as ties resolving to passing.

Three things worth knowing before touching it:

- **It barely matters in God Mode.** Allowing loners changes the auction on ~1%
  of deals (6 of 480 measured), and always the same way: a made contract becomes
  a lone march. That follows from the scoring -- going alone
  only gains when the caller can take all five unaided, since 3-4 tricks is `+1`
  either way and a euchre costs the same `2`. At the eldest seat over 32 deals,
  going alone was better on 0, worse on 8, equal on 24.
- **It is not monotone for either team.** The loner is an extra option for
  *both* sides, so team 0's value moves down on the deals where team 1 is the
  one with the loner. Don't assert a direction.
- **Cost:** ~72 solves per auction instead of 36, but lone solves are ~3x
  cheaper, so the wall clock goes up by roughly a third, not double.

`order_up` short-circuits one case: if the caller goes alone and the **dealer is
the partner sitting out**, the dealer picks up into a hand that never plays, so
all six discards are worth exactly the same and the choice is unobservable. It
solves one (pitching the up-card, by convention) rather than six. `test_loners.py`
checks that the six really do agree rather than taking it on trust.

`first_bid_options(deal)` is the front-end shape of the question: `{"pass",
"order", "order alone"}`, all on the first bidder's own team's scale, so the
largest number is the best bid. `first_bid_choice` remains the two-option form.

## Repo notes

- `fast_search.solve` / `solve_line` validate their arguments through `_validate`. They used to take anything: `starting_player=6` indexed straight past the end of the `(4,)` and `(4, 5)` state arrays and **segfaulted the interpreter** (njit has no bounds checking), while a hand with a card count other than 5 returned `-1000` from `solve` and wrote past the end of `solve_line`'s trick buffers. `solve_line` now sizes those buffers with `_ALL` rather than `hands.shape[1]`. `_final` and the forced-outcome cutoffs still hardcode 5 tricks and 3 needed, which is why anything but a 5-card hand is rejected outright rather than solved.
- `Dealer.stack_deck` appends to a player's hand instead of replacing it. Replacing meant the second call dropped the first call's cards while leaving them removed from the deck, so they were dealt to nobody -- `generate_hands` hit this whenever `stack` and `up_card` named the same player, and the hand then played out with 17 live cards. It also now validates the seat, the card shape (accepting a bare `(2,)` card), and that the stack cannot overfill a 5-card hand; the last used to surface as `ValueError: Negative dimensions are not allowed` from inside `np.random.choice`.
- `Dealer.__post_init__` honours `players` rather than always building four hands, and rejects a table the deck cannot seat.
- `Dealer.stack_deck` used `np.isin(self.deck, stack_cards).all(axis=1)`, which compared each *coordinate* against every value in the stack rather than matching whole cards, so it silently deleted extra cards from the deck (stacking 9d `[9,0]` and Ac `[0,-14]` also removed Ah `[-14,0]`). Those cards then could not be dealt to anyone. It now matches rows and raises if it does not match exactly `len(stack_cards)` cards. Measured effect on one affected stack: EV moved from +1.530 to +1.608 over 5000 deals, non-overlapping CIs.
- `test_hand.txt` is the canonical fixture, also inlined in `tests/test_fast_search.py` and `tests/test_solver.py`. Its true value is 2 with `starting_player=2, caller=0`, and -2 solved alone.
- `interface.ipynb` is deliberately one worked example, four cells: a hand dealt with `deal_from_order`, the auction solved in God Mode with `allow_loners=True`, and the play. The deal is written out by hand rather than seeded, because the example only works if the loner is obvious -- seat 0 holds both bowers plus A-K of trump and an outside ace, and `first_bid_options` reads pass +2 / order +2 / order alone +4. It is not a dashboard: anything that sweeps deals or measures EV belongs in a script or the tests, where it runs headless and gets checked. Earlier versions grew EV sweep cells; `git log -p -- interface.ipynb` has those if one is wanted back.
