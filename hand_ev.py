"""
EV of one specific hand, played out by a table that cannot see it.

The existing answer to "what is this hand worth" is God Mode: pin your five
cards and the up-card, deal the other seats at random a few thousand times, and
solve each deal exactly. That is the *correct* value of the hand against
opponents who can see it, which is nobody. The number it gives is the price of
a hand at a table that does not exist.

This asks the same question of a real-ish table. Every sampled deal is **played
out** by four PIMC sim players -- each seeing only its own five cards, the
up-card and the play so far -- through the same referee `pimc_sweep.py` uses.
The auction is walked one seat at a time rather than solved, so the hand can be
passed out, over-called, or ordered up by somebody else, and the score that
comes back is what happened at the table rather than what was available at it.

    python hand_ev.py "JS AS 9H 9D TC" --up 9S --seat 0 --dealer 3
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --deals 2000 --player-eval-sims 50
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --both --deals 500
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --deals 5000 --workers 8
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --engine python   # the old path
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --deals 10000         --player-eval-sims 10000 --epsilon 0.4 --workers 10
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --epsilon none   # exact, slow
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --assume order    # if I order it
    python hand_ev.py "JS AS 9H 9D TC" --up 9S --seat 2 --assume order         --no-let-auction-play                 # ...ignoring who bids first

**Walking the auction and conditioning on a bid are different questions.** By
default the asking seat bids for itself, so the reported mean mixes the deals it
called with the ones it passed and somebody else called -- that is what a hand
is worth *at a table*. `--assume order` pins the opening bid instead and prices
**that bid**.

With `--assume`, the auction still runs and the bid is pinned only if it reaches
the seat: from eldest it always does, from a later seat an earlier player can
call first and take the option away. That is reported rather than hidden, as
three numbers with no headline among them -- how often the option arrived, what
the call returned when it did, and what the hand returned across all deals. How
often you get to bid is as much a part of a hand's worth as what the bid pays.

`--no-let-auction-play` skips the auction entirely and prices the call every
deal, which is the only way to compare seats on equal footing, since otherwise
a late seat's number is contaminated by how often it gets preempted. It needs
`--assume order` or `--assume order-alone`; there is nothing to price without
one, and it is incoherent for `--assume pass`, where passing *is* the auction
continuing.

Two nested sim counts, doing different jobs:

  * `--deals` is how many layouts of the *other* seats get dealt -- the total
    hand sims. This is what the reported interval measures: outcomes run -4..+4
    with a standard deviation near 2, so the 95% interval is about
    4/sqrt(deals) -- ~0.13 at 1,000 deals, ~0.04 at 10,000.
  * `--player-eval-sims` is how many worlds each player imagines per decision.
    It does not tighten that interval, because it is not averaged into the
    reported mean at all -- it changes what the players *do*, which moves the
    mean itself rather than its error bar. How far it has to go before the
    decisions settle is not measured here, and is not safe to assume from the
    God Mode numbers elsewhere in this repo.

**Cost used to be `deals x player_eval_sims`, and no longer is.** A PIMC
decision averages every option over N sampled worlds and takes the best, but
the average is not the answer -- the argmax is, and that is usually settled
long before N. `--epsilon` stops sampling a decision once its remaining
options are within that many points of the leader, on the grounds that picking
either one moves the answer by less than the band. See `players._race`.

What that changes is the shape of the cost, not just its size: a decision is
capped at roughly `(z * sd / epsilon)^2` worlds, so past that point a larger
budget buys nothing and costs nothing. Measured on `JS AS 9H 9D TC` with `9S`
up, pass model `"zero"`, loners on, 6 deals a point:

    eval sims   epsilon 0.15   epsilon 0.40   epsilon 1.00
          200      1.049 s/deal   0.446 s/deal   0.320 s/deal
          800      1.393          0.464          0.330
        3,200      1.435          0.461          0.337
       12,800      1.429          0.457          0.331

Flat from 800 on. `--player-eval-sims 12800` costs what 800 costs. With
`--epsilon none` the old behaviour is back, exactly -- the same worlds, the
same rng, the same answers -- and so is the old linear cost, 0.39 s per deal
per 10 sims.

So the sweep is priced by `deals` and `epsilon`, and `--player-eval-sims` is
close to free above ~800.

**And then it was compiled.** `--engine fast` (the default) plays the deals in
`fastsim.py`, which is this file's model with `table.py`, `players.py`,
`observation.py` and `bidding.py` rewritten as integers and njit, over the
bitboard solver in `bitcore.py`. It is not a different model and not an
approximation -- the sampling, the decisions, the pass models, the stopping
rule and the tie-breaks are the ones `players.py` documents, and
`tests/test_fastsim.py` holds the two implementations to the same answers.
`--engine python` is the old path, kept because it is the readable one and the
one the unit suite drives.

Measured on `TH AS AD KD JD` with `9H` up, seat 2, dealer 0, `--assume order`,
today's default budgets at `--epsilon 0.05`, ten workers:

    engine   deals    wall       per deal    EV over all deals
    python   4,000    10.9 min   0.163 s     +0.926 +/- 0.036
    fast    10,000    20 s       0.0020 s    +0.912 +/- 0.023

**82x per deal, and the same answer** -- the two means differ by 0.014 against
a combined interval of 0.043, and the two engines call the hand at 88.3% and
88.8% and are euchred on 11.4% and 12.0% of what they call. They do not agree
deal by deal and cannot: `random.Random` does not exist inside njit, so the
compiled engine carries its own splitmix64 and imagines different layouts. What
matches is the distribution, which is the only thing this script reports.

A second check on a question that exercises far more of the code -- the whole
auction walked rather than a pinned bid, so every seat bids, round two runs,
loners are on the table and hands get passed in:

    `JS AS 9H 9D TC`, `9S` up, seat 0, dealer 3, eleven workers

    engine   deals    per deal    EV to your team    you called   euchred
    python    1,500   0.2615 s    -0.445 +/- 0.078   70.3%        52.1%
    fast     20,000   0.0022 s    -0.487 +/- 0.021   70.2%        52.2%

119x here, because this hand spends its time bidding -- 349 sampled worlds a
deal against 50 -- and bidding is where the value bounds on the score pay
most. The two means differ by 0.042 against a
combined interval of 0.081, and the auction profiles line up to a fraction of
a point: partner calls 4.7% and 4.8%, opponents 25.0% and 24.8%, spades named
88.0% and 86.6%. The compiled engine also passes 28 of 20,000 deals in, which
is the 0.2% rate CLAUDE.md had to amend a 0-of-60 claim down to.

The old cost table, which is what `--engine python` still runs at:

    deals   eval sims   epsilon   serial        10 workers
   10,000     default      0.05   ~2.8 hours    ~34 minutes
    1,000      10,000      0.40   ~8 minutes    ~2 minutes
   10,000      10,000      0.40   ~76 minutes   ~20 minutes
   10,000      10,000      0.15   ~4 hours      ~52 minutes
   10,000      10,000      none   ~45 days      ~10 days

**The first run after a checkout or an edit compiles**, which takes about 80
seconds and is then cached on disk for every run after it. The 22 seconds above
is a warm run; the cold one is about a hundred. Editing either module
invalidates the cache -- including editing only `bitcore.py`, which numba would
not notice on its own; see `fastsim._drop_stale_cache`.

The band is not free, and what it costs was measured rather than assumed: over
900 paired deals at 400 eval sims, `--epsilon 0.15` moved the answer by
+0.073 +/- 0.077 points and `--epsilon 0.40` by +0.024 +/- 0.085, against the
exact run on identical layouts. Small next to a euchre, and not clearly
different from zero at that many deals.

**But a wide band caps the sample count as well as the cost.** At
`--epsilon 0.4`, going from 800 eval sims to 12,800 changed nothing whatsoever
-- 900 of 900 deals identical. So that run is an ~800-sample run that cannot be
told apart from a 12,800-sample one *at that band*, which is not the same claim
as the decisions having settled. Narrow the band if the question is whether
they do.

`--workers` is threads under the compiled engine and processes under the Python
one. Neither scales with the core count. Over 4,000 deals on a 12-thread
machine the compiled engine ran 4,000 deals in 56.7 s on one thread, 16.4 s on
four, 9.2 s on ten and 8.2 s on twelve -- 6.9x at the top, not 12x. The process pool measured
4.6x. Results do not depend on the worker count -- every deal seeds itself from
its own index -- so a parallel run and a serial one give the same number, which
is the cheapest available check that the parallel path is sound. The *node*
count does wobble by a handful, because threads share one transposition table
and race for its slots; what they find there is always a true value for the
position asked about, so the wobble is in how much work gets done and never in
what comes out.

`--tt-bits` sizes the compiled engine's transposition table, which every thread
shares and nothing ever clears. On the run above, 2^24 slots (134 MB) takes
23.1 s, 2^25 (268 MB) 21.6 s, 2^26 (537 MB) 20.3 s and 2^27 (1.1 GB) 19.4 s --
a flat curve, and it was not always: before the value bounds went into the
search it was 91 s at 2^24 against 47 s at 2^26. The default scales with the
sweep, stops at 2^26, and never takes more than an eighth of the machine.

`--both` also solves each of the same deals in God Mode and reports the paired
difference. Paired, because the two tables play identical layouts: the
difference has far less variance than either mean, so a few hundred deals
separate them where a few thousand would be needed unpaired.
"""
import argparse
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import bidding as b
import game
import players
import rotation as r
import table as t

DEALS = 1000

# What the asking seat is assumed to do with its first bid. AUCTION walks the
# auction and lets the seat decide, which mixes the deals it called with the
# ones it passed; the others condition on one opening bid so the reported mean
# is the value of *that bid*. See players.ForcedOpeningBid.
ASSUME_AUCTION = "auction"
ASSUME_ORDER = "order"
ASSUME_ALONE = "order-alone"
ASSUME_PASS = "pass"
# Round two. The seat passes the up-card, and on the deals where every seat
# passes it and the auction comes back round, names `Setup.suit` -- alone, or
# not -- or passes again. Only those deals are the question; the rest are
# recorded as not reached and, on the compiled engine, not played.
ASSUME_NAME = "name"
ASSUME_NAME_ALONE = "name-alone"
ASSUME_PASS_TWICE = "pass-twice"
ROUND_ONE_ASSUMPTIONS = (ASSUME_ORDER, ASSUME_ALONE, ASSUME_PASS)
ROUND_TWO_ASSUMPTIONS = (ASSUME_NAME, ASSUME_NAME_ALONE, ASSUME_PASS_TWICE)
ASSUMPTIONS = ((ASSUME_AUCTION,) + ROUND_ONE_ASSUMPTIONS
               + ROUND_TWO_ASSUMPTIONS)

# Defaults per decision kind, set to the measured mean number of sampled worlds
# each needs before its argmax stops moving -- 40,390 decisions, see
# notes/settle_counts.md. At these budgets 99.4-100% of decisions with a real
# margin (>0.15) keep the leader they finish with; the ones that drift are
# near-ties, where either answer is worth the same by construction.
# Canonical home is players.py -- these are a property of the player, not of
# this script. Unlike the PIMC sim scripts, hand_ev adopts them as its default.
PLAYER_EVAL_SIMS = players.RESEARCHED_PLAY
BID_EVAL_SIMS = players.RESEARCHED_BID
DISCARD_EVAL_SIMS = players.RESEARCHED_DISCARD
EPSILON = 0.05


def _epsilon(text):
    """`--epsilon none` turns early stopping off; anything else is a number."""
    if text.strip().lower() in ("none", "off"):
        return None
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("epsilon must not be negative")
    return value


# ------------------------------------------------------------- the question


@dataclass(frozen=True)
class Setup:
    """
    Everything that pins the question down. Picklable, so workers get a copy.

    `hand` and `up_card` are what the asker can see; `seat` is where they are
    sitting and `dealer` says who deals, which fixes the bidding order and so
    how many seats speak before them.
    """

    hand: Tuple[r.Card, ...]
    up_card: Optional[r.Card]
    seat: int
    dealer: int
    player_eval_sims: int
    bid_eval_sims: int
    discard_eval_sims: int
    pass_model: str
    allow_loners: bool
    stick: bool
    seed: int
    prune_discards: bool = False
    assume: str = ASSUME_AUCTION
    # True: run the auction and pin the opening bid only if it reaches this
    # seat, so an earlier caller can take the option away. False: skip the
    # auction entirely and price the call from this seat regardless.
    let_auction_play: bool = True
    # Last, and defaulted, so the field order the tests build a Setup with
    # keeps working. None means exact averaging -- see players._race.
    epsilon: Optional[float] = None
    # Which implementation plays the deals out. They are the same model; see
    # `run_fast`. "python" is the readable one and the one the unit suite
    # drives; "fast" is the compiled one, and is ~200x quicker.
    engine: str = "fast"
    # The suit named under ASSUME_NAME / ASSUME_NAME_ALONE, as a natural suit
    # index. Never the up-card's suit, which round two forbids.
    suit: Optional[int] = None

    @property
    def round_two(self) -> bool:
        return self.assume in ROUND_TWO_ASSUMPTIONS

    def deal(self, i: int) -> game.Deal:
        """Deal `i` of the sweep. Derived from `i` alone, so workers agree."""
        return game.deal_around(known_hand=list(self.hand), seat=self.seat,
                                up_card=self.up_card,
                                rng=random.Random(self.seed + i),
                                dealer=self.dealer)

    def table(self, i: int):
        """Four PIMC sim players, seeded per deal and per seat."""
        table = [players.PIMCPlayer(samples=self.player_eval_sims,
                                    bid_samples=self.bid_eval_sims,
                                    discard_samples=self.discard_eval_sims,
                                    pass_model=self.pass_model,
                                    epsilon=self.epsilon,
                                    prune_discards=self.prune_discards,
                                    rng=random.Random(self.seed * 7919 + i * 4 + s))
                 for s in range(game.PLAYERS)]
        if self.round_two:
            table[self.seat] = players.ForcedRoundTwoBid(
                table[self.seat],
                t.PASS if self.assume == ASSUME_PASS_TWICE else t.NAME,
                suit=self.suit, alone=(self.assume == ASSUME_NAME_ALONE))
        elif self.assume != ASSUME_AUCTION and self.let_auction_play:
            action = t.PASS if self.assume == ASSUME_PASS else t.ORDER
            table[self.seat] = players.ForcedOpeningBid(
                table[self.seat], action,
                alone=(self.assume == ASSUME_ALONE))
        return table


@dataclass(frozen=True)
class Record:
    """One deal's outcome, on the asking seat's own team's scale. Picklable."""

    value: int                       # points to the asker's team
    caller: int                      # -1 if the deal was passed out
    trump: int
    alone: bool
    caller_tricks: int
    caller_score: int
    # Did the --assume pin actually fire? It cannot when an earlier seat has
    # already ended the auction, which is impossible from the eldest seat and
    # possible from any other. A sweep that averaged those in would be
    # answering a different question than the one asked, quietly.
    forced: bool = True

    @property
    def passed_out(self) -> bool:
        return self.caller < 0


PASSED_OUT = Record(0, -1, -1, False, 0, 0)
# A deal the pinned bid never reached, and so was not played.
NOT_REACHED = Record(0, -1, -1, False, -1, 0, forced=False)


def role_of(caller: int, seat: int) -> str:
    """Who ended up with the contract, said relative to the asking seat."""
    if caller < 0:
        return "passed out"
    if caller == seat:
        return "you called"
    if caller % 2 == seat % 2:
        return "partner called"
    return "opponent called"


ROLES = ("you called", "partner called", "opponent called", "passed out")


def role_label(role: str, setup: "Setup") -> str:
    """
    `role_of`'s answer, said in a way that survives `--assume`.

    Under `--assume pass` the asking seat can still end up with the contract,
    and the line saying so reads like a contradiction until you know that
    **only the opening bid is pinned** -- `players.ForcedOpeningBid`. The seat
    passes on the up-card, the up-card is turned down, and round two comes back
    around to a seat that is now bidding for itself.

    Which round that call came in is not guessed at, it is forced: the pinned
    bid is the seat's first, the seat's first bid is in round one, and an
    auction that ends before reaching the seat ends without the seat calling
    at all. So every call it makes under this assumption is a round-two call.
    """
    if role == "you called" and (setup.assume == ASSUME_PASS
                                 or setup.round_two):
        return "you called (round two)"
    return role


# ------------------------------------------------------------- playing them


def play_one(setup: Setup, i: int) -> Record:
    """One sampled layout, played out by four PIMC sim players."""
    table = setup.table(i)
    if setup.assume in (ASSUME_ORDER, ASSUME_ALONE) and not setup.let_auction_play:
        # No auction at all, so the call always happens by construction.
        result = t.play_pinned_order(setup.deal(i), table, setup.seat,
                                     alone=(setup.assume == ASSUME_ALONE))
        fired = True
    else:
        result = t.play_deal(setup.deal(i), table,
                             stick_the_dealer=setup.stick,
                             allow_loners=setup.allow_loners)
        fired = getattr(table[setup.seat], "forced", True)
    if result.passed_out:
        return replace(PASSED_OUT, forced=fired)
    return Record(value=b.value_to(setup.seat, result.value),
                  caller=result.contract.caller,
                  trump=result.contract.trump,
                  alone=result.contract.alone,
                  caller_tricks=result.caller_tricks,
                  caller_score=result.caller_score, forced=fired)


def solve_one(setup: Setup, i: int) -> Record:
    """The same layout in God Mode -- the old baseline, for contrast."""
    out = b.solve_bidding(setup.deal(i), stick_the_dealer=setup.stick,
                          allow_loners=setup.allow_loners)
    if out.passed_out:
        return PASSED_OUT
    return Record(value=b.value_to(setup.seat, out.value),
                  caller=out.contract.caller,
                  trump=out.contract.trump,
                  alone=out.contract.alone,
                  caller_tricks=-1,          # solve_bidding does not report them
                  caller_score=b.value_to(out.contract.caller, out.value))


# Each worker's own copy of the question, set once when the pool starts. A
# Setup is small, but pickling it per deal would dominate a cheap sweep.
_SETUP = None


def _init(setup: Setup):
    global _SETUP
    _SETUP = setup


def _work(job):
    i, want_god = job
    return i, play_one(_SETUP, i), (solve_one(_SETUP, i) if want_god else None)


def run(setup: Setup, deals: int, both: bool = False, workers: int = 1,
        progress=None):
    """
    Play `deals` sampled layouts. Returns (pimc records, god mode records),
    both in deal order.

    Sorted back into deal order on the way out, because the pool hands results
    back as they finish. The means and counts would not care, but a caller that
    reads `records[i]` as deal `i` would be quietly wrong, and so would any
    later attempt to compare a parallel run against a serial one record by
    record. Serial below two workers: a worker pays the JIT warmup on the way
    in and a short sweep never earns that back.
    """
    jobs = [(i, both) for i in range(deals)]
    out = []
    done = 0

    if workers > 1:
        with mp.Pool(workers, initializer=_init, initargs=(setup,)) as pool:
            for row in pool.imap_unordered(_work, jobs, chunksize=4):
                out.append(row)
                done += 1
                if progress:
                    progress(done, deals)
    else:
        _init(setup)
        for job in jobs:
            out.append(_work(job))
            done += 1
            if progress:
                progress(done, deals)

    out.sort(key=lambda row: row[0])
    return ([a for _, a, _ in out],
            [c for _, _, c in out if c is not None])



# ---------------------------------------------------------- the fast engine
#
# `fastsim` is this file's whole sweep compiled -- deal, auction, sampling,
# search and all. It is a second implementation of the same model rather than a
# different one, and `tests/test_fastsim.py` is what holds the two together:
# the compiled solver against `fast_search`, the compiled auction against
# `bidding.solve_bidding`, the compiled inference against `observation.py`, and
# the whole thing against the Python table with an error bar.
#
# The two do **not** agree deal by deal. They draw their worlds from different
# random number generators -- `random.Random` does not exist inside njit -- so
# deal 7 sees different imagined layouts under each, and lands where it lands.
# What matches is the distribution, which is the only thing the sweep reports.

FAST = "fast"
PYTHON = "python"
ENGINES = (FAST, PYTHON)

# How big a transposition table to give the compiled engine. It is shared by
# every thread and never cleared, and it is the single biggest thing between
# this script and its old runtime.
#
# There is no size to choose, because bigger is better everywhere and costs
# nothing. 2^27 buys 4% over 2^26 and a whole gigabyte, so 2^26 is the ceiling;
# below it, measured on `TH AS AD KD JD` with `9H` up through `api.evaluate`
# -- three actions, ten threads, warm table, ms per deal:
#
#      deals     2^24     2^26    ratio
#        250     4.81     3.59    1.34x
#      1,000     8.28     3.75    2.21x
#      4,000    11.07     6.29    1.76x
#     10,000    10.00     7.50    1.33x
#
# 2^26 wins at every one of them. The ratio is not monotone in the deal count
# and no trend should be read off it -- an earlier version of this comment did
# exactly that and was wrong. The allocation is lazily-zeroed pages and costs
# nothing measurable up front (0.00 s for 2^26), so there is nothing on the
# other side of the trade.
#
# This used to scale with `deals`, which left a 1,000-deal query on 2^24 and
# cost it 2.2x for no saving. So the only question left is what the machine
# can spare: no more than an eighth of physical memory, which is still a lot
# to ask quietly. `describe` prints what it took and `--tt-bits` overrides it
# in either direction.
TT_MIN_BITS = 22
TT_MAX_BITS = 26
TT_MEMORY_SHARE = 8

# Pieces of work per thread, for load balance. See `run_fast`.
CHUNKS_PER_THREAD = 4

# Progress reporting splits the sweep into blocks, and every block ends at a
# barrier. Two numbers keep that from costing more than it reports: how many
# updates to aim for, and how much work each chunk must have inside a block
# before a barrier is worth paying for. Without the second one a small sweep
# collapses -- 1,000 deals over 40 chunks in 20 blocks is barely one deal per
# chunk per block, and the threads spend their time waiting for each other.
# Measured on 1,000 deals at ten threads: 13.5s that way against 5.6s with
# blocks this size, for a run that takes 5.3s with no progress at all.
PROGRESS_BLOCKS = 20
DEALS_PER_CHUNK_PER_BLOCK = 8


def system_memory():
    """Physical memory in bytes, or None where we cannot find out."""
    try:
        import os
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


def default_tt_bits() -> int:
    """
    How big a transposition table to allocate: as much as the machine spares.

    Deliberately not a function of the sweep size. 2^26 is faster at every
    sweep size measured and the allocation costs nothing up front, so there is
    nothing to trade off -- see the table above the constants. Falls back to
    the floor rather than guessing when the memory size cannot be read: a
    table that is too small costs time, and one that is too big costs the user
    their machine.
    """
    ram = system_memory()
    if not ram:
        return TT_MIN_BITS
    bits = TT_MIN_BITS
    while bits < TT_MAX_BITS and (1 << (bits + 1)) * 8 <= ram // TT_MEMORY_SHARE:
        bits += 1
    return bits


def compiled_already() -> bool:
    """
    Is the compiled engine's on-disk cache warm, or is this run paying for it?

    A cold run compiles for about eighty seconds before it plays a single
    deal, which looks like a hang rather than like a build. Cheap to check and
    worth saying out loud -- and a false answer either way costs nothing but
    the line.
    """
    import os

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "__pycache__")
    try:
        return any(name.startswith("fastsim.") and name.endswith(".nbi")
                   for name in os.listdir(cache))
    except OSError:
        return False


def _card_id(card: r.Card) -> int:
    """A natural card as `fastsim` numbers it: suit * 6 + rank."""
    return card.suit * 6 + (card.rank - r.NINE)


def codes(fastsim):
    """
    The integers `fastsim` names these by, keyed by the strings this file uses.

    Read off the module rather than written out, because a literal here that
    drifted from the constant there would not fail -- it would quietly run a
    different pass model, or condition on a different opening bid, and report
    the answer to a question nobody asked.
    """
    return ({players.PASS_GOD_MODE: fastsim.PASS_GOD,
             players.PASS_ZERO: fastsim.PASS_ZERO,
             players.PASS_FLOOR: fastsim.PASS_FLOOR,
             players.PASS_GUARD: fastsim.PASS_GUARD},
            {ASSUME_AUCTION: fastsim.ASSUME_AUCTION,
             ASSUME_ORDER: fastsim.ASSUME_ORDER,
             ASSUME_ALONE: fastsim.ASSUME_ALONE,
             ASSUME_PASS: fastsim.ASSUME_PASS,
             ASSUME_NAME: fastsim.ASSUME_NAME,
             ASSUME_NAME_ALONE: fastsim.ASSUME_NAME_ALONE,
             ASSUME_PASS_TWICE: fastsim.ASSUME_PASS_TWICE})


def _from_row(row, setup: Setup, god: bool) -> Record:
    """One row of the compiled sweep's output array, as a `Record`."""
    import fastsim

    caller = int(row[fastsim.R_CALLER])
    if caller < 0:
        return replace(PASSED_OUT, forced=bool(row[fastsim.R_FORCED]))
    if god:
        # `fastsim.solve_bidding` reports net points to team 0, the scale
        # `bidding` works on; everything below is on the asker's own scale.
        value = int(row[fastsim.R_VALUE])
        return Record(value=b.value_to(setup.seat, value), caller=caller,
                      trump=int(row[fastsim.R_TRUMP]),
                      alone=bool(row[fastsim.R_ALONE]),
                      caller_tricks=-1,
                      caller_score=b.value_to(caller, value))
    return Record(value=int(row[fastsim.R_VALUE]), caller=caller,
                  trump=int(row[fastsim.R_TRUMP]),
                  alone=bool(row[fastsim.R_ALONE]),
                  caller_tricks=int(row[fastsim.R_TRICKS]),
                  caller_score=int(row[fastsim.R_SCORE]),
                  forced=bool(row[fastsim.R_FORCED]))


def new_table(bits: Optional[int] = None):
    """
    A transposition table a caller can hold on to across queries.

    `run_fast` makes one per call otherwise. A long-lived process should make
    one at boot and pass it in -- it is never cleared, so entries from the last
    query are still true for the next one and the table only gets warmer.
    """
    import numpy as np

    return np.zeros(1 << (bits if bits is not None else default_tt_bits()),
                    dtype=np.int64)


def run_fast(setup: Setup, deals: int, both: bool = False, workers: int = 1,
             progress=None, tt_bits: Optional[int] = None, table=None,
             start: int = 0):
    """
    The sweep, compiled. Same arguments and same answers as `run`.

    `workers` is threads here rather than processes, and it is also how the
    deals are split: thread `c` takes every `workers`-th one. The deal is a
    function of its index, so **the answer does not depend on how many there
    are** -- the same invariant the process pool had, and the cheapest check
    that the parallel path is sound.

    Returns `(pimc records, god records, spent)`, where `spent` is the summed
    per-thread tallies -- nodes searched, and the worlds and solves each kind
    of decision paid for. `report_spend` prints them; they are the numbers to
    look at first when a sweep is slower than it should be.

    `table` is a transposition table from `new_table` to use instead of
    allocating one. It is never cleared and its entries are keyed by
    everything their value depends on, so sharing it across queries is both
    safe and faster -- see "The compiled engine" in CLAUDE.md. Pass one from a
    server; leave it None and each call pays for its own.

    `prune_discards` is not supported: it is an unproven axiom, it buys 1.09x,
    and this engine is already three orders of magnitude past what it was
    worth. Ask for it and you get the Python path.

    `start` plays deals `start .. start + deals - 1` instead of from zero, so
    a sweep can be extended without replaying -- or reseeding -- what it has
    already played. Deal `i` is the same deal whichever call plays it.
    """
    import numpy as np
    import numba
    import fastsim

    threads = max(1, min(int(workers), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(threads)
    # More pieces than threads, because deals are not equally expensive: a
    # hand that gets passed out costs a fraction of one that plays five
    # tricks, and a thread that draws a run of cheap ones would otherwise
    # finish early and wait at the block boundary.
    chunks = threads * CHUNKS_PER_THREAD

    pin = 0
    for card in setup.hand:
        pin |= 1 << _card_id(card)
    up = _card_id(setup.up_card) if setup.up_card is not None else -1

    pass_models, assumptions = codes(fastsim)
    tt = table if table is not None else new_table(tt_bits)
    # Rows are indexed by deal number, so the array covers everything below
    # `start` too. Seven int64s a deal; not worth an offset through the njit.
    end = start + deals
    out = np.zeros((end, fastsim.RECORD), dtype=np.int64)
    god = np.zeros((end if both else 1, fastsim.RECORD), dtype=np.int64)
    counters = np.zeros((chunks, fastsim.COUNTERS), dtype=np.int64)

    # Run in blocks so a progress line has something to say -- the table
    # carries over between them, which is the point of hoisting it out here.
    # Each block ends in a barrier, though, so a quiet run does not pay for
    # one it would never read, and a reported one gets blocks big enough to
    # keep every thread busy inside them.
    block = deals if progress is None else max(
        chunks * DEALS_PER_CHUNK_PER_BLOCK, -(-deals // PROGRESS_BLOCKS))
    for lo in range(start, end, block):
        hi = min(lo + block, end)
        fastsim.run_deals(
            lo, hi, setup.seed, pin, setup.seat, up, setup.dealer,
            setup.player_eval_sims, setup.bid_eval_sims,
            setup.discard_eval_sims,
            0.0 if setup.epsilon is None else float(setup.epsilon),
            setup.epsilon is not None, players.MIN_WORLDS,
            pass_models[setup.pass_model], assumptions[setup.assume],
            -1 if setup.suit is None else int(setup.suit),
            1 if setup.let_auction_play else 0,
            1 if setup.stick else 0, 1 if setup.allow_loners else 0,
            chunks, tt, out, god, 1 if both else 0, counters)
        if progress:
            progress(hi - start, deals)

    return ([_from_row(out[i], setup, False) for i in range(start, end)],
            [_from_row(god[i], setup, True) for i in range(start, end)]
            if both else [],
            counters.sum(axis=0))


def run_fast_listed(setup: Setup, indices, workers: int = 1, progress=None,
                    tt_bits: Optional[int] = None, table=None):
    """
    `run_fast` over chosen deal numbers rather than a range: deal `i` is
    played exactly as `run_fast` would play it, and the records come back in
    the order `indices` gives them.

    Exists for round two. Which deals come back round is fixed by the auction
    before the seat's bid, so it is the same for every action; once one action
    has found them, the rest need play only those. On a hand the bidding gets
    back to one deal in fifteen, that is most of the cost of every action
    after the first.

    Returns `(records, spent)`.
    """
    import numpy as np
    import numba
    import fastsim

    threads = max(1, min(int(workers), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(threads)
    chunks = threads * CHUNKS_PER_THREAD

    pin = 0
    for card in setup.hand:
        pin |= 1 << _card_id(card)
    up = _card_id(setup.up_card) if setup.up_card is not None else -1

    pass_models, assumptions = codes(fastsim)
    tt = table if table is not None else new_table(tt_bits)
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    out = np.zeros((n, fastsim.RECORD), dtype=np.int64)
    counters = np.zeros((chunks, fastsim.COUNTERS), dtype=np.int64)

    block = n if progress is None else max(
        chunks * DEALS_PER_CHUNK_PER_BLOCK, -(-n // PROGRESS_BLOCKS))
    for lo in range(0, n, max(1, block)):
        hi = min(lo + block, n)
        part = np.zeros((hi - lo, fastsim.RECORD), dtype=np.int64)
        fastsim.run_listed(
            indices[lo:hi], setup.seed, pin, setup.seat, up, setup.dealer,
            setup.player_eval_sims, setup.bid_eval_sims,
            setup.discard_eval_sims,
            0.0 if setup.epsilon is None else float(setup.epsilon),
            setup.epsilon is not None, players.MIN_WORLDS,
            pass_models[setup.pass_model], assumptions[setup.assume],
            -1 if setup.suit is None else int(setup.suit),
            1 if setup.let_auction_play else 0,
            1 if setup.stick else 0, 1 if setup.allow_loners else 0,
            chunks, tt, part, counters)
        out[lo:hi] = part
        if progress:
            progress(hi, n)

    return ([_from_row(out[j], setup, False) for j in range(n)],
            counters.sum(axis=0))


# A sweep told to keep going until enough deals reach the seat stops here
# regardless, so a hand the auction almost never brings back cannot run
# forever. Reported, not hidden: the caller sees how many it got.
MAX_DEALS = 500_000


def run_fast_until(setup: Setup, min_reached: int, deals: int,
                   workers: int = 1, progress=None,
                   tt_bits: Optional[int] = None, table=None,
                   max_deals: int = MAX_DEALS):
    """
    `run_fast`, extended batch by batch until `min_reached` deals have
    reached the seat (`Record.forced`), or `max_deals` have been played.

    `deals` is the first batch. Each batch after it is sized from the reach
    rate seen so far, with a margin, so a hand that comes back one deal in
    twenty takes two or three batches rather than dozens. The deals are
    numbered on from where the last batch stopped, so the result is exactly
    the first N deals of the sweep -- the same records a single
    `run_fast(setup, N)` would give, which is what lets every other action be
    priced over the same N afterwards.

    `progress(fraction)` is called with how far towards `min_reached` the
    sweep is, as a fraction.

    Returns `(records, spent)`.
    """
    records, spent = [], None
    reached = 0
    batch = max(1, min(deals, max_deals))
    while True:
        base = reached

        def note(done, total, base=base, batch=batch, played=len(records)):
            if progress is None:
                return
            # Within a batch, guess the reach by the rate so far; it is only
            # ever a progress bar.
            rate = base / played if played else 1.0 / 20
            guess = base + rate * done
            progress(min(0.99, guess / max(1, min_reached)))

        got, _, cost = run_fast(setup, batch, workers=workers,
                                progress=note if progress else None,
                                tt_bits=tt_bits, table=table,
                                start=len(records))
        records.extend(got)
        spent = cost if spent is None else spent + cost
        reached += sum(1 for rec in got if rec.forced)
        if reached >= min_reached or len(records) >= max_deals:
            break
        rate = reached / len(records)
        want = (min_reached - reached) / rate * 1.05 if rate else batch * 4
        batch = int(min(max(want, 64), max_deals - len(records)))
    if progress is not None:
        progress(1.0)
    return records, spent

# ------------------------------------------------------------- reading them


def interval(values):
    """Mean and a 95% half-width. Same convention as `pimc_sweep.interval`."""
    n = len(values)
    if n < 2:
        return (float(values[0]) if values else 0.0), 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * (var / n) ** 0.5


def assumed_verb(setup: Setup) -> str:
    """What `--assume` has the seat do, as a past-tense phrase."""
    if setup.assume == ASSUME_NAME:
        return "named %s (round two)" % r.suit_name(setup.suit)
    if setup.assume == ASSUME_NAME_ALONE:
        return "named %s alone (round two)" % r.suit_name(setup.suit)
    return {ASSUME_ORDER: "ordered",
            ASSUME_ALONE: "ordered alone",
            ASSUME_PASS: "passed round one",
            ASSUME_PASS_TWICE: "passed in both rounds"}[setup.assume]


def report(name: str, records, setup: Setup):
    """What the hand was worth, and what happened to it."""
    n = len(records)
    mean, half = interval([rec.value for rec in records])

    print("  %s" % name)

    if setup.assume == ASSUME_AUCTION:
        print("    %-30s %+.3f +/- %.3f points per deal"
              % ("EV to your team", mean, half))
    else:
        # Three numbers, deliberately with no headline among them. How often
        # the option arrived is as much a part of what the hand is worth as
        # what the call returned when it did: a hand worth +0.4 whenever you
        # get to order it, that only gets the chance a third of the time, is a
        # different proposition from one worth +0.4 every deal.
        verb = assumed_verb(setup)
        got = [rec for rec in records if rec.forced]
        sub_mean, sub_half = interval([rec.value for rec in got])
        print("    %-30s %d of %d (%.1f%%)"
              % ("deals you " + verb, len(got), n, 100.0 * len(got) / n))
        if got:
            print("    %-30s %+.3f +/- %.3f points per deal"
                  % ("EV given you " + verb, sub_mean, sub_half))
        if setup.round_two:
            # The deals the auction never brought back are not part of this
            # question, and the compiled engine does not even play them, so
            # everything below is over the reached ones alone.
            records, n = got, len(got)
            if not n:
                return sub_mean, sub_half
        else:
            print("    %-30s %+.3f +/- %.3f points per deal"
                  % ("EV over all deals", mean, half))

    roles = Counter(role_of(rec.caller, setup.seat) for rec in records)
    print("    %-30s" % "how the auction went")
    for role in ROLES:
        if not roles[role]:
            continue
        subset = [rec.value for rec in records
                  if role_of(rec.caller, setup.seat) == role]
        sub_mean, sub_half = interval(subset)
        print("      %-26s %5d (%4.1f%%)  %+.3f +/- %.3f"
              % (role_label(role, setup), roles[role],
                 100.0 * roles[role] / n, sub_mean, sub_half))

    ours = [rec for rec in records
            if not rec.passed_out and rec.caller % 2 == setup.seat % 2]
    if ours:
        euchred = sum(1 for rec in ours if rec.caller_score < 0)
        alone = sum(1 for rec in ours if rec.alone)
        print("    %-30s %d of %d (%.1f%%)"
              % ("your team euchred", euchred, len(ours),
                 100.0 * euchred / len(ours)))
        print("    %-30s %d of %d (%.1f%%)"
              % ("your team called alone", alone, len(ours),
                 100.0 * alone / len(ours)))

    trumps = Counter(rec.trump for rec in records if not rec.passed_out)
    if trumps:
        print("    %-30s %s"
              % ("trump called",
                 ", ".join("%s %d" % (r.suit_name(s), c)
                           for s, c in trumps.most_common())))
    return mean, half


def report_spend(spent, deals):
    """
    What the compiled engine spent getting there, per deal.

    Not decoration: these are the four numbers that say where a slow sweep is
    slow. Worlds are how many layouts a decision imagined before its argmax
    settled -- well under the budget, which is the stopping rule working --
    and solves are how many whole deals or positions that cost.
    """
    import fastsim

    per = lambda i: spent[i] / max(1, deals)
    print("\n  what it cost, per deal")
    print("    %-30s %d" % ("positions searched",
                            round(per(fastsim.C_NODES))))
    print("    %-30s %.1f worlds, %.1f solves"
          % ("bidding", per(fastsim.C_BID_WORLDS),
             per(fastsim.C_BID_SOLVES)))
    print("    %-30s %.1f worlds, %.1f solves"
          % ("the discard", per(fastsim.C_DISCARD_WORLDS),
             per(fastsim.C_DISCARD_SOLVES)))
    print("    %-30s %.1f worlds over 20 cards"
          % ("the play", per(fastsim.C_PLAY_WORLDS)))


def paired(pimc, god):
    """
    PIMC minus God Mode on the same layouts.

    Paired rather than two independent means: both tables played identical
    deals, so layout luck cancels deal by deal instead of statistically.
    """
    diffs = [a.value - c.value for a, c in zip(pimc, god)]
    mean, half = interval(diffs)
    print("\n  PIMC sim minus God Mode, on the same %d layouts" % len(diffs))
    print("    %-30s %+.3f +/- %.3f points per deal"
          % ("difference", mean, half))
    print("    %-30s %s"
          % ("reading",
             "God Mode over-rates this hand" if mean < -half else
             "God Mode under-rates this hand" if mean > half else
             "no separation at this sample size"))
    return mean, half


# ---------------------------------------------------------------- the front


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="EV of one hand, played out by four PIMC sim players.")
    parser.add_argument("hand", help='your five cards, e.g. "JS AS 9H 9D TC"')
    parser.add_argument("--up", "--up-card", dest="up", default=None,
                        help="the up-card, e.g. 9S. Omit to deal one at random")
    parser.add_argument("--seat", type=int, default=0,
                        help="where you are sitting, 0-3 (default 0)")
    parser.add_argument("--dealer", type=int, default=3,
                        help="the dealing seat, 0-3 (default 3, so you are "
                             "eldest and speak first)")
    parser.add_argument("--deals", type=int, default=DEALS,
                        help="layouts of the other seats to play out (default "
                             "%d). This is the axis the error bar is on" % DEALS)
    parser.add_argument("--min-reached", type=int, default=None,
                        help="with --assume, keep dealing past --deals until "
                             "at least this many deals have reached your bid "
                             "(capped at %d deals). Most useful for round "
                             "two, which few deals get back to"
                             % MAX_DEALS)
    parser.add_argument("--player-eval-sims", type=int, default=None,
                        help="worlds each player imagines per card-play "
                             "decision (default %d, the measured mean settle "
                             "point). Setting it also carries the bid and "
                             "discard budgets with it unless those are given "
                             "explicitly" % PLAYER_EVAL_SIMS)
    parser.add_argument("--bid-eval-sims", type=int, default=None,
                        help="worlds per bidding decision (default %d). These "
                             "cost far more each under the 'god' pass model"
                             % BID_EVAL_SIMS)
    parser.add_argument("--discard-eval-sims", type=int, default=None,
                        help="worlds per discard decision (default %d -- the "
                             "hungriest of the three)" % DISCARD_EVAL_SIMS)
    parser.add_argument("--pass-model", default=players.PASS_ZERO,
                        choices=(players.PASS_ZERO, players.PASS_GOD_MODE),
                        help="how a player prices passing: 'zero' is worth "
                             "nothing and ~4x faster, 'god' runs the rest of "
                             "the auction in God Mode (default zero)")
    parser.add_argument("--assume", default=ASSUME_AUCTION,
                        choices=ASSUMPTIONS,
                        help="what your seat does with its first bid. "
                             "'auction' (default) lets it decide, so the mean "
                             "mixes the deals you called with the ones you "
                             "passed. The others pin the opening bid and "
                             "condition on it -- 'order' is the value of "
                             "ordering this hand up. 'name', 'name-alone' and "
                             "'pass-twice' are round two: you pass the "
                             "up-card, and on the deals where it comes back "
                             "round to you, name --suit or pass again")
    parser.add_argument("--suit", default=None,
                        help="the suit to name with --assume name or "
                             "name-alone: S, H, D or C. Not the up-card's")
    parser.add_argument("--let-auction-play",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="with --assume, run the auction and pin your "
                             "opening bid only if it reaches you, so an "
                             "earlier seat can take the option away. "
                             "--no-let-auction-play skips the auction and "
                             "prices the call from your seat every deal")
    parser.add_argument("--epsilon", type=_epsilon, default=EPSILON,
                        help="indifference band, in points. A decision stops "
                             "sampling once its remaining options are within "
                             "this of the leader, since picking either moves "
                             "the answer by less than it. 0 stops only on "
                             "proven gaps; 'none' disables early stopping and "
                             "restores exact averaging (default %g)" % EPSILON)
    parser.add_argument("--prune-top-trumps", action="store_true",
                        help="Axiom 1: never consider discarding the right "
                             "bower, left bower or ace of trump. A search-space "
                             "cut, not a model change -- but it rests on an "
                             "axiom, not a proof. See notes/discard_dominance.md")
    parser.add_argument("--both", action="store_true",
                        help="also solve each layout in God Mode and report "
                             "the paired difference")
    parser.add_argument("--workers", type=int, default=1,
                        help="how many deals to play at once. Threads under "
                             "--engine fast, processes under --engine python, "
                             "where each also pays the ~15s JIT warmup")
    parser.add_argument("--engine", default=FAST, choices=ENGINES,
                        help="which implementation plays the deals. 'fast' "
                             "(default) is the compiled one in fastsim.py; "
                             "'python' is table.py and players.py, which is "
                             "the readable one and about 200x slower")
    parser.add_argument("--tt-bits", type=int, default=None,
                        help="log2 of the compiled engine's transposition "
                             "table, which costs 8 bytes a slot and is shared "
                             "by every thread. Defaults to roughly one slot "
                             "per position the sweep will look at, capped at "
                             "%d (%d MB)" % (TT_MAX_BITS,
                                             (1 << TT_MAX_BITS) * 8 // 10 ** 6))
    parser.add_argument("--no-loners", action="store_true",
                        help="forbid going alone")
    parser.add_argument("--stick", action="store_true",
                        help="stick the dealer: the dealer may not pass in "
                             "round two")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="no progress line")
    return parser.parse_args(argv)


def _budget(explicit, carried, measured):
    """
    One decision kind's sample budget.

    Three levels, and the middle one is the reason this is a function rather
    than an `or`: an explicit `--bid-eval-sims` wins; failing that
    `--player-eval-sims` carries to every kind, which is what it has always
    done and what `--player-eval-sims 10000` has to keep meaning; failing that
    the kind's own measured mean settle point applies.
    """
    if explicit is not None:
        return explicit
    if carried is not None:
        return carried
    return measured


def setup_from(args) -> Setup:
    hand = tuple(r.parse_hand(args.hand))
    if len(hand) != game.HAND_SIZE:
        raise SystemExit("a hand is %d cards, got %d: %s"
                         % (game.HAND_SIZE, len(hand), r.hand_name(hand)))
    up_card = r.parse_card(args.up) if args.up else None
    if up_card is not None and up_card in hand:
        raise SystemExit("%s is both in your hand and the up-card"
                         % r.card_name(up_card))
    for name, value in (("seat", args.seat), ("dealer", args.dealer)):
        if not 0 <= value < game.PLAYERS:
            raise SystemExit("%s must be 0-%d, got %r"
                             % (name, game.PLAYERS - 1, value))

    if args.assume in ROUND_TWO_ASSUMPTIONS and up_card is None:
        raise SystemExit("--assume %s needs --up: round two is defined by "
                         "which suit was turned down" % args.assume)
    try:
        suit = check_round_two(args.assume, args.suit, up_card,
                               not args.no_loners)
    except ValueError as why:
        raise SystemExit(str(why))

    if not args.let_auction_play and args.assume not in (ASSUME_ORDER,
                                                         ASSUME_ALONE):
        raise SystemExit(
            "--no-let-auction-play only means something with --assume %s or "
            "%s: there is no pinned call to price without one, and skipping "
            "the auction is incoherent for --assume %s, where passing is the "
            "auction continuing."
            % (ASSUME_ORDER, ASSUME_ALONE, ASSUME_PASS))

    if args.engine == FAST and args.prune_top_trumps:
        raise SystemExit(
            "--prune-top-trumps is only implemented on --engine python. It is "
            "an unproven axiom worth 1.09x, and the compiled engine is already "
            "far past what that buys; see notes/discard_dominance.md.")

    return Setup(hand=hand, up_card=up_card, seat=args.seat,
                 dealer=args.dealer,
                 player_eval_sims=_budget(args.player_eval_sims, None,
                                          PLAYER_EVAL_SIMS),
                 bid_eval_sims=_budget(args.bid_eval_sims,
                                       args.player_eval_sims, BID_EVAL_SIMS),
                 discard_eval_sims=_budget(args.discard_eval_sims,
                                           args.player_eval_sims,
                                           DISCARD_EVAL_SIMS),
                 pass_model=args.pass_model, epsilon=args.epsilon,
                 prune_discards=args.prune_top_trumps,
                 assume=args.assume,
                 let_auction_play=args.let_auction_play,
                 allow_loners=not args.no_loners, stick=args.stick,
                 seed=args.seed, engine=args.engine, suit=suit)


def check_round_two(assume: str, suit, up_card,
                    allow_loners: bool) -> Optional[int]:
    """
    The suit a round-two assumption names, checked; None for the others.

    `suit` may be a letter ("H"), a name ("hearts") or a natural suit index.
    Raises ValueError for the turned-down suit, which round two cannot name,
    for a named call without a suit, and for a loner with loners off.
    """
    if assume not in (ASSUME_NAME, ASSUME_NAME_ALONE):
        if suit is not None:
            raise ValueError("a suit only means something with --assume %s "
                             "or %s" % (ASSUME_NAME, ASSUME_NAME_ALONE))
        return None
    if suit is None:
        raise ValueError("--assume %s needs a suit to name" % assume)
    if isinstance(suit, str):
        text = suit.strip().lower()
        found = [s for s in range(4)
                 if text in (r.suit_name(s), r.suit_name(s)[0])]
        if not found:
            raise ValueError("no such suit: %r" % (suit,))
        suit = found[0]
    if not 0 <= int(suit) < 4:
        raise ValueError("no such suit: %r" % (suit,))
    suit = int(suit)
    if up_card is not None and suit == up_card.suit:
        raise ValueError("%s was turned down, so it cannot be named in "
                         "round two" % r.suit_name(suit))
    if assume == ASSUME_NAME_ALONE and not allow_loners:
        raise ValueError("cannot name a suit alone with loners off")
    return suit


def describe(setup: Setup, args):
    order = [(setup.dealer + 1 + i) % game.PLAYERS for i in range(game.PLAYERS)]
    print("Euchre: what is this hand worth at a table that cannot see it?")
    print("  %-30s %s" % ("your hand", r.hand_name(setup.hand)))
    print("  %-30s %s" % ("up-card",
                          r.card_name(setup.up_card) if setup.up_card
                          else "(dealt at random)"))
    print("  %-30s seat %d, dealer %d, you speak %d of %d"
          % ("seating", setup.seat, setup.dealer,
             order.index(setup.seat) + 1, game.PLAYERS))
    print("  %-30s %d deals x %d play / %d bid / %d discard eval sims, "
          "pass model %r%s%s%s"
          % ("sweep", args.deals, setup.player_eval_sims, setup.bid_eval_sims,
             setup.discard_eval_sims, setup.pass_model,
             ", exact (no early stop)" if setup.epsilon is None
             else ", epsilon %g" % setup.epsilon,
             "" if setup.allow_loners else ", loners off",
             ", stick the dealer" if setup.stick else ""))
    if setup.round_two:
        then = ("pass again" if setup.assume == ASSUME_PASS_TWICE
                else "name %s%s" % (r.suit_name(setup.suit),
                                    " alone" if setup.assume
                                    == ASSUME_NAME_ALONE else ""))
        print("  %-30s you pass the up-card; if every seat passes it and the "
              "bidding comes back to you, you %s" % ("assumption", then))
    elif setup.assume != ASSUME_AUCTION:
        print("  %-30s you %s%s"
              % ("assumption",
                 {ASSUME_ORDER: "order it up",
                  ASSUME_ALONE: "order it up alone",
                  ASSUME_PASS: "pass on the up-card"}[setup.assume],
                 " whenever the auction reaches you"
                 if setup.let_auction_play
                 else " every deal (auction skipped)"))
        if setup.assume == ASSUME_PASS:
            print("  %-30s %s" % ("", "only that bid is pinned -- round two "
                                      "is yours to bid for yourself"))
    if setup.engine == FAST:
        bits = args.tt_bits if args.tt_bits is not None else default_tt_bits()
        print("  %-30s compiled, %d thread%s, %d MB transposition table"
              % ("engine", args.workers, "" if args.workers == 1 else "s",
                 (1 << bits) * 8 // 10 ** 6))
        if not compiled_already():
            print("  %-30s %s" % ("", "first run since an edit -- about 80s "
                                      "of compiling before the first deal"))
    else:
        print("  %-30s python, %d process%s"
              % ("engine", args.workers, "" if args.workers == 1 else "es"))
    print()


def main(argv=None):
    args = parse_args(argv)
    setup = setup_from(args)
    describe(setup, args)

    started = time.time()

    def progress(done, total):
        every = max(1, total // 20)
        if done % every and done != total:
            return
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        print("  ... %d/%d deals, %.0fs elapsed, ~%.0fs left"
              % (done, total, elapsed, (total - done) / rate if rate else 0),
              file=sys.stderr)

    spent = None
    if args.min_reached is not None:
        if setup.engine != FAST or args.both or setup.assume == ASSUME_AUCTION:
            raise SystemExit("--min-reached needs --assume, --engine fast and "
                             "no --both")
        pimc, spent = run_fast_until(
            setup, args.min_reached, args.deals, workers=args.workers,
            tt_bits=args.tt_bits,
            progress=None if args.quiet else
            (lambda f: print("  ... %3.0f%% of %d reached, %.0fs"
                             % (100 * f, args.min_reached,
                                time.time() - started), file=sys.stderr)))
        god = []
        args.deals = len(pimc)
    elif setup.engine == FAST:
        pimc, god, spent = run_fast(setup, args.deals, both=args.both,
                                    workers=args.workers,
                                    tt_bits=args.tt_bits,
                                    progress=None if args.quiet else progress)
    else:
        pimc, god = run(setup, args.deals, both=args.both,
                        workers=args.workers,
                        progress=None if args.quiet else progress)

    report("PIMC sim (nobody can see your hand)", pimc, setup)
    if spent is not None:
        report_spend(spent, args.deals)
    if args.both:
        print()
        report("God Mode (every seat sees everything)", god, setup)
        paired(pimc, god)

    elapsed = time.time() - started
    print("\n  %.1fs total, %.4fs per deal%s"
          % (elapsed, elapsed / max(1, args.deals),
             "" if setup.engine == FAST
             else " (the first solve in each process pays ~15s of "
                  "JIT warmup)"))
    return pimc, god


if __name__ == "__main__":
    main()
