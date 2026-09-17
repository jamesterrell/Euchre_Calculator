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
close to free above ~800:

    deals   eval sims   epsilon   serial        10 workers
   10,000     default      0.05   ~2.8 hours    ~34 minutes
    1,000      10,000      0.40   ~8 minutes    ~2 minutes
   10,000      10,000      0.40   ~76 minutes   ~20 minutes
   10,000      10,000      0.15   ~4 hours      ~52 minutes
   10,000      10,000      none   ~45 days      ~10 days

The first row is a bare run at today's defaults -- 132/231/266 eval sims at
epsilon 0.05, measured at 0.205 s/deal. The rest are the earlier runs at 10,000
eval sims, which is what the epsilon comparison below was measured on.

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

`--workers` spreads deals over processes; each pays the ~20 s JIT warmup once,
and the measured speedup is 4.6x on a 12-thread machine -- 0.95 s/deal against
4.35 s/deal serial over 900 deals -- rather than the 10x the core count
suggests. Results do not depend on the worker count -- every deal seeds itself
from its own index -- so a parallel run and a serial one give the same number.

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
ASSUMPTIONS = (ASSUME_AUCTION, ASSUME_ORDER, ASSUME_ALONE, ASSUME_PASS)

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
        if self.assume != ASSUME_AUCTION and self.let_auction_play:
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


# ------------------------------------------------------------- reading them


def interval(values):
    """Mean and a 95% half-width. Same convention as `pimc_sweep.interval`."""
    n = len(values)
    if n < 2:
        return (float(values[0]) if values else 0.0), 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * (var / n) ** 0.5


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
        verb = {ASSUME_ORDER: "ordered",
                ASSUME_ALONE: "ordered alone",
                ASSUME_PASS: "passed"}[setup.assume]
        got = [rec for rec in records if rec.forced]
        sub_mean, sub_half = interval([rec.value for rec in got])
        print("    %-30s %d of %d (%.1f%%)"
              % ("deals you " + verb, len(got), n, 100.0 * len(got) / n))
        if got:
            print("    %-30s %+.3f +/- %.3f points per deal"
                  % ("EV given you " + verb, sub_mean, sub_half))
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
              % (role, roles[role], 100.0 * roles[role] / n,
                 sub_mean, sub_half))

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
                             "ordering this hand up")
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
                        help="processes to spread deals over; each pays the "
                             "~15s JIT warmup once")
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

    if not args.let_auction_play and args.assume not in (ASSUME_ORDER,
                                                         ASSUME_ALONE):
        raise SystemExit(
            "--no-let-auction-play only means something with --assume %s or "
            "%s: there is no pinned call to price without one, and skipping "
            "the auction is incoherent for --assume %s, where passing is the "
            "auction continuing."
            % (ASSUME_ORDER, ASSUME_ALONE, ASSUME_PASS))

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
                 seed=args.seed)


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
    if setup.assume != ASSUME_AUCTION:
        print("  %-30s you %s%s"
              % ("assumption",
                 {ASSUME_ORDER: "order it up",
                  ASSUME_ALONE: "order it up alone",
                  ASSUME_PASS: "pass"}[setup.assume],
                 " whenever the auction reaches you"
                 if setup.let_auction_play
                 else " every deal (auction skipped)"))
    if args.workers > 1:
        print("  %-30s %d processes" % ("workers", args.workers))
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

    pimc, god = run(setup, args.deals, both=args.both, workers=args.workers,
                    progress=None if args.quiet else progress)

    report("PIMC sim (nobody can see your hand)", pimc, setup)
    if args.both:
        print()
        report("God Mode (every seat sees everything)", god, setup)
        paired(pimc, god)

    elapsed = time.time() - started
    print("\n  %.1fs total, %.3fs per deal (the first solve in each process "
          "pays ~15s of JIT warmup)" % (elapsed, elapsed / max(1, args.deals)))
    return pimc, god


if __name__ == "__main__":
    main()
