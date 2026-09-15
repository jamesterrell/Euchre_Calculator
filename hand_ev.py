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

**Cost is `deals x player_eval_sims`, and that is the whole story.** Measured on
`JS AS 9H 9D TC` with `9S` up -- `pass_model="zero"`, loners on -- one deal cost
0.39 s at 10 sims per eval, so about **39 ms per sim**, scaling linearly in both
axes. A hand that gets ordered up nearly every deal, like that one, is the
expensive case: five tricks always get played.

    deals   eval sims   serial       10 workers (measured ~3x, not 10x)
    1,000          10   ~7 minutes   ~2 minutes
    1,000         100   ~1 hour      ~20 minutes
   10,000          10   ~1 hour      ~20 minutes
   10,000         100   ~11 hours    ~4 hours
   10,000      10,000   ~45 days     ~15 days

`--workers` spreads deals over processes; each pays the ~20 s JIT warmup once,
and the observed speedup is around 3x on a 12-thread machine rather than the
10x the core count suggests. Results do not depend on the worker count -- every
deal seeds itself from its own index -- so a parallel run and a serial one give
the same number.

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
from dataclasses import dataclass
from typing import Optional, Tuple

import bidding as b
import game
import players
import rotation as r
import table as t

DEALS = 1000
PLAYER_EVAL_SIMS = 10


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
    pass_model: str
    allow_loners: bool
    stick: bool
    seed: int

    def deal(self, i: int) -> game.Deal:
        """Deal `i` of the sweep. Derived from `i` alone, so workers agree."""
        return game.deal_around(known_hand=list(self.hand), seat=self.seat,
                                up_card=self.up_card,
                                rng=random.Random(self.seed + i),
                                dealer=self.dealer)

    def table(self, i: int):
        """Four PIMC sim players, seeded per deal and per seat."""
        return [players.PIMCPlayer(samples=self.player_eval_sims,
                                   bid_samples=self.bid_eval_sims,
                                   pass_model=self.pass_model,
                                   rng=random.Random(self.seed * 7919 + i * 4 + s))
                for s in range(game.PLAYERS)]


@dataclass(frozen=True)
class Record:
    """One deal's outcome, on the asking seat's own team's scale. Picklable."""

    value: int                       # points to the asker's team
    caller: int                      # -1 if the deal was passed out
    trump: int
    alone: bool
    caller_tricks: int
    caller_score: int

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
    result = t.play_deal(setup.deal(i), setup.table(i),
                         stick_the_dealer=setup.stick,
                         allow_loners=setup.allow_loners)
    if result.passed_out:
        return PASSED_OUT
    return Record(value=b.value_to(setup.seat, result.value),
                  caller=result.contract.caller,
                  trump=result.contract.trump,
                  alone=result.contract.alone,
                  caller_tricks=result.caller_tricks,
                  caller_score=result.caller_score)


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
    Play `deals` sampled layouts. Returns (pimc records, god mode records).

    Serial below two workers, because a worker pays the JIT warmup on the way
    in and a short sweep never earns that back.
    """
    jobs = [(i, both) for i in range(deals)]
    pimc, god = [], []
    done = 0

    if workers > 1:
        with mp.Pool(workers, initializer=_init, initargs=(setup,)) as pool:
            results = pool.imap_unordered(_work, jobs, chunksize=4)
            for _, a, c in results:
                pimc.append(a)
                if c is not None:
                    god.append(c)
                done += 1
                if progress:
                    progress(done, deals)
    else:
        _init(setup)
        for job in jobs:
            _, a, c = _work(job)
            pimc.append(a)
            if c is not None:
                god.append(c)
            done += 1
            if progress:
                progress(done, deals)

    return pimc, god


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
    print("    %-30s %+.3f +/- %.3f points per deal"
          % ("EV to your team", mean, half))

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
    parser.add_argument("--player-eval-sims", type=int,
                        default=PLAYER_EVAL_SIMS,
                        help="worlds each player imagines per card-play "
                             "decision (default %d)" % PLAYER_EVAL_SIMS)
    parser.add_argument("--bid-eval-sims", type=int, default=None,
                        help="worlds per bidding decision; defaults to "
                             "--player-eval-sims. These cost far more each "
                             "under the 'god' pass model")
    parser.add_argument("--pass-model", default=players.PASS_ZERO,
                        choices=(players.PASS_ZERO, players.PASS_GOD_MODE),
                        help="how a player prices passing: 'zero' is worth "
                             "nothing and ~4x faster, 'god' runs the rest of "
                             "the auction in God Mode (default zero)")
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

    return Setup(hand=hand, up_card=up_card, seat=args.seat,
                 dealer=args.dealer, player_eval_sims=args.player_eval_sims,
                 bid_eval_sims=(args.player_eval_sims
                                if args.bid_eval_sims is None
                                else args.bid_eval_sims),
                 pass_model=args.pass_model,
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
    print("  %-30s %d deals x %d play / %d bid eval sims, pass model %r%s%s"
          % ("sweep", args.deals, setup.player_eval_sims, setup.bid_eval_sims,
             setup.pass_model,
             "" if setup.allow_loners else ", loners off",
             ", stick the dealer" if setup.stick else ""))
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
