"""
God Mode against the PIMC sim: what does seeing the other hands buy?

Every deal is played twice -- once by a God Mode table that sees everything,
once by a PIMC sim table that sees only its own cards and the play so far --
and this reports where the two part company.

    python pimc_sweep.py                    # 40 deals, both tables
    python pimc_sweep.py 200 --samples 30   # more deals, more search
    python pimc_sweep.py 60 --head-to-head  # what does seeing actually buy?
    python pimc_sweep.py 100 --pass-model zero --bid-samples 16

Three things worth watching, all places God Mode was expected to mislead:

  * **Passing out.** God Mode essentially never does it -- 0 of 1600 measured
    -- because every seat knows how the auction will go and can always find a
    call that is at worst harmless. Real tables throw hands in constantly.
  * **Loners.** God Mode almost never calls one: it only pays if you take all
    five unaided, and God Mode can see that you cannot. Real players call them
    far more often, and lose them.
  * **Euchres.** A contract chosen without seeing the defence should fail more
    often than one chosen while seeing it.

`--head-to-head` puts the PIMC sim on one team and God Mode on the other, each
deal played twice with the teams swapped so seat and dealer advantages cancel
exactly rather than statistically. What is left is the price of not knowing.

Sizing: a PIMC decision costs (options x samples) solves, and bidding costs far
more per sample than card play because pricing a pass means running the rest of
the auction. A few seconds per deal at the defaults, plus ~15 s JIT warmup.
"""
import argparse
import random
import sys
import time
from collections import Counter

import bidding as b
import game
import players
import rotation as r
import table as t

DEALS = 40


class Profile:
    """Running tally of what one table did over a sweep."""

    def __init__(self, name):
        self.name = name
        self.deals = 0
        self.passed_out = 0
        self.loners = 0
        self.euchred = 0
        self.marches = 0
        self.caller_points = 0
        self.caller_tricks = 0
        self.contracts = 0
        self.by_round = Counter()
        self.scores = []

    def add(self, result):
        self.deals += 1
        self.scores.append(result.value)
        if result.passed_out:
            self.passed_out += 1
            return

        self.contracts += 1
        self.by_round[result.contract.bidding_round] += 1
        self.caller_points += result.caller_score
        self.caller_tricks += result.caller_tricks
        if result.contract.alone:
            self.loners += 1
        if result.euchred:
            self.euchred += 1
        if result.caller_tricks == 5:
            self.marches += 1

    def _pct(self, n):
        return 100.0 * n / self.deals if self.deals else 0.0

    def rows(self):
        made = self.contracts - self.euchred
        return [
            ("deals", "%d" % self.deals),
            ("passed out", "%d (%.1f%%)" % (self.passed_out,
                                            self._pct(self.passed_out))),
            ("called alone", "%d (%.1f%%)" % (self.loners,
                                              self._pct(self.loners))),
            ("ordered up (round 1)", "%d" % self.by_round[b.ROUND_ONE]),
            ("named suit (round 2)", "%d" % self.by_round[b.ROUND_TWO]),
            ("contracts made", "%d of %d" % (made, self.contracts)),
            ("euchred", "%d (%.1f%% of calls)"
             % (self.euchred,
                100.0 * self.euchred / self.contracts if self.contracts else 0)),
            ("marches", "%d (%.1f%% of calls)"
             % (self.marches,
                100.0 * self.marches / self.contracts if self.contracts else 0)),
            ("mean tricks to the caller", "%.2f"
             % (self.caller_tricks / self.contracts if self.contracts else 0)),
            ("mean points to the caller", "%+.3f"
             % (self.caller_points / self.contracts if self.contracts else 0)),
        ]


def interval(values):
    """Mean and a 95% interval. Sampling error dominates everything here."""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0), 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * (var / n) ** 0.5


def pimc_table(args, seed):
    return [players.PIMCPlayer(samples=args.samples,
                               bid_samples=args.bid_samples,
                               discard_samples=args.discard_samples,
                               pass_model=args.pass_model,
                               rng=random.Random(seed * 100 + s))
            for s in range(4)]


def god_mode_table():
    return [players.GodModePlayer() for _ in range(4)]


def mixed_table(args, seed, pimc_team):
    """The PIMC sim in one team's seats, God Mode in the other's."""
    return [players.PIMCPlayer(samples=args.samples,
                               bid_samples=args.bid_samples,
                               discard_samples=args.discard_samples,
                               pass_model=args.pass_model,
                               rng=random.Random(seed * 100 + s))
            if s % 2 == pimc_team else players.GodModePlayer()
            for s in range(4)]


def show(profile):
    print("  %s" % profile.name)
    for label, value in profile.rows():
        print("    %-28s %s" % (label, value))


def compare(args):
    """Play every deal with both tables and report the two profiles."""
    honest = Profile("PIMC sim (each seat sees only its own hand)")
    god_mode = Profile("God Mode (every seat sees everything)")

    same_trump = same_caller = comparable = 0
    started = time.time()

    for i in range(args.deals):
        seed = args.seed + i
        deal = game.deal_random(rng=random.Random(seed), dealer=i % 4)

        a = t.play_deal(deal, pimc_table(args, seed),
                        stick_the_dealer=args.stick,
                        allow_loners=not args.no_loners)
        c = t.play_deal(deal, god_mode_table(),
                        stick_the_dealer=args.stick,
                        allow_loners=not args.no_loners)
        honest.add(a)
        god_mode.add(c)

        if not a.passed_out and not c.passed_out:
            comparable += 1
            same_trump += a.contract.trump == c.contract.trump
            same_caller += a.contract.caller == c.contract.caller

        if args.verbose:
            print("  deal %3d  pimc: %-52s  god: %s"
                  % (i + 1, a, c))
        elif (i + 1) % 10 == 0:
            print("  ... %d/%d deals, %.0fs"
                  % (i + 1, args.deals, time.time() - started),
                  file=sys.stderr)

    print()
    show(honest)
    print()
    show(god_mode)

    print("\n  auctions that agreed")
    if comparable:
        print("    %-28s %d of %d (%.0f%%)"
              % ("same trump suit", same_trump, comparable,
                 100.0 * same_trump / comparable))
        print("    %-28s %d of %d (%.0f%%)"
              % ("same caller", same_caller, comparable,
                 100.0 * same_caller / comparable))
    else:
        print("    (no deal produced a contract at both tables)")
    return honest, god_mode


def head_to_head(args):
    """
    The PIMC sim against God Mode, teams swapped on every deal.

    Playing each deal both ways cancels seat and dealer advantages exactly, so
    what is left is down to what the players know, not where they sat.
    """
    margins = []
    started = time.time()

    for i in range(args.deals):
        seed = args.seed + i
        deal = game.deal_random(rng=random.Random(seed), dealer=i % 4)

        # PIMC as team 0, then PIMC as team 1 on the same deal.
        first = t.play_deal(deal, mixed_table(args, seed, 0),
                            stick_the_dealer=args.stick,
                            allow_loners=not args.no_loners)
        second = t.play_deal(deal, mixed_table(args, seed, 1),
                             stick_the_dealer=args.stick,
                             allow_loners=not args.no_loners)

        # `value` is net to team 0 both times, so PIMC's margin is the first
        # reading minus the second.
        margins.append(first.value - second.value)

        if args.verbose:
            print("  deal %3d  pimc as team 0: %+d   pimc as team 1: %+d"
                  % (i + 1, first.value, -second.value))
        elif (i + 1) % 10 == 0:
            print("  ... %d/%d deals, %.0fs"
                  % (i + 1, args.deals, time.time() - started),
                  file=sys.stderr)

    mean, half = interval(margins)
    print("\n  PIMC minus God Mode, over %d deals played both ways"
          % args.deals)
    print("    %-28s %+.3f +/- %.3f points per deal" % ("margin", mean, half))
    print("    %-28s %s" % ("reading",
                            "PIMC is behind" if mean < -half else
                            "PIMC is ahead" if mean > half else
                            "no separation at this sample size"))
    return margins


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play deals with players who cannot see each other's hands.")
    parser.add_argument("deals", nargs="?", type=int, default=DEALS,
                        help="how many deals to play (default %d)" % DEALS)
    parser.add_argument("--samples", type=int,
                        default=players.RESEARCHED_PLAY,
                        help="layouts sampled per card-play decision "
                             "(default %d, the measured mean settle point)"
                             % players.RESEARCHED_PLAY)
    parser.add_argument("--bid-samples", type=int,
                        default=players.RESEARCHED_BID,
                        help="layouts sampled per bidding decision (default "
                             "%d); these cost far more each than card-play "
                             "samples" % players.RESEARCHED_BID)
    parser.add_argument("--discard-samples", type=int,
                        default=players.RESEARCHED_DISCARD,
                        help="layouts sampled per discard decision (default "
                             "%d, the hungriest of the three)"
                             % players.RESEARCHED_DISCARD)
    parser.add_argument("--pass-model", default=players.PASS_GOD_MODE,
                        choices=players.PASS_MODELS,
                        help="how a PIMC sim player prices passing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-loners", action="store_true",
                        help="forbid going alone, so the numbers stay "
                             "comparable with older four-handed measurements")
    parser.add_argument("--stick", action="store_true",
                        help="stick the dealer: the dealer may not pass in "
                             "round two")
    parser.add_argument("--head-to-head", action="store_true",
                        help="the PIMC sim against God Mode instead of "
                             "profiling each table separately")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every deal")
    args = parser.parse_args(argv)

    print("Euchre: God Mode vs the PIMC sim")
    print("  %d deals, %d play / %d bid / %d discard samples, pass model %r%s"
          % (args.deals, args.samples, args.bid_samples, args.discard_samples,
             args.pass_model,
             "" if not args.no_loners else ", loners off"))
    started = time.time()

    if args.head_to_head:
        head_to_head(args)
    else:
        compare(args)

    print("\n  %.1fs total (the first solve pays ~15s of JIT warmup)"
          % (time.time() - started))


if __name__ == "__main__":
    main()
