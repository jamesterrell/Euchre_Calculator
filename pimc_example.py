"""
One deal, played out loud by four PIMC sim players.

`pimc_sweep.py` plays hundreds of deals and reports totals. This plays exactly
one and narrates every decision -- what each seat can see, what its options
were worth, and which it took. Read this first to see what `PIMCPlayer` does.

    python pimc_example.py                    # the pinned example deal
    python pimc_example.py --pass-model zero  # call only on a positive number
    python pimc_example.py --samples 40       # think harder (and slower)
    python pimc_example.py --researched-defaults  # the measured settle points
    python pimc_example.py --god-mode         # the same deal, in God Mode
    python pimc_example.py --quiet            # decisions only, no working

## How passing gets priced, and why a seat calls a hand it expects to lose

Every option is compared on one number, passing included, and the largest wins.
So what a pass is *worth* decides how willing a seat is to bid:

    --pass-model god    (default) a pass is worth whatever the rest of the
                        auction does -- each imagined deal is handed to the
                        God Mode auction and played out.
    --pass-model zero   a pass is worth nothing. Ties keep the first option and
                        passing is listed first, so this is exactly the rule
                        "order up only if ordering up is positive".

They differ because **passing is not free**. Declining hands the deal to the
next seat, and what they do with it can be worse for you than the call you
turned down: a call worth -1 is still right if passing lets the opponents march
for -2. That is defensive bidding, and `zero` cannot express it -- it treats
every pass as though the hand were about to be thrown in.

`god` is not simply more willing to bid, or less. It prices each seat's pass on
that seat's own prospects, so one deal can push both ways: on `--seed 11` it
values the eldest hand's pass at +0.94 (talking it out of a +0.69 call) and the
dealer's at -1.94 (talking it into a -1.25 call). `zero` reverses both. In
aggregate `god` calls more often and is euchred about twice as much for it, but
head to head against God Mode the two are indistinguishable (-1.26 vs -1.34
+/- 0.36 points a deal). `zero` runs ~4x faster. See `players.py`.

Where `zero` is exactly right: the last seat to speak in round two. If it
passes the deal really is thrown in for nothing, so both models agree.

Other deals worth looking at:

    --seed 11              three seats pass and the dealer orders up at an
                           expectation it knows is negative, because passing
                           scored worse. It is euchred. With --pass-model zero
                           the auction changes shape entirely: seat 0 compares
                           its +0.69 call against 0.00, takes it, and the
                           bidding never reaches the dealer.
    --seed 14              seat 2 calls and the *dealer is an opponent*, so the
                           discard is chosen to hurt the contract.
    --seed 24 --dealer 2   the up-card is turned down and round two begins.
    --seed 8               a loner, made: four points.

Every number under a decision is an average over sampled layouts, on that
seat's own team's scale, so **bigger is better and the largest is the move
taken**. Ties keep the first option, which is why passing prints first.

What to watch is the gap between what a seat knows and what is true. The
referee prints all four hands at the top; no player ever sees that block. Each
seat sees its own five cards, the up-card and what has been played, and every
number it computes comes from guessing the rest a few dozen times and solving
each guess exactly.

Nothing here is a test -- `tests/test_table.py` checks the behaviour. This file
exists to be read.
"""
import argparse
import random

import bidding as b
import observation as ob
import players
import rotation as r
import table as t
from game import PLAYERS, deal_random

# Pinned rather than random, so the commentary above stays true of what prints.
# Chosen because the left bower turns up on both sides of its rule inside three
# tricks: with diamonds trump the jack of hearts is trump and cannot follow a
# heart lead, and the jack of diamonds is trump and must follow one. A player
# reading the printed suit would get both wrong and still look reasonable.
SEED = 16
DEALER = 3

# What this example has always narrated at. Kept as the default so the deal
# reads the same as the docstring describes; --researched-defaults opts in to
# players.RESEARCHED_*, the measured mean settle points.
SAMPLES = 24
BID_SAMPLES = 16

RULE = "=" * 70
THIN = "-" * 70


def card_list(cards):
    return " ".join(r.card_name(c) for c in cards)


class Log:
    """
    Shared scratchpad for the four narrators.

    Each seat only sees its own turn, so somebody has to hold the state that
    makes the narration read like a game: the round, the trick, the cards on
    the table. `table.py` would be the natural home, but it has no opinions
    about output.
    """

    def __init__(self, show_scores=True, pass_model=None):
        self.show_scores = show_scores
        self.pass_model = pass_model
        self.round_shown = None
        self.trick = []
        self.trick_no = 0
        self.contract = None
        self.tally = [0, 0]

    def scores(self, scored, chosen, fmt=str, indent=8):
        """Print every option and its averaged value, best last."""
        if not (self.show_scores and scored):
            return
        pad = " " * indent
        for option, value in sorted(scored.items(), key=lambda kv: kv[1]):
            mark = "  <-- taken" if option == chosen else ""
            print("%s%-26s %+6.2f%s" % (pad, fmt(option), value, mark))

    def note_play(self, seat, card):
        self.trick.append((seat, card))
        width = 3 if self.contract.alone else 4
        if len(self.trick) < width:
            return

        trump = self.contract.trump
        winner = t.trick_winner(self.trick, trump)
        won_by = next(c for s, c in self.trick if s == winner)
        self.tally[winner % 2] += 1
        print("      %s trick %d to seat %d with %s     (team 0: %d, team 1: %d)"
              % ("-" * 4, self.trick_no + 1, winner, r.card_name(won_by),
                 self.tally[0], self.tally[1]))
        self.trick = []
        self.trick_no += 1


class Narrator:
    """
    Wraps a player and says what it is doing. Decides nothing itself.

    Everything it prints comes from `turn.observation`, the seat's own view, so
    the narration cannot show what the player was not entitled to see. The one
    exception is the deal block at the top, printed before anybody bids.
    """

    def __init__(self, seat, inner, log):
        self.seat = seat
        self.inner = inner
        self.log = log

    def _scored(self):
        return getattr(self.inner, "last_scores", {}) or {}

    # ------------------------------------------------------------- bidding

    def bid(self, turn):
        if self.log.round_shown != turn.bidding_round:
            self.log.round_shown = turn.bidding_round
            print()
            if turn.bidding_round == b.ROUND_ONE:
                print("  Round one -- order up %s, or pass."
                      % r.suit_name(turn.deal.up_card.suit))
            else:
                print("  Round two -- %s is turned down. Name another suit, "
                      "or pass." % r.suit_name(turn.deal.up_card.suit))

        seat = turn.seat
        who = " (dealer)" if seat == turn.deal.dealer else ""
        if turn.index == 0 and turn.bidding_round == b.ROUND_ONE:
            who = " (eldest)"
        print("\n    seat %d%s holds %s"
              % (seat, who, card_list(turn.observation.hand)))

        bid = self.inner.bid(turn)
        scored = self._scored()
        self.log.scores(scored, bid, fmt=str)
        print("      seat %d %s" % (seat, bid))
        self._explain(bid, scored)
        return bid

    def _explain(self, bid, scored):
        """
        Say something when a seat calls a hand it expects to lose.

        It reads like a bug and is not one. Every option sits on one scale and
        the largest wins, so a call worth -1.25 beats a pass worth -1.94: the
        seat is pessimistic about declining, not optimistic about the contract.
        Whether that pessimism is earned is what `--pass-model` changes.
        """
        if bid.action == t.PASS or not scored:
            return
        value = scored.get(bid)
        passing = next((v for o, v in scored.items() if o.action == t.PASS),
                       None)
        if value is None or passing is None or value >= 0:
            return
        print("        ^ it expects to lose this contract, and took it anyway "
              "because passing")
        print("          scored %+.2f -- worse. Declining does not end the "
              "deal, it hands it on," % passing)
        print("          and here the seat thinks it comes back hurting.")
        if self.log.pass_model != players.PASS_ZERO:
            print("          Under --pass-model zero a pass is worth 0.00 and "
                  "this call loses to it.")
            print("          Earlier seats are repriced too, though, so the "
                  "auction may never get here.")

    def discard(self, turn):
        print("\n    seat %d (dealer) picks up %s and holds six: %s"
              % (turn.seat, r.card_name(turn.observation.up_card),
                 card_list(turn.options)))
        if turn.seat % 2 == turn.caller % 2:
            print("      One has to go. It is picking up for a contract its "
                  "own side wants to make,")
            print("      so it keeps whatever brings the contract home.")
        else:
            print("      One has to go, and the *dealer* chooses it, not the "
                  "caller. Ordered up by")
            print("      the opposition, it keeps whatever hurts the contract "
                  "most. That is real")
            print("      Euchre, and it is minimaxed here rather than assumed "
                  "away.")

        card = self.inner.discard(turn)
        self.log.scores(self._scored(), card, fmt=r.card_name)
        print("      seat %d buries %s" % (turn.seat, r.card_name(card)))
        return card

    # ---------------------------------------------------------------- play

    def play(self, turn):
        observation = turn.observation
        if not observation.current_trick:
            print("\n    Trick %d -- seat %d leads."
                  % (turn.trick_no + 1, turn.seat))

        hand = card_list(observation.hand)
        if len(turn.legal) == 1:
            print("      seat %d holds %s -- only %s is legal"
                  % (turn.seat, hand, r.card_name(turn.legal[0])))
        elif len(turn.legal) < len(observation.hand):
            led = observation.current_trick[0][1]
            print("      seat %d holds %s -- must follow %s, so: %s"
                  % (turn.seat, hand, r.suit_name(
                      ob.effective_suit(led, turn.trump)),
                     card_list(turn.legal)))
        else:
            print("      seat %d holds %s -- anything is legal"
                  % (turn.seat, hand))

        known = self._voids(observation)
        if known:
            print("        (it has seen: %s)" % known)

        card = self.inner.play(turn)
        self.log.scores(self._scored(), card, fmt=r.card_name, indent=10)
        print("      seat %d plays %s" % (turn.seat, r.card_name(card)))
        self.log.note_play(turn.seat, card)
        return card

    def _voids(self, observation):
        """The inferences this seat is carrying into the decision."""
        voids = observation.voids()
        parts = []
        for seat in range(PLAYERS):
            if seat == observation.seat:
                continue
            shown = voids.get(seat, frozenset())
            if shown:
                parts.append("seat %d has no %s"
                             % (seat, "/".join(sorted(r.suit_name(s)
                                                      for s in shown))))
        return ", ".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Play one deal with PIMC sim players, out loud.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dealer", type=int, default=DEALER)
    parser.add_argument("--samples", type=int, default=None,
                        help="layouts sampled per card-play decision "
                             "(default %d)" % SAMPLES)
    parser.add_argument("--bid-samples", type=int, default=None,
                        help="layouts sampled per bidding decision (default "
                             "%d)" % BID_SAMPLES)
    parser.add_argument("--discard-samples", type=int, default=None,
                        help="layouts sampled per discard decision; defaults "
                             "to --bid-samples")
    parser.add_argument("--researched-defaults", action="store_true",
                        help="use the measured mean settle point for each kind "
                             "of decision -- %d play / %d bid / %d discard. "
                             "See notes/settle_counts.md"
                             % (players.RESEARCHED_PLAY, players.RESEARCHED_BID,
                                players.RESEARCHED_DISCARD))
    parser.add_argument("--pass-model", default=players.PASS_GOD_MODE,
                        choices=players.PASS_MODELS,
                        help="how a seat prices passing. 'zero' makes the rule "
                             "exactly 'call only if calling is worth more than "
                             "nothing'")
    parser.add_argument("--god-mode", action="store_true",
                        help="use God Mode players instead, for "
                             "comparison on the same deal")
    parser.add_argument("--no-loners", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="skip the per-option numbers")
    args = parser.parse_args(argv)
    args.samples = players.sample_budget(
        args.samples, args.researched_defaults, players.RESEARCHED_PLAY,
        SAMPLES)
    args.bid_samples = players.sample_budget(
        args.bid_samples, args.researched_defaults, players.RESEARCHED_BID,
        BID_SAMPLES)
    args.discard_samples = players.sample_budget(
        args.discard_samples, args.researched_defaults,
        players.RESEARCHED_DISCARD, args.bid_samples)

    deal = deal_random(rng=random.Random(args.seed), dealer=args.dealer)
    kind = "God Mode" if args.god_mode else "PIMC sim"

    print(RULE)
    print(" One deal, played by four %s players" % kind)
    print(RULE)
    print("\n  What the referee knows. No player below ever sees this:\n")
    print(deal.describe())
    print("\n  Play begins to the dealer's left, so seat %d leads trick one."
          % deal.first_bidder)
    if not args.god_mode:
        print("  %d play / %d bid / %d discard samples a decision%s"
              % (args.samples, args.bid_samples, args.discard_samples,
                 " (researched defaults)" if args.researched_defaults
                 else ""))

    log = Log(show_scores=not args.quiet, pass_model=args.pass_model)
    if args.god_mode:
        inner = [players.GodModePlayer() for _ in range(PLAYERS)]
    else:
        inner = [players.PIMCPlayer(samples=args.samples,
                                    bid_samples=args.bid_samples,
                                    discard_samples=args.discard_samples,
                                    pass_model=args.pass_model,
                                    rng=random.Random(1000 + seat))
                 for seat in range(PLAYERS)]
    table = [Narrator(seat, inner[seat], log) for seat in range(PLAYERS)]

    print("\n" + THIN)
    print(" THE AUCTION")
    print(THIN)
    if not args.god_mode:
        if args.pass_model == players.PASS_ZERO:
            print("\n  A pass is priced at nothing, so a seat calls only if "
                  "calling beats 0.00.")
            print("  That cannot express a defensive call -- see the module "
                  "docstring.")
        else:
            print("\n  A pass is priced by playing the rest of the auction out "
                  "in each imagined deal,")
            print("  so it is rarely worth exactly nothing, and a seat may "
                  "call a hand it expects")
            print("  to lose if declining looks worse still.")

    # The narrators need the contract to score tricks; the referee settles it
    # part-way through, so it is picked up from the result afterwards. Play is
    # narrated as it happens, so log.contract is set by run_auction's return.
    contract, _ = t.run_auction(deal, table, allow_loners=not args.no_loners)
    if contract is None:
        print("\n  Nobody called. The deal is thrown in, 0 points.")
        return

    log.contract = contract
    print("\n  Contract: %s" % contract)
    if contract.alone:
        print("  Seat %d sits the hand out; tricks are three cards wide."
              % contract.sitting)

    print("\n" + THIN)
    print(" THE PLAY")
    print(THIN)

    caller_tricks, plays, winners = t.play_contract(contract, table)
    from fast_search import _final
    caller_score = int(_final(caller_tricks, contract.alone))

    print("\n" + THIN)
    print(" RESULT")
    print(THIN)
    print("\n  %s" % contract)
    print("  The calling team took %d of 5 tricks -> %+d to them, %+d to team 0."
          % (caller_tricks, caller_score,
             b.net_to_team0(caller_score, contract.caller)))
    if caller_score < 0:
        print("  Euchred: they called it and could not make three.")
    elif caller_tricks == 5:
        print("  A march -- all five tricks.")

    if not args.god_mode:
        spent = sum(p.solves for p in inner)
        print("\n  Between them the four seats solved %d complete Euchre hands "
              "to play this one." % spent)
        truth = b.play_value(contract.deal, contract.trump, contract.caller,
                             contract.alone)
        if truth == caller_score:
            print("  Played in God Mode -- every hand face up -- the same "
                  "contract is also worth %+d," % truth)
            print("  so nothing was lost in the play. Whatever was decided "
                  "here, was decided in the auction.")
        else:
            print("  Played in God Mode -- every hand face up -- the same "
                  "contract is worth %+d." % truth)
            print("  The %d-point gap is the price of not being able to see: "
                  "somewhere above, a seat" % abs(truth - caller_score))
            print("  played a card that was right in most of the worlds it "
                  "imagined and wrong in this one.")


if __name__ == "__main__":
    main()
