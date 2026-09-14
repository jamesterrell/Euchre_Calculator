"""
One deal, played out loud by four Monte Carlo players.

`pimc_sweep.py` plays hundreds of deals and reports the totals. This plays
exactly one and narrates every decision in it -- what each seat can see, what
its options were worth, and which one it took. It is the "show your working"
version, and the thing to read first if you want to know what `PIMCPlayer`
is actually doing.

    python pimc_example.py                    # the pinned example deal
    python pimc_example.py --pass-model zero  # call only on a positive number
    python pimc_example.py --samples 40       # think harder (and slower)
    python pimc_example.py --perfect          # the same deal, double-dummy
    python pimc_example.py --quiet            # decisions only, no working

## How passing gets priced, and why a seat calls a hand it expects to lose

Every option is compared on one number, passing included, and the largest wins.
So what a pass is *worth* decides everything about how willing a seat is to bid:

    --pass-model dd     (default) a pass is worth whatever the rest of the
                        auction does. Each imagined deal is handed to the
                        perfect-knowledge auction and played out.
    --pass-model zero   a pass is worth nothing. Since ties keep the first
                        option and passing is listed first, this is exactly
                        the rule "order up only if ordering up is positive".

The two differ because **passing is not free**. Declining does not end the deal
-- it hands it to the next seat, and what they do with it can be worse for you
than the call you turned down. A seat holding a call worth -1 should still make
it if passing lets the opponents march for -2. That is defensive bidding, it is
real Euchre, and `--pass-model zero` cannot express it: it treats every pass as
though the hand were about to be thrown in.

Note that `dd` is not simply *more* willing to bid, or simply less. It prices
each seat's pass on that seat's own prospects, so the same deal can push in both
directions at once -- on `--seed 11` it values the eldest hand's pass at a
cheerful +0.94 (talking it out of a call worth +0.69) and the dealer's at -1.94
(talking it into a call worth -1.25). Switching to `zero` reverses both. What
the aggregate measurements say is only that `dd` calls more often overall and is
euchred about twice as much for it.

Where `zero` is exactly right rather than approximately: the last seat to speak
in round two. If it passes, the deal really is thrown in for nothing, so both
models price that pass at 0 and agree.

Measured, neither is clearly stronger -- head to head against perfect knowledge
they are indistinguishable (-1.26 vs -1.34 +/- 0.36 points a deal), though
`zero` is euchred about half as often and runs ~4x faster. See `players.py`.

Other deals worth looking at, if this one starts to feel like the only one:

    --seed 11              three seats pass and the dealer orders it up at an
                           expectation it knows is negative, because passing
                           scored worse still. It is euchred. Run it again with
                           --pass-model zero and the auction changes shape
                           entirely: seat 0, whose pass was being priced at a
                           cheerful +0.94, now compares its +0.69 call against
                           0.00 and takes it, so the bidding ends at the first
                           seat and never reaches the dealer.
    --seed 14              seat 2 calls and the *dealer is an opponent*, so the
                           discard is chosen to hurt the contract.
    --seed 24 --dealer 2   nobody orders, the up-card is turned down, and the
                           auction goes to a second round.
    --seed 8               a loner, made: four points.

Every number printed under a decision is an average over sampled layouts, on
that seat's own team's scale, so **bigger is always better and the largest one
is the move taken**. Ties keep the first option listed, which is why passing is
printed first.

The important thing to watch is the gap between what a seat knows and what is
actually true. The referee prints all four hands at the top because it has
them; no player ever sees that block. Each seat sees its own five cards, the
up-card, and whatever has been played -- and every number it computes comes
from guessing the rest, a few dozen times, and solving each guess exactly.

Nothing here is a test. `tests/test_table.py` is where the behaviour is
checked; this file exists to be read.
"""
import argparse
import random

import bidding as b
import observation as ob
import players
import rotation as r
import table as t
from game import PLAYERS, deal_random

# Pinned rather than picked at random, so the commentary above stays true of
# what actually prints. This deal was chosen because the left bower turns up on
# both sides of its rule inside three tricks: with diamonds trump the jack of
# hearts is trump and so cannot follow a heart lead, and the jack of diamonds is
# trump and so must follow one. A player that read the printed suit would play
# both of them wrongly, and the output would look perfectly reasonable.
SEED = 16
DEALER = 3

RULE = "=" * 70
THIN = "-" * 70


def card_list(cards):
    return " ".join(r.card_name(c) for c in cards)


class Log:
    """
    Shared scratchpad for the four narrators.

    Each seat only ever sees its own turn, so somebody has to hold the running
    state that makes the narration read like a game: which round we are in,
    which trick, and the cards on the table. The referee would be the natural
    home for it, but `table.py` deliberately has no opinions about output.
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

    It reads `turn.observation` -- the seat's own view -- for everything it
    prints, so the narration cannot show anything the player was not entitled
    to see. The one exception is the deal block at the very top, which is
    printed by the referee's own copy before anybody bids.
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
        seat is not optimistic about the contract, it is pessimistic about what
        happens if it declines. Whether that pessimism is earned is exactly
        what `--pass-model` changes.
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
        description="Play one deal with Monte Carlo players, out loud.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dealer", type=int, default=DEALER)
    parser.add_argument("--samples", type=int, default=24,
                        help="layouts sampled per card-play decision")
    parser.add_argument("--bid-samples", type=int, default=16,
                        help="layouts sampled per bidding decision")
    parser.add_argument("--pass-model", default=players.PASS_DD,
                        choices=(players.PASS_DD, players.PASS_ZERO),
                        help="how a seat prices passing. 'zero' makes the rule "
                             "exactly 'call only if calling is worth more than "
                             "nothing'")
    parser.add_argument("--perfect", action="store_true",
                        help="use perfect-knowledge players instead, for "
                             "comparison on the same deal")
    parser.add_argument("--no-loners", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="skip the per-option numbers")
    args = parser.parse_args(argv)

    deal = deal_random(rng=random.Random(args.seed), dealer=args.dealer)
    kind = ("perfect-knowledge" if args.perfect
            else "Monte Carlo (PIMC)")

    print(RULE)
    print(" One deal, played by four %s players" % kind)
    print(RULE)
    print("\n  What the referee knows. No player below ever sees this:\n")
    print(deal.describe())
    print("\n  Play begins to the dealer's left, so seat %d leads trick one."
          % deal.first_bidder)

    log = Log(show_scores=not args.quiet, pass_model=args.pass_model)
    if args.perfect:
        inner = [players.PerfectPlayer() for _ in range(PLAYERS)]
    else:
        inner = [players.PIMCPlayer(samples=args.samples,
                                    bid_samples=args.bid_samples,
                                    pass_model=args.pass_model,
                                    rng=random.Random(1000 + seat))
                 for seat in range(PLAYERS)]
    table = [Narrator(seat, inner[seat], log) for seat in range(PLAYERS)]

    print("\n" + THIN)
    print(" THE AUCTION")
    print(THIN)
    if not args.perfect:
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

    if not args.perfect:
        spent = sum(p.solves for p in inner)
        print("\n  Between them the four seats solved %d complete Euchre hands "
              "to play this one." % spent)
        truth = b.play_value(contract.deal, contract.trump, contract.caller,
                             contract.alone)
        if truth == caller_score:
            print("  Played double-dummy -- every hand face up -- the same "
                  "contract is also worth %+d," % truth)
            print("  so nothing was lost in the play. Whatever was decided "
                  "here, was decided in the auction.")
        else:
            print("  Played double-dummy -- every hand face up -- the same "
                  "contract is worth %+d." % truth)
            print("  The %d-point gap is the price of not being able to see: "
                  "somewhere above, a seat" % abs(truth - caller_score))
            print("  played a card that was right in most of the worlds it "
                  "imagined and wrong in this one.")


if __name__ == "__main__":
    main()
