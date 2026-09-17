"""
What does a hand need before ordering up is not a losing bid?

Two questions, and they are not the same one:

  1. Is ordering with **no trump** ever right? Everybody knows it is not. The
     point of measuring it is to find out whether "never" is literally true or
     merely nearly always, which is the difference between an axiom you can
     prune on and a heuristic you cannot.
  2. What is the **minimum hand strength** for a call that is not negative EV?

"Hand strength" has no canonical definition in Euchre, and that is the whole
difficulty with question 2. Pick a scoring metric and you get a threshold that
is partly an artifact of the metric. So this script does not invent one. It
reports the EV against several plain, interpretable descriptions of a hand --
trump count, highest trump, off-suit aces -- and lets the boundary be read off
whichever the reader trusts.

Two values are reported for every bucket, because they answer different
questions:

  * **order** -- what the seat's team scores if it orders up. "Not negative EV"
    in the literal sense.
  * **gain** -- ordering minus passing, where passing is worth whatever the
    rest of the auction does (`bidding.rest_of_auction`). This is the actual
    decision criterion, and it is the stricter one: a call can score above zero
    and still be worse than declining. CLAUDE.md's "passing is not free".

**God Mode, and that cuts a specific way.** Every value here is exact play with
all four hands visible, so the defenders never misdefend. That makes these
numbers *pessimistic* about calling relative to a real table. As a threshold
for pruning a PIMC bidder it is therefore the wrong direction for safety -- it
would prune calls a PIMC player would rightly make. Read it as "what is needed
against perfect defence", and re-measure before using it to prune.

    python notes/order_threshold.py            # 4000 deals
    python notes/order_threshold.py 20000
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bidding as b
import game
import rotation as r

# Trump order, high to low. The left bower is the other jack of the colour.
TRUMP_NAMES = ("right bower", "left bower", "ace", "king", "queen", "ten", "nine")

# Bidding runs from the dealer's left. Index 1 is the dealer's *partner*, and
# naming it that rather than "seat 2 of 4" is the difference between a table
# you can read and one you cannot: ordering up hands the up-card to the dealer,
# so whether the dealer is your partner or your opponent is the single largest
# thing about the decision.
POSITION = ("eldest (dealer's left)", "dealer's partner", "third hand", "dealer")


def is_trump(card, trump):
    return card.suit == trump or (card.rank == r.JACK
                                  and card.suit == r.same_colour(trump))


def trump_rank(card, trump):
    """0 = right bower, 6 = nine. Only meaningful for trump cards."""
    if card.rank == r.JACK:
        return 0 if card.suit == trump else 1
    return {r.ACE: 2, r.KING: 3, r.QUEEN: 4, r.TEN: 5, r.NINE: 6}[card.rank]


def describe(hand, trump, gets_up_card):
    """
    Plain features of a hand, under `trump`.

    `gets_up_card` is for the dealer: ordering hands it the up-card, so the
    trump it will actually hold is one more than it was dealt. Counting the
    dealt hand instead would make the dealer's zero-trump row mean something
    different from everybody else's, which is exactly the kind of quiet
    mismatch that makes a threshold table wrong.
    """
    cards = list(hand)
    trumps = [c for c in cards if is_trump(c, trump)]
    count = len(trumps) + (1 if gets_up_card else 0)
    if gets_up_card:
        best = min([trump_rank(c, trump) for c in trumps]
                   + [trump_rank(r.Card(trump, r.NINE), trump)])
    else:
        best = min([trump_rank(c, trump) for c in trumps], default=None)
    aces = sum(1 for c in cards
               if c.rank == r.ACE and not is_trump(c, trump))
    return count, best, aces


def sweep(n, seed=0, allow_loners=False):
    rows = []
    for i in range(n):
        deal = game.deal_random(rng=random.Random(seed + i), dealer=i % 4)
        trump = deal.up_card.suit
        order = list(deal.bidding_order())
        for index, seat in enumerate(order):
            gets = (seat == deal.dealer)
            count, best, aces = describe(deal.hands[seat], trump, gets)
            ordered = b.value_to(seat, b.order_up(deal, seat)[0])
            passed = b.value_to(seat, b.rest_of_auction(
                deal, index + 1, order, False, allow_loners,
                b.ROUND_ONE).value)
            rows.append(dict(seat=seat, index=index, dealer=deal.dealer,
                             count=count, best=best, aces=aces,
                             order=ordered, gain=ordered - passed,
                             deal=deal))
    return rows


def table(rows, key, label, width=26):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    print("\n  %-*s %6s %8s %8s %8s %8s"
          % (width, label, "n", "order", ">=0", "gain", "gain>=0"))
    for name in sorted(buckets, key=lambda k: (k is None, k)):
        got = buckets[name]
        n = len(got)
        o = sum(x["order"] for x in got) / n
        g = sum(x["gain"] for x in got) / n
        po = 100.0 * sum(1 for x in got if x["order"] >= 0) / n
        pg = 100.0 * sum(1 for x in got if x["gain"] >= 0) / n
        print("  %-*s %6d %+8.3f %7.1f%% %+8.3f %7.1f%%"
              % (width, name, n, o, po, g, pg))


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 4000
    rows = sweep(n)
    print("God Mode, %d deals, every seat's order-up valued on its own team's "
          "scale.\n'order' = what ordering scores. 'gain' = ordering minus "
          "passing." % n)

    table(rows, lambda x: "%d trump" % x["count"], "effective trump held")
    table(rows,
          lambda x: ("no trump" if x["best"] is None
                     else "best is %s" % TRUMP_NAMES[x["best"]]),
          "highest trump held")
    table(rows, lambda x: "%d off-suit ace(s)" % x["aces"], "off-suit aces")
    table(rows, lambda x: POSITION[x["index"]], "bidding position", width=30)
    table(rows, lambda x: "%s, %d trump" % (POSITION[x["index"]], x["count"]),
          "position and trump count", width=30)

    print("\n  --- question 1: ordering with no trump at all ---")
    none = [x for x in rows if x["count"] == 0]
    if not none:
        print("  no zero-trump cases in %d deals" % n)
        return
    best = max(none, key=lambda x: x["order"])
    print("  %d zero-trump order-ups. mean order %+.3f, mean gain %+.3f"
          % (len(none), sum(x["order"] for x in none) / len(none),
             sum(x["gain"] for x in none) / len(none)))
    print("  order >= 0 in %d of %d (%.2f%%); gain >= 0 in %d (%.2f%%)"
          % (sum(1 for x in none if x["order"] >= 0), len(none),
             100.0 * sum(1 for x in none if x["order"] >= 0) / len(none),
             sum(1 for x in none if x["gain"] >= 0),
             100.0 * sum(1 for x in none if x["gain"] >= 0) / len(none)))
    print("  best zero-trump case seen: order %+d, gain %+d, seat %d, hand %s, "
          "up-card %s" % (best["order"], best["gain"], best["seat"],
                          r.hand_name(best["deal"].hands[best["seat"]]),
                          r.card_name(best["deal"].up_card)))


if __name__ == "__main__":
    main(sys.argv)
