"""
Why does God Mode ever get euchred?

It looks like a bug: a seat that sees all 24 cards orders up a contract it
knows will fail. It is not one. `bidding._best` lists passing first and keeps
the first of equals, so a call is only ever taken when it is *strictly* better
than declining -- and being euchred hands the opposition exactly 2, the same as
standing aside and letting them march. The only continuation worse than that is
an opposing **lone** march at 4. So the only hand worth ordering up into a
certain euchre is one that takes a loner off the table, for a saving of two.

This script checks that claim rather than arguing it. Over 300 deals it finds
every God Mode euchre, re-prices the pass the caller turned down, and then
re-runs the same deals with loners forbidden.

    python notes/order_up_study.py          # 300 deals, ~1 min
    python notes/order_up_study.py 50

Measured, 300 deals from seed 0 with the dealer rotating:

    euchres with loners on:  39 of 300
    euchres with loners off: 0 of 300
    what passing was worth to the caller: -4, in all 39
    pass branch was an opposing loner: 39 of 39

The README's 300-deal table reports the same 39 as God Mode's 13.0% euchre
rate. Every one of them is a sacrifice.
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bidding as b
import game
import rotation as r

DEALS = 300


def deal_at(i):
    """Deal `i` of the sweep -- the same deals `pimc_sweep.py` uses."""
    return game.deal_random(rng=random.Random(i), dealer=i % 4)


def euchres_with_loners(n):
    """Every God Mode euchre, with what passing would have been worth."""
    found = []
    for i in range(n):
        deal = deal_at(i)
        out = b.solve_bidding(deal, allow_loners=True)
        if out.passed_out:
            continue
        if b.value_to(out.contract.caller, out.value) != -2:
            continue

        # The pass the caller turned down: the rest of the auction from the
        # next seat on, on the caller's own scale.
        order = list(deal.bidding_order())
        seat = out.contract.caller
        passed = b.rest_of_auction(deal, order.index(seat) + 1, order,
                                   False, True, out.contract.bidding_round)
        found.append((i, deal, out.contract, b.value_to(seat, passed.value),
                      None if passed.passed_out else passed.contract))
    return found


def euchres_without_loners(n):
    """The same deals with `allow_loners=False`. Expected: none."""
    total = 0
    for i in range(n):
        out = b.solve_bidding(deal_at(i), allow_loners=False)
        if not out.passed_out:
            total += b.value_to(out.contract.caller, out.value) == -2
    return total


def main(argv=None):
    n = int(argv[0]) if argv else DEALS
    found = euchres_with_loners(n)
    without = euchres_without_loners(n)

    print("God Mode euchres over %d deals" % n)
    print("  with loners allowed      %d" % len(found))
    print("  with loners forbidden    %d" % without)

    print("\n  what passing was worth to the caller")
    for value, count in sorted(Counter(e[3] for e in found).items()):
        print("    %+d  %d" % (value, count))

    lone = sum(1 for e in found if e[4] is not None and e[4].alone)
    print("\n  pass branch was an opposing loner    %d of %d"
          % (lone, len(found)))

    print("\n  the first few, in full")
    for i, deal, contract, value, threat in found[:5]:
        print("    deal %d (dealer %d, up %s)"
              % (i, deal.dealer, r.card_name(deal.up_card)))
        print("      %s -- euchred, -2" % contract)
        print("      passing instead: %s -> %+d" % (threat, value))


if __name__ == "__main__":
    main(sys.argv[1:])
