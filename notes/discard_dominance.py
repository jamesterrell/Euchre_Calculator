"""
Is it ever uniquely best for the dealer to discard a top trump?

The claim under test, in the form that would actually buy something: when the
dealer picks up, the right bower, the left bower and the ace of trump can be
struck off the candidate list without ever losing an optimal discard. That is a
*dominance* claim, and the useful version is not "discarding a top trump is bad"
but "there is always an equally good discard that is not a top trump" -- ties
are fine, since a pruner only has to keep one optimal card.

This cannot be proved by enumeration. The dealer's best discard depends on all
four hands, and there are 24!/(5!^4 4!) deals. So the script hunts for a
counterexample instead: find one and the claim is dead, find none in a large
sample and the claim is a well-supported heuristic, which is a different and
weaker thing. It says which it found.

Why a counterexample is plausible rather than absurd: in trick-taking games a
higher card is not always weakly better to hold. Winning a trick you would
rather have ducked hands you the lead, and leading away from a tenace can cost
your side a trick -- the endplay. The dealer is often on the defending side
here, where that is exactly the shape of position that bites.

    python notes/discard_dominance.py            # 20000 deals
    python notes/discard_dominance.py 50000
"""
import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bidding as b
import game
import rotation as r


def top_trumps(trump):
    """Right bower, left bower, ace of trump -- the three cards under test."""
    return {
        r.Card(trump, r.JACK): "right bower",
        r.Card(r.same_colour(trump), r.JACK): "left bower",
        r.Card(trump, r.ACE): "ace of trump",
    }


def discard_values(deal, caller):
    """{card: value on the dealer's own team's scale} for every legal discard."""
    trump = deal.up_card.suit
    dealer = deal.dealer
    out = {}
    for card in deal.hands[dealer]:
        after = deal.pick_up(discard=card)
        net = b.net_to_team0(b.play_value(after, trump, caller), caller)
        out[card] = b.value_to(dealer, net)
    return out


def study(n, alone=False, seed=0, report_every=5000):
    held = Counter()          # how often the dealer held each top trump
    forced = Counter()        # ...and every optimal discard was a top trump
    tied = Counter()          # ...and it was among the optimal discards
    counterexamples = []
    considered = 0
    decisive = 0              # positions where the discard changes the value

    for i in range(n):
        # Report as it goes. A long sweep that prints only at the end has
        # nothing to show if it is interrupted, which is exactly what happened
        # to the first attempt at this one.
        if report_every and i and i % report_every == 0:
            print("    ... %d deals, %d decisive positions, %d counterexamples"
                  % (i, decisive, len(counterexamples)), flush=True)
        deal = game.deal_random(rng=random.Random(seed + i), dealer=i % 4)
        trump = deal.up_card.suit
        tops = top_trumps(trump)
        mine = [c for c in deal.hands[deal.dealer] if c in tops]
        if not mine:
            continue

        for caller in range(game.PLAYERS):
            if alone and (caller + 2) % game.PLAYERS == deal.dealer:
                continue          # dealer sits out; the discard is unobservable
            considered += 1
            values = discard_values(deal, caller)
            best = max(values.values())
            optimal = [c for c, v in values.items() if v == best]
            # Positions where every discard is worth the same carry no
            # information about the claim: the outcome is already forced, so
            # "a top trump was optimal" is vacuous there. They are most of the
            # sample, which is why the raw deal count overstates the evidence.
            if len(set(values.values())) > 1:
                decisive += 1

            for card in mine:
                held[tops[card]] += 1
                if card in optimal:
                    tied[tops[card]] += 1

            if all(c in tops for c in optimal):
                for c in optimal:
                    forced[tops[c]] += 1
                counterexamples.append((deal, caller, values, optimal, tops))

    return held, forced, tied, counterexamples, considered, decisive


def show(deal, caller, values, optimal, tops):
    trump = deal.up_card.suit
    print("    dealer %d, caller %d, trump %s, up-card %s"
          % (deal.dealer, caller, r.suit_name(trump), r.card_name(deal.up_card)))
    print("    dealer's hand  %s" % r.hand_name(deal.hands[deal.dealer]))
    for card, value in sorted(values.items(), key=lambda kv: -kv[1]):
        print("      pitch %-3s -> %+d to the dealer's team%s"
              % (r.card_name(card), value,
                 "   <-- optimal, and a %s" % tops[card]
                 if card in optimal and card in tops else
                 "   <-- optimal" if card in optimal else ""))


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 20000
    for alone in (False, True):
        label = "called alone" if alone else "four-handed"
        held, forced, tied, bad, considered, decisive = study(n, alone=alone)
        print("\n%s -- %d deals, %d (deal, caller) pairs where the dealer held "
              "a top trump" % (label, n, considered))
        for name in ("right bower", "left bower", "ace of trump"):
            print("  %-13s held %6d | among the optimal discards %6d (%5.1f%%) "
                  "| uniquely optimal %d"
                  % (name, held[name], tied[name],
                     100.0 * tied[name] / held[name] if held[name] else 0.0,
                     forced[name]))
        if bad:
            print("  COUNTEREXAMPLES: %d of %d (%.4f%%) -- the claim is false"
                  % (len(bad), considered, 100.0 * len(bad) / considered))
            for case in bad[:3]:
                print()
                show(*case)
        else:
            print("  no counterexample in %d decisive positions: every one of "
                  "them had an optimal discard that was not a top trump"
                  % decisive)


if __name__ == "__main__":
    main(sys.argv)
