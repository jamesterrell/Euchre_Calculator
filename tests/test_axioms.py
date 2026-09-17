"""
The axioms, and the tests that keep them falsifiable.

`notes/discard_dominance.md` states Axiom 1: when the dealer picks up, some
optimal discard is never the right bower, the left bower or the ace of trump.
It was adopted because a sweep of 15,515 decisive God Mode positions failed to
break it -- not because anything proves it.

That is a dangerous kind of belief to wire into `bidding.py`. `solve_bidding`
is this project's *exact* baseline: CLAUDE.md opens by calling God Mode "the
exact baseline, not a model of a real table", and `tests/test_table.py` pins a
table of four `GodModePlayer`s to it. If the axiom is false, a pruned auction
is wrong in a way nothing would notice, because the only thing it would be
checked against is itself.

So the prune is off by default everywhere, and this file is what stops it being
a matter of faith: it runs both paths over a sample of deals and asserts they
agree. A counterexample surfaces here as a failing test naming the deal, rather
than as a number that is quietly a bit wrong.

These are deliberately a few hundred deals rather than a few dozen. Two God
Mode auctions per deal is not cheap, but an axiom checked on twenty deals is
decoration.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import bidding as b
import game
import rotation as r


def deals(n, start=0):
    return [game.deal_random(rng=random.Random(start + i), dealer=i % 4)
            for i in range(n)]


class TestTopTrumps(unittest.TestCase):
    def test_it_names_three_real_cards(self):
        for trump in r.SUITS:
            tops = b.top_trumps(trump)
            self.assertEqual(len(set(tops)), 3)
            right, left, ace = tops
            self.assertEqual(right, r.Card(trump, r.JACK))
            self.assertEqual(left, r.Card(r.same_colour(trump), r.JACK))
            self.assertEqual(ace, r.Card(trump, r.ACE))
            for card in tops:
                self.assertIn(card, r.full_deck())

    def test_the_left_bower_is_not_of_the_trump_suit(self):
        # The one that catches a copy-paste: the left bower is the *other*
        # jack of the colour, so it prints as a different suit than trump.
        for trump in r.SUITS:
            _, left, _ = b.top_trumps(trump)
            self.assertNotEqual(left.suit, trump)
            self.assertEqual(r.same_colour(left.suit), trump)


class TestDiscardCandidates(unittest.TestCase):
    def test_unpruned_is_every_dealt_card(self):
        for d in deals(40):
            self.assertEqual(list(b.discard_candidates(d)),
                             list(d.hands[d.dealer]))

    def test_pruning_drops_exactly_the_top_trumps(self):
        for d in deals(200):
            tops = set(b.top_trumps(d.up_card.suit))
            got = b.discard_candidates(d, prune=True)
            self.assertFalse(set(got) & tops)
            self.assertEqual(set(got), set(d.hands[d.dealer]) - tops)

    def test_it_never_prunes_to_nothing(self):
        # Only three cards are ever struck and the dealer holds five, so two
        # always survive. Asserted rather than reasoned about, because an empty
        # candidate list would make order_up return a max() over nothing.
        for d in deals(300):
            self.assertGreaterEqual(len(b.discard_candidates(d, prune=True)), 2)


class TestAxiomOneHolds(unittest.TestCase):
    """The axiom itself, checked the only way it can be: by disagreement."""

    def test_the_pruned_auction_matches_the_exact_one(self):
        bad = []
        for i, d in enumerate(deals(250)):
            for loners in (False, True):
                exact = b.solve_bidding(d, allow_loners=loners)
                pruned = b.solve_bidding(d, allow_loners=loners, prune=True)
                if exact.value != pruned.value:
                    bad.append((i, loners, exact, pruned))
        self.assertEqual(bad, [], "Axiom 1 is false -- a pruned auction "
                                  "disagreed with the exact one")

    def test_the_dealers_own_choice_is_never_worse_pruned(self):
        # Tighter than the auction test: compare order_up directly, on the
        # dealer's own scale, for every caller. The auction can hide a bad
        # discard behind a seat that declines to call at all.
        for d in deals(120):
            for caller in range(game.PLAYERS):
                exact, _ = b.order_up(d, caller)
                pruned, _ = b.order_up(d, caller, prune=True)
                self.assertEqual(
                    exact, pruned,
                    "dealer %d, caller %d: pruning changed the value of "
                    "ordering up" % (d.dealer, caller))

    def test_it_holds_for_loners_too(self):
        for d in deals(120):
            for caller in range(game.PLAYERS):
                if (caller + 2) % game.PLAYERS == d.dealer:
                    continue          # dealer sits out; discard unobservable
                self.assertEqual(b.order_up(d, caller, alone=True)[0],
                                 b.order_up(d, caller, alone=True,
                                            prune=True)[0])


class TestPruningIsOffByDefault(unittest.TestCase):
    """
    The baseline must stay exact unless somebody asks for otherwise.

    An axiom that silently switched itself on would make every God Mode number
    in CLAUDE.md conditional on it.
    """

    def test_solve_bidding_defaults_to_exact(self):
        for d in deals(30):
            self.assertEqual(b.solve_bidding(d).value,
                             b.solve_bidding(d, prune=False).value)

    def test_order_up_considers_every_card_by_default(self):
        d = deals(1)[0]
        self.assertEqual(len(b.discard_candidates(d)), game.HAND_SIZE)

    def test_the_pimc_player_defaults_to_exact(self):
        import players
        self.assertFalse(players.PIMCPlayer().prune_discards)


if __name__ == "__main__":
    unittest.main()
