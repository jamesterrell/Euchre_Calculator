"""
The proposed axioms, and the counterexample that killed the first one.

**Axiom 1 is false.** It was proposed as: when the dealer picks up, some optimal
discard is never the right bower, the left bower or the ace of trump -- so those
three could be struck off the candidate list for free. A sweep of 10,000 deals
found no counterexample and it was very nearly adopted. Extending the same sweep
to 100,000 deals found one, at a rate of about 1 in 100,000 positions.

`notes/discard_dominance.md` has the full story. The short version is that
`bidding.discard_candidates(prune=True)` is a **heuristic with a known
counterexample**, not a sound prune, and this file exists to make sure nobody
can forget that. It is off by default everywhere.

The counterexample is pinned rather than reached for by seed, for the reason
CLAUDE.md gives about `BRUTE_FORCEABLE`: a seeded pick makes the suite's
coverage a matter of luck, and one reseed loses the only position in the file
that proves anything.

Why it works is worth understanding, because it is the shape any further
counterexample will have. Trump is hearts, so the dealer holds JH (the right
bower), AH (the ace of trump) and three diamonds. Pitching the ace of trump
keeps **three** diamonds and two trump; pitching a diamond keeps three trump and
two diamonds. The ace is redundant sitting behind the right bower, and the extra
diamond is worth more as length than the ace is as a winner -- so the hand with
*fewer and weaker* trump takes all five tricks and the one with more takes four.
That is the blocking/length effect that uniform random dealing almost never
produces, which is exactly why 10,000 deals missed it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import bidding as b
import game
import rotation as r

# Seed 94137 dealt by seat 1, written out rather than reseeded. Trump is
# hearts (KH up). The dealer holds TD KD JH 9D AH and pitching AH -- the ace of
# trump -- is the only discard that makes a march.
COUNTEREXAMPLE = (
    "JC QS JD TC 9H  TD KD JH 9D AH  JS AS QH 9C QC  KC AD AC TH QD  "
    "KH  TS 9S KS")
COUNTEREXAMPLE_DEALER = 1


def the_counterexample():
    return game.deal_from_order(r.parse_hand(COUNTEREXAMPLE),
                                dealer=COUNTEREXAMPLE_DEALER)


def deals(n, start=0):
    return [game.deal_random(rng=random.Random(start + i), dealer=i % 4)
            for i in range(n)]


def discard_values(deal, caller):
    """{card: value on the dealer's own team's scale}."""
    return {c: b.value_to(deal.dealer,
                          b.net_to_team0(
                              b.play_value(deal.pick_up(c), deal.up_card.suit,
                                           caller), caller))
            for c in deal.hands[deal.dealer]}


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


class TestAxiomOneIsFalse(unittest.TestCase):
    """
    The pinned witness. If this ever starts passing as an axiom again,
    something has changed that should not have.
    """

    def test_the_deal_is_what_the_note_says_it_is(self):
        deal = the_counterexample()
        self.assertEqual(deal.dealer, 1)
        self.assertEqual(deal.up_card, r.parse_card("KH"))
        self.assertEqual(set(deal.hands[1]), set(r.parse_hand("TD KD JH 9D AH")))

    def test_pitching_the_ace_of_trump_is_uniquely_optimal(self):
        deal = the_counterexample()
        ace = r.parse_card("AH")
        for caller in (1, 3):           # the dealer's own team
            values = discard_values(deal, caller)
            best = max(values.values())
            optimal = [c for c, v in values.items() if v == best]
            self.assertEqual(optimal, [ace],
                             "caller %d: expected the ace of trump to be the "
                             "only optimal discard, got %s"
                             % (caller, [r.card_name(c) for c in optimal]))
            self.assertEqual(values[ace], 2)
            for card in deal.hands[deal.dealer]:
                if card != ace:
                    self.assertEqual(values[card], 1)

    def test_the_prune_costs_a_march_here(self):
        # The concrete price of the heuristic: it cannot see the only discard
        # that makes the hand, so it reports +1 where the truth is +2.
        deal = the_counterexample()
        for caller in (1, 3):
            exact, _ = b.order_up(deal, caller)
            pruned, _ = b.order_up(deal, caller, prune=True)
            self.assertNotEqual(exact, pruned)
            self.assertEqual(b.value_to(deal.dealer, exact), 2)
            self.assertEqual(b.value_to(deal.dealer, pruned), 1)

    def test_the_ace_of_trump_is_not_even_in_the_pruned_candidates(self):
        deal = the_counterexample()
        self.assertNotIn(r.parse_card("AH"),
                         b.discard_candidates(deal, prune=True))


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


class TestThePruneIsRarelyWrong(unittest.TestCase):
    """
    Not "never wrong" -- that claim is dead. Just rarely, and measurably so.

    The measured rate over 100,000 deals was 2 divergent positions in 195,964
    (0.001%) four-handed and 1 in 146,973 alone. A few hundred deals here will
    almost certainly see none, so this asserts a bound rather than zero: it is
    a guard against the prune becoming *badly* wrong, not evidence that it is
    right.
    """

    def test_divergence_stays_rare_on_a_sample(self):
        diverged = 0
        total = 0
        for d in deals(150):
            for caller in range(game.PLAYERS):
                total += 1
                if b.order_up(d, caller)[0] != b.order_up(d, caller,
                                                          prune=True)[0]:
                    diverged += 1
        self.assertLess(diverged, max(2, total // 100),
                        "the top-trump prune diverged on %d of %d positions, "
                        "far more often than the 0.001%% measured over 100,000 "
                        "deals -- something is wrong with it"
                        % (diverged, total))


class TestPruningIsOffByDefault(unittest.TestCase):
    """
    The baseline must stay exact unless somebody asks for otherwise.

    This mattered when the prune was believed sound and matters more now that
    it is known not to be: nothing may switch it on by itself.
    """

    def test_solve_bidding_defaults_to_exact(self):
        for d in deals(30):
            self.assertEqual(b.solve_bidding(d).value,
                             b.solve_bidding(d, prune=False).value)

    def test_the_exact_auction_still_finds_the_march(self):
        # solve_bidding's default path has to see what the prune cannot.
        deal = the_counterexample()
        self.assertEqual(b.value_to(deal.dealer, b.order_up(deal, 1)[0]), 2)

    def test_order_up_considers_every_card_by_default(self):
        d = deals(1)[0]
        self.assertEqual(len(b.discard_candidates(d)), game.HAND_SIZE)

    def test_the_pimc_player_defaults_to_exact(self):
        import players
        self.assertFalse(players.PIMCPlayer().prune_discards)


if __name__ == "__main__":
    unittest.main()
