"""
Unit tests for observation.py.

A sampler is a nasty thing to test, because the failure mode is not a crash --
it is a player that quietly knows slightly too much or slightly too little, and
then wins or loses slightly more than it should. Nothing downstream will
complain. So the tests here are mostly invariants asserted over many draws:

  * every sampled world is a whole Euchre deck, no card gained or lost. This is
    the same invariant `game.Deal.check` exists for, and for the same reason --
    `dealer.py` shipped a card-losing bug twice.
  * no sampled world contradicts something the seat has already watched happen:
    a void it showed, a card it played, the up-card going under.
  * the observer's own hand is never replaced by a guess.

`check_world` is written from the Observation rather than from the sampler's
bookkeeping, so it can disagree with `sample_world` -- the two derive the same
constraints by different routes.

The left bower gets its own tests in both directions. It is the one card whose
suit is not the suit printed on it, so it is the one card that can put a player
in a suit it has already shown out of.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import game
import observation as ob
import rotation as r

DRAWS = 30


def a_deal(seed=0, dealer=0):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


class TestStructure(unittest.TestCase):
    def test_counts_are_public(self):
        d = a_deal(1, dealer=3)
        o = ob.observe(d, seat=0)
        self.assertEqual(o.counts(), [5, 5, 5, 5])

        plays = ((0, d.hands[0][0]), (1, d.hands[1][0]))
        o = ob.observe(d, seat=0, plays=plays, trump=r.SPADES, caller=0)
        self.assertEqual(o.counts(), [4, 4, 5, 5])
        self.assertEqual(len(o.hand), 4)

    def test_trick_boundaries(self):
        d = a_deal(2, dealer=3)
        plays = tuple((s, d.hands[s][0]) for s in range(4))
        plays += ((0, d.hands[0][1]),)
        o = ob.observe(d, seat=0, plays=plays, trump=r.SPADES, caller=0)

        self.assertEqual(o.width, 4)
        self.assertEqual(o.trick_no, 1)
        self.assertEqual(len(o.completed_tricks()), 1)
        self.assertEqual(o.current_trick, ((0, d.hands[0][1]),))

    def test_a_loner_trick_is_three_cards(self):
        d = a_deal(3, dealer=3)
        o = ob.observe(d, seat=0, trump=r.SPADES, caller=0, alone=True)
        self.assertEqual(o.sitting, 2)
        self.assertEqual(o.width, 3)

    def test_check_rejects_a_hand_that_disagrees_with_the_play(self):
        d = a_deal(4)
        bad = ob.Observation(seat=0, hand=tuple(d.hands[0]), dealer=0,
                             up_card=d.up_card,
                             plays=((0, d.hands[0][0]),), trump=r.SPADES,
                             caller=0)
        with self.assertRaises(ValueError):
            bad.check()

    def test_only_the_dealer_buries_a_card(self):
        d = a_deal(5, dealer=3)
        with self.assertRaises(ValueError):
            ob.Observation(seat=0, hand=tuple(d.hands[0]), dealer=3,
                           up_card=d.up_card, up_state=ob.PICKED_UP,
                           discard=d.buried[0]).check()


class TestVoids(unittest.TestCase):
    """Failing to follow is remembered, and read in effective suits."""

    def observation(self, plays, trump=r.SPADES):
        return ob.Observation(
            seat=0, hand=tuple(r.parse_hand("KH QH 9H TD")), dealer=3,
            up_card=r.parse_card("AS"), up_state=ob.PICKED_UP,
            plays=tuple(plays), trump=trump, caller=0).check()

    def test_a_seat_that_does_not_follow_is_void(self):
        C = r.parse_card
        o = self.observation([(0, C("AH")), (1, C("9D")), (2, C("TH")),
                              (3, C("QS"))])
        voids = o.voids()
        self.assertEqual(voids[1], frozenset({r.HEARTS}))
        self.assertEqual(voids[3], frozenset({r.HEARTS}))
        self.assertEqual(voids[2], frozenset())

    def test_the_left_bower_does_not_follow_its_printed_suit(self):
        # Spades is trump, so the jack of clubs is trump. Ruffing a heart lead
        # with it shows a heart void and says nothing at all about clubs.
        C = r.parse_card
        o = self.observation([(0, C("AH")), (1, C("9D")), (2, C("TH")),
                              (3, C("JC"))])
        self.assertEqual(o.voids()[3], frozenset({r.HEARTS}))
        self.assertNotIn(r.CLUBS, o.voids()[3])

    def test_the_left_bower_cannot_follow_a_club_lead(self):
        # A club is led and seat 1 plays the left bower. That is trump, not a
        # club, so seat 1 has shown out of clubs -- the inference the printed
        # suit would miss.
        C = r.parse_card
        o = self.observation([(0, C("AC")), (1, C("JC")), (2, C("9C")),
                              (3, C("TC"))])
        self.assertEqual(o.voids()[1], frozenset({r.CLUBS}))
        self.assertEqual(o.voids()[2], frozenset())

    def test_voids_survive_into_later_tricks(self):
        C = r.parse_card
        o = ob.Observation(
            seat=0, hand=tuple(r.parse_hand("KH QH TC")), dealer=3,
            up_card=C("AS"), up_state=ob.PICKED_UP,
            plays=((0, C("AH")), (1, C("9D")), (2, C("TH")), (3, C("QS")),
                   (3, C("KS")), (0, C("JH")), (1, C("TD")), (2, C("9H"))),
            trump=r.SPADES, caller=0).check()
        self.assertIn(r.HEARTS, o.voids()[1])


class TestSampling(unittest.TestCase):
    def draws(self, o, n=DRAWS, seed=0):
        rng = random.Random(seed)
        return [ob.check_world(o, w) for w in ob.sample_worlds(o, n, rng)]

    def test_a_fresh_deal_samples_cleanly(self):
        for seed in range(6):
            d = a_deal(seed, dealer=seed % 4)
            for seat in range(4):
                with self.subTest(seed=seed, seat=seat):
                    self.draws(ob.observe(d, seat), n=8, seed=seed)

    def test_the_observer_keeps_its_own_hand(self):
        d = a_deal(7, dealer=1)
        o = ob.observe(d, seat=2)
        for w in self.draws(o):
            self.assertEqual(w.hands[2], tuple(d.hands[2]))

    def test_a_turned_down_up_card_is_in_nobodys_hand(self):
        d = a_deal(8, dealer=1)
        o = ob.observe(d, seat=0, up_state=ob.TURNED_DOWN)
        for w in self.draws(o):
            for hand in w.hands:
                self.assertNotIn(d.up_card, hand)
            self.assertIn(d.up_card, w.kitty)

    def test_an_ordered_up_card_is_in_the_dealers_hand(self):
        d = a_deal(9, dealer=3)
        after = d.pick_up(discard=d.hands[3][0])
        o = ob.observe(after, seat=0, trump=d.up_card.suit, caller=0,
                       up_state=ob.PICKED_UP)
        for w in self.draws(o):
            self.assertIn(d.up_card, w.hands[3])

    def test_the_dealer_does_not_guess_at_its_own_discard(self):
        d = a_deal(10, dealer=3)
        pitched = d.hands[3][0]
        after = d.pick_up(discard=pitched)
        o = ob.observe(after, seat=3, trump=d.up_card.suit, caller=0,
                       up_state=ob.PICKED_UP, discard=pitched)
        for w in self.draws(o):
            self.assertIn(pitched, w.kitty)
            self.assertNotIn(pitched, [c for h in w.hands for c in h])

    def test_a_dealer_who_shows_out_of_trump_buried_the_up_card(self):
        # Spades is turned up and ordered; the dealer then fails to follow a
        # trump lead. It cannot still be holding the up-card, so no sampled
        # world may put it there.
        C = r.parse_card
        o = ob.Observation(
            seat=0, hand=tuple(r.parse_hand("JS JC KH QH")), dealer=3,
            up_card=C("AS"), up_state=ob.PICKED_UP,
            plays=((0, C("9S")), (1, C("TS")), (2, C("QS")), (3, C("9H"))),
            trump=r.SPADES, caller=0).check()
        self.assertIn(r.SPADES, o.voids()[3])
        for w in self.draws(o, n=15):
            self.assertNotIn(C("AS"), w.hands[3])

    def test_a_dealer_with_no_cards_left_buried_the_up_card(self):
        """
        The other way the up-card can be shown to be buried.

        Seat 3 ordered up the ace of spades, then led trump every trick and won
        every trick, so it never had to follow and never showed out of
        anything -- the void rule that normally catches this has nothing to go
        on. But it has now played all five of its cards and none of them was
        the ace, so the ace cannot be in its hand either. Without that second
        reading the sampler tries to force a card into an empty hand.
        """
        C = r.parse_card
        plays = (
            (3, C("JS")), (0, C("9H")), (1, C("9D")), (2, C("9C")),
            (3, C("JC")), (0, C("TH")), (1, C("TD")), (2, C("TC")),
            (3, C("KS")), (0, C("QH")), (1, C("QD")), (2, C("QC")),
            (3, C("QS")), (0, C("KH")), (1, C("KD")), (2, C("KC")),
            (3, C("TS")), (0, C("AH")),
        )
        o = ob.Observation(seat=1, hand=(C("AD"),), dealer=3,
                           up_card=C("AS"), up_state=ob.PICKED_UP,
                           plays=plays, trump=r.SPADES, caller=3).check()

        self.assertEqual(o.counts()[3], 0)
        self.assertEqual(o.voids()[3], frozenset())
        self.assertFalse(ob._dealer_holds_up_card(o))
        for w in self.draws(o, n=10):
            self.assertNotIn(C("AS"), w.hands[3])

    def test_no_world_contradicts_a_void(self):
        C = r.parse_card
        o = ob.Observation(
            seat=0, hand=tuple(r.parse_hand("KH QH 9H TD")), dealer=3,
            up_card=C("AS"), up_state=ob.PICKED_UP,
            plays=((0, C("AH")), (1, C("9D")), (2, C("TH")), (3, C("JC"))),
            trump=r.SPADES, caller=0).check()
        for w in self.draws(o, n=40):
            for seat in (1, 3):
                for card in w.hands[seat]:
                    self.assertNotEqual(
                        ob.effective_suit(card, r.SPADES), r.HEARTS,
                        "seat %d was dealt %s after showing out of hearts"
                        % (seat, r.card_name(card)))

    def test_a_loners_sitting_partner_still_holds_five(self):
        # The seat never plays, but its five cards are out of circulation for
        # everybody else. Forgetting that makes every other hand too good.
        d = a_deal(11, dealer=3)
        o = ob.observe(d, seat=0, trump=r.SPADES, caller=0, alone=True)
        self.assertEqual(o.sitting, 2)
        for w in self.draws(o, n=10):
            self.assertEqual(len(w.hands[2]), 5)

    def test_the_pending_discard_holds_six(self):
        d = a_deal(12, dealer=2)
        six = tuple(d.hands[2]) + (d.up_card,)
        o = ob.Observation(seat=2, hand=six, dealer=2, up_card=d.up_card,
                           up_state=ob.PICKED_UP, trump=d.up_card.suit,
                           caller=0, pending_discard=True).check()
        self.assertEqual(o.counts()[2], 6)
        for w in self.draws(o, n=10):
            self.assertEqual(len(w.hands[2]), 6)
            self.assertEqual(len(w.kitty), 3)

    def test_worlds_actually_differ(self):
        # A sampler that returned the same layout every time would pass every
        # consistency check above and be useless.
        d = a_deal(13, dealer=1)
        o = ob.observe(d, seat=0)
        seen = {w.hands[1] for w in self.draws(o, n=20)}
        self.assertGreater(len(seen), 10)


class TestUnseen(unittest.TestCase):
    def test_unseen_counts_add_up(self):
        d = a_deal(14, dealer=3)
        o = ob.observe(d, seat=0, up_state=ob.TURNED_DOWN)
        # 24 less this seat's five and the buried up-card.
        self.assertEqual(len(o.unseen()), 18)
        self.assertEqual(sum(ob._capacities(o)), 18)

    def test_played_cards_are_seen_by_everyone(self):
        d = a_deal(15, dealer=3)
        plays = tuple((s, d.hands[s][0]) for s in range(4))
        o = ob.observe(d, seat=0, plays=plays, trump=r.SPADES, caller=0)
        for _, card in plays:
            self.assertNotIn(card, o.unseen())


if __name__ == "__main__":
    unittest.main()
