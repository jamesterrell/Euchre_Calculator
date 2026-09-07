"""
Unit tests for game.py.

The invariant everything else rests on is that all 24 cards stay accounted for
across every operation. A deal that quietly loses or duplicates a card produces
a game that cannot be reasoned about, and `dealer.py` shipped exactly that bug
twice -- so `check()` is asserted after every state change here, not just at the
start.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import game
import rotation as r
from fast_search import definitive_winner


def a_deal(seed=0, dealer=0):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


class TestStructure(unittest.TestCase):
    def test_four_hands_of_five(self):
        d = a_deal()
        self.assertEqual(len(d.hands), 4)
        for hand in d.hands:
            self.assertEqual(len(hand), 5)

    def test_kitty_is_an_up_card_plus_three_buried(self):
        d = a_deal()
        self.assertEqual(len(d.buried), 3)
        self.assertIsInstance(d.up_card, r.Card)

    def test_all_24_cards_accounted_for(self):
        for seed in range(30):
            d = a_deal(seed)
            cards = d.all_cards()
            self.assertEqual(len(cards), 24)
            self.assertEqual(len(set(cards)), 24)
            self.assertEqual(set(cards), set(r.full_deck()))

    def test_up_card_is_in_nobody_hand_before_the_pickup(self):
        for seed in range(30):
            d = a_deal(seed)
            for hand in d.hands:
                self.assertNotIn(d.up_card, hand)

    def test_deals_differ(self):
        seen = {tuple(a_deal(s).hands[0]) for s in range(20)}
        self.assertGreater(len(seen), 1)

    def test_deal_is_immutable(self):
        d = a_deal()
        with self.assertRaises(Exception):
            d.dealer = 2


class TestSeats(unittest.TestCase):
    def test_bidding_starts_left_of_the_dealer(self):
        for dealer in range(4):
            d = a_deal(dealer=dealer)
            self.assertEqual(d.first_bidder, (dealer + 1) % 4)
            self.assertEqual(d.bidding_order()[0], d.first_bidder)

    def test_bidding_order_covers_every_seat_once_and_ends_on_the_dealer(self):
        for dealer in range(4):
            order = a_deal(dealer=dealer).bidding_order()
            self.assertEqual(sorted(order), [0, 1, 2, 3])
            self.assertEqual(order[-1], dealer, "the dealer speaks last")

    def test_partners_are_across_the_table(self):
        d = a_deal()
        for seat in range(4):
            self.assertEqual(d.partner(seat), (seat + 2) % 4)
            self.assertEqual(d.partner(d.partner(seat)), seat)
            self.assertEqual(d.partner(seat) % 2, seat % 2,
                             "partners must share the team parity the solver uses")

    def test_rejects_a_seat_that_does_not_exist(self):
        cards = r.full_deck()
        for bad in (-1, 4, 9):
            with self.assertRaises(ValueError):
                game.deal_from_order(cards, dealer=bad)


class TestPickUp(unittest.TestCase):
    def test_dealer_ends_with_five_cards_holding_the_up_card(self):
        d = a_deal()
        after = d.pick_up(discard=d.hands[d.dealer][0])
        self.assertEqual(len(after.hands[d.dealer]), 5)
        self.assertIn(d.up_card, after.hands[d.dealer])
        self.assertTrue(after.picked_up)

    def test_discard_goes_under_the_kitty(self):
        d = a_deal()
        thrown = d.hands[d.dealer][2]
        after = d.pick_up(discard=thrown)
        self.assertIn(thrown, after.buried)
        self.assertEqual(len(after.buried), 4)
        for hand in after.hands:
            self.assertNotIn(thrown, hand)

    def test_still_24_cards_afterwards(self):
        for seed in range(20):
            d = a_deal(seed)
            after = d.pick_up(discard=d.hands[d.dealer][1])
            self.assertEqual(set(after.all_cards()), set(r.full_deck()))
            after.check()

    def test_the_dealer_may_discard_the_up_card_itself(self):
        """Legal, and sometimes right: take it and throw it straight back."""
        d = a_deal()
        after = d.pick_up(discard=d.up_card)
        self.assertEqual(after.hands[d.dealer], d.hands[d.dealer])
        self.assertIn(d.up_card, after.buried)

    def test_other_hands_are_untouched(self):
        d = a_deal()
        after = d.pick_up(discard=d.hands[d.dealer][0])
        for seat in range(4):
            if seat != d.dealer:
                self.assertEqual(after.hands[seat], d.hands[seat])

    def test_does_not_mutate_the_original(self):
        d = a_deal()
        before = d.hands
        d.pick_up(discard=d.hands[d.dealer][0])
        self.assertEqual(d.hands, before)
        self.assertFalse(d.picked_up)

    def test_rejects_a_card_the_dealer_does_not_hold(self):
        d = a_deal()
        outsider = next(c for c in d.hands[(d.dealer + 1) % 4])
        with self.assertRaises(ValueError):
            d.pick_up(discard=outsider)

    def test_rejects_picking_up_twice(self):
        d = a_deal()
        after = d.pick_up(discard=d.hands[d.dealer][0])
        with self.assertRaises(ValueError):
            after.pick_up(discard=after.hands[after.dealer][0])

    def test_up_card_stays_on_record_when_turned_down(self):
        """Every seat saw it; the bidding depends on that information."""
        d = a_deal()
        self.assertEqual(d.turn_down().up_card, d.up_card)
        self.assertFalse(d.turn_down().picked_up)


class TestDealAround(unittest.TestCase):
    def test_pins_a_hand_and_an_up_card(self):
        known = r.parse_hand("JH JD AH")
        up = r.parse_card("KH")
        for seed in range(25):
            d = game.deal_around(known_hand=known, seat=1, up_card=up,
                                 rng=random.Random(seed))
            for card in known:
                self.assertIn(card, d.hands[1])
            self.assertEqual(d.up_card, up)
            d.check()

    def test_pinned_cards_appear_exactly_once(self):
        known = r.parse_hand("JS JC AS")
        for seed in range(25):
            d = game.deal_around(known_hand=known, seat=2, rng=random.Random(seed))
            cards = d.all_cards()
            for card in known:
                self.assertEqual(cards.count(card), 1)

    def test_a_full_five_card_hand_is_dealt_exactly(self):
        known = r.parse_hand("JH JD AH KH QH")
        d = game.deal_around(known_hand=known, seat=3, rng=random.Random(1))
        self.assertEqual(set(d.hands[3]), set(known))

    def test_works_with_nothing_pinned(self):
        game.deal_around(rng=random.Random(0)).check()

    def test_up_card_only(self):
        up = r.parse_card("9S")
        d = game.deal_around(up_card=up, rng=random.Random(4))
        self.assertEqual(d.up_card, up)
        for hand in d.hands:
            self.assertNotIn(up, hand)

    def test_rejects_a_card_pinned_twice(self):
        with self.assertRaises(ValueError):
            game.deal_around(known_hand=r.parse_hand("JH JH"), seat=0)

    def test_rejects_a_hand_pinned_to_the_up_card(self):
        with self.assertRaises(ValueError):
            game.deal_around(known_hand=r.parse_hand("JH AH"), seat=0,
                             up_card=r.parse_card("JH"))

    def test_rejects_an_oversized_hand(self):
        with self.assertRaises(ValueError):
            game.deal_around(known_hand=r.parse_hand("JH JD AH KH QH TH"), seat=0)

    def test_rejects_a_bad_seat(self):
        with self.assertRaises(ValueError):
            game.deal_around(known_hand=r.parse_hand("JH"), seat=7)


class TestDealFromOrder(unittest.TestCase):
    def test_five_to_each_seat_in_turn_then_the_kitty(self):
        cards = r.full_deck()
        d = game.deal_from_order(cards)
        self.assertEqual(d.hands[0], tuple(cards[0:5]))
        self.assertEqual(d.hands[3], tuple(cards[15:20]))
        self.assertEqual(d.up_card, cards[20])
        self.assertEqual(d.buried, tuple(cards[21:24]))

    def test_is_deterministic(self):
        cards = r.full_deck()
        self.assertEqual(game.deal_from_order(cards), game.deal_from_order(cards))

    def test_rejects_a_short_deck(self):
        with self.assertRaises(ValueError):
            game.deal_from_order(r.full_deck()[:23])

    def test_rejects_a_deck_with_duplicates(self):
        cards = r.full_deck()
        cards[5] = cards[0]
        with self.assertRaises(ValueError):
            game.deal_from_order(cards)


class TestFeedsTheSolver(unittest.TestCase):
    """The point of the deal is to become a solvable position."""

    def test_a_dealt_hand_solves_under_any_call(self):
        for seed in range(10):
            d = game.deal_random(rng=random.Random(seed))
            for trump in r.SUITS:
                arr = r.deal_to_engine(d.hands, trump)
                score = definitive_winner(arr, d.first_bidder, d.first_bidder)
                self.assertIn(score, (-2, 1, 2))

    def test_solving_after_the_pickup_uses_the_new_dealer_hand(self):
        d = game.deal_random(rng=random.Random(3), dealer=0)
        after = d.pick_up(discard=d.hands[0][0])
        trump = d.up_card.suit
        arr = r.deal_to_engine(after.hands, trump)
        self.assertEqual(arr.shape, (4, 5, 2))
        self.assertIn(definitive_winner(arr, 1, 1), (-2, 1, 2))

    def test_ordering_up_puts_the_up_card_suit_in_the_dealers_hand(self):
        """The up-card's suit is what gets called when it is ordered up."""
        d = game.deal_random(rng=random.Random(7), dealer=2)
        after = d.pick_up(discard=d.hands[2][0])
        engine = r.deal_to_engine(after.hands, d.up_card.suit)
        up_vector = r.card_to_engine(d.up_card, d.up_card.suit)
        self.assertIn(up_vector, engine[2].tolist())


class TestCheck(unittest.TestCase):
    def test_catches_a_missing_card(self):
        d = a_deal()
        broken = game.Deal(hands=d.hands, up_card=d.up_card,
                           buried=d.buried[:2], dealer=d.dealer)
        with self.assertRaises(ValueError):
            broken.check()

    def test_catches_a_duplicate(self):
        d = a_deal()
        hands = list(d.hands)
        hands[1] = hands[0]
        with self.assertRaises(ValueError):
            game.Deal(hands=tuple(hands), up_card=d.up_card,
                      buried=d.buried, dealer=d.dealer).check()

    def test_catches_a_short_hand(self):
        d = a_deal()
        hands = list(d.hands)
        hands[2] = hands[2][:4]
        with self.assertRaises(ValueError):
            game.Deal(hands=tuple(hands), up_card=d.up_card,
                      buried=d.buried, dealer=d.dealer).check()


if __name__ == "__main__":
    unittest.main()
