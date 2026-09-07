"""
Unit tests for dealer.py.

Includes regressions for three bugs the dealer has actually had:

  * `np.isin` matched card *coordinates* rather than whole cards, so stacking
    9d and Ac silently deleted Ah from the deck as well;
  * stacking twice for one player replaced that player's hand while leaving the
    earlier cards removed from the deck, so those cards could not be dealt to
    anyone (generate_hands hit this whenever `stack` and `up_card` named the
    same player);
  * an oversized stack surfaced as `ValueError: Negative dimensions are not
    allowed` from inside np.random.choice, several frames from the cause.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

import numpy as np

import euchre_testkit as k
from dealer import Dealer, HAND_SIZE
from deck import full_euchre_deck


def fresh():
    return Dealer(deck=full_euchre_deck, players=4)


class TestConstruction(unittest.TestCase):
    def test_creates_one_empty_hand_per_player(self):
        d = fresh()
        self.assertEqual(list(d.hands), ["hand0", "hand1", "hand2", "hand3"])
        self.assertTrue(all(len(h) == 0 for h in d.hands.values()))

    def test_honours_the_players_field(self):
        """`players` used to be ignored -- four hands were built regardless."""
        self.assertEqual(list(Dealer(deck=full_euchre_deck, players=2).hands),
                         ["hand0", "hand1"])

    def test_rejects_a_table_the_deck_cannot_seat(self):
        with self.assertRaises(ValueError):
            Dealer(deck=full_euchre_deck, players=5)   # 25 cards needed, 24 available
        with self.assertRaises(ValueError):
            Dealer(deck=full_euchre_deck, players=0)


class TestDealing(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_deals_five_distinct_cards_to_each_player(self):
        d = fresh()
        d.deal_cards()
        for i in range(4):
            self.assertEqual(len(d.hands[f"hand{i}"]), HAND_SIZE)
        flat = np.vstack([d.hands[f"hand{i}"] for i in range(4)])
        self.assertEqual(len(np.unique(flat, axis=0)), 20)

    def test_sets_zero_based_hand_attributes(self):
        d = fresh()
        d.deal_cards()
        for i in range(4):
            self.assertTrue(hasattr(d, f"hand{i}"), "missing attribute hand%d" % i)
            np.testing.assert_array_equal(getattr(d, f"hand{i}"), d.hands[f"hand{i}"])

    def test_every_dealt_card_comes_from_the_deck(self):
        d = fresh()
        d.deal_cards()
        flat = np.vstack([d.hands[f"hand{i}"] for i in range(4)])
        known = {(int(c[0]), int(c[1])) for c in full_euchre_deck}
        for c in flat:
            self.assertIn((int(c[0]), int(c[1])), known)

    def test_does_not_mutate_the_shared_deck(self):
        before = full_euchre_deck.copy()
        d = fresh()
        d.stack_deck(k.hand("JS", "JC"), 0)
        d.deal_cards()
        np.testing.assert_array_equal(full_euchre_deck, before)

    def test_repeated_deals_differ(self):
        """A fixed deal would silently invalidate every simulation built on it."""
        seen = set()
        for _ in range(20):
            d = fresh()
            d.deal_cards()
            seen.add(tuple(map(tuple, d.hands["hand0"].tolist())))
        self.assertGreater(len(seen), 1)


class TestStacking(unittest.TestCase):
    def setUp(self):
        np.random.seed(99)

    def test_stacked_cards_land_in_the_named_hand(self):
        d = fresh()
        cards = k.hand("JS", "JC", "AS")
        d.stack_deck(cards, 2)
        d.deal_cards()
        held = {k.name_of(c) for c in d.hands["hand2"]}
        self.assertTrue({"JS", "JC", "AS"} <= held)
        self.assertEqual(len(d.hands["hand2"]), HAND_SIZE)

    def test_removes_exactly_the_stacked_cards(self):
        d = fresh()
        d.stack_deck(k.hand("JS", "JC", "AS"), 0)
        self.assertEqual(len(d.deck), 24 - 3)

    def test_coordinate_match_regression(self):
        """
        Stacking 9d [9, 0] and Ac [0, -14] must not also remove Ah [-14, 0].

        np.isin compared each coordinate against every value in the stack, so
        any card built from numbers appearing anywhere in the stack vanished.
        """
        d = fresh()
        d.stack_deck(k.hand("9D", "AC"), 0)
        self.assertEqual(len(d.deck), 22)
        remaining = {k.name_of(c) for c in d.deck}
        self.assertIn("AH", remaining)
        self.assertNotIn("9D", remaining)
        self.assertNotIn("AC", remaining)

    def test_stacking_twice_accumulates(self):
        """
        Regression: the second call used to overwrite the first player's hand
        while the first call's cards stayed removed from the deck, so they were
        dealt to nobody.
        """
        d = fresh()
        d.stack_deck(k.hand("JS", "JC", "AS"), 0)
        d.stack_deck(k.hand("KS"), 0)
        self.assertEqual(len(d.hands["hand0"]), 4)
        self.assertEqual({k.name_of(c) for c in d.hands["hand0"]},
                         {"JS", "JC", "AS", "KS"})
        d.deal_cards()
        flat = np.vstack([d.hands[f"hand{i}"] for i in range(4)])
        self.assertEqual(len(np.unique(flat, axis=0)), 20,
                         "a stacked card went missing from the deal")

    def test_accepts_a_bare_single_card(self):
        """generate_hands' up_card is naturally a single (2,) vector."""
        d = fresh()
        d.stack_deck(np.array(k.CARD["JS"]), 3)
        self.assertEqual(k.name_of(d.hands["hand3"][0]), "JS")
        self.assertEqual(len(d.deck), 23)

    def test_rejects_a_card_not_in_the_deck(self):
        d = fresh()
        with self.assertRaises(ValueError):
            d.stack_deck(np.array([[0, 999]]), 0)

    def test_rejects_a_card_already_stacked(self):
        d = fresh()
        d.stack_deck(k.hand("JS"), 0)
        with self.assertRaises(ValueError):
            d.stack_deck(k.hand("JS"), 1)

    def test_rejects_duplicates_within_one_stack(self):
        d = fresh()
        with self.assertRaises(ValueError):
            d.stack_deck(k.hand("JS", "JS"), 0)

    def test_rejects_an_oversized_stack(self):
        """Used to fail as 'Negative dimensions are not allowed' inside deal_cards."""
        d = fresh()
        with self.assertRaises(ValueError) as cm:
            d.stack_deck(full_euchre_deck[:6], 0)
        self.assertIn("exceed", str(cm.exception))

    def test_rejects_an_overfilling_second_stack(self):
        d = fresh()
        d.stack_deck(k.hand("JS", "JC", "AS", "KS"), 0)
        with self.assertRaises(ValueError):
            d.stack_deck(k.hand("QS", "TS"), 0)

    def test_rejects_a_seat_that_does_not_exist(self):
        d = fresh()
        for bad in (-1, 4, 9):
            with self.assertRaises(ValueError):
                d.stack_deck(k.hand("JS"), bad)

    def test_rejects_a_badly_shaped_stack(self):
        d = fresh()
        with self.assertRaises(ValueError):
            d.stack_deck(np.array([[0, 140, 7]]), 0)

    def test_a_full_stack_is_dealt_untouched(self):
        d = fresh()
        cards = k.hand("JS", "JC", "AS", "KS", "QS")
        d.stack_deck(cards, 1)
        d.deal_cards()
        np.testing.assert_array_equal(d.hands["hand1"], cards)
        np.testing.assert_array_equal(d.hand1, cards)


if __name__ == "__main__":
    unittest.main()
