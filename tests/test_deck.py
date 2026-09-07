"""
Unit tests for deck.py and the card encoding it defines.

The encoding is the load-bearing part of the whole engine: suit is direction,
strength is magnitude, spades is always trump, and the left bower sits inside
the trump axis. Everything downstream reads cards positionally, so these tests
pin the representation itself rather than any behaviour built on it.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

import numpy as np

import euchre_testkit as k
from deck import full_euchre_deck
from fast_search import encode_hands, _decode


class TestDeckContents(unittest.TestCase):
    def test_shape_and_dtype(self):
        self.assertEqual(full_euchre_deck.shape, (24, 2))
        self.assertTrue(np.issubdtype(full_euchre_deck.dtype, np.integer))

    def test_all_cards_distinct(self):
        self.assertEqual(len(np.unique(full_euchre_deck, axis=0)), 24)

    def test_matches_the_named_deck(self):
        """deck.py and the test kit's card table must describe the same 24 cards."""
        from_deck = {(int(c[0]), int(c[1])) for c in full_euchre_deck}
        self.assertEqual(from_deck, set(k.CARD.values()))

    def test_suit_counts(self):
        counts = {}
        for c in full_euchre_deck:
            s = k.suit_of(c)
            counts[s] = counts.get(s, 0) + 1
        # 6 hearts, 6 diamonds, 5 clubs (the jack is the left bower),
        # 7 on the trump axis (5 spades + both bowers)
        self.assertEqual(counts[k.HEARTS], 6)
        self.assertEqual(counts[k.DIAMONDS], 6)
        self.assertEqual(counts[k.CLUBS], 5)
        self.assertEqual(counts[k.TRUMP], 7)

    def test_clubs_has_no_jack_of_its_own(self):
        """The jack of clubs is the left bower and lives on the trump axis."""
        clubs = [c for c in full_euchre_deck if k.suit_of(c) == k.CLUBS]
        self.assertNotIn(11, [k.suit_strength(c)[1] for c in clubs])
        self.assertIn((0, 135), {(int(c[0]), int(c[1])) for c in full_euchre_deck})


class TestCardEncoding(unittest.TestCase):
    def test_trump_and_plain_strengths_never_overlap(self):
        """No cross-suit comparison can go wrong if the ranges are disjoint."""
        trump, plain = [], []
        for c in full_euchre_deck:
            s, v = k.suit_strength(c)
            (trump if s == k.TRUMP else plain).append(v)
        self.assertGreater(min(trump), max(plain))
        self.assertGreater(min(trump), 80)   # the documented norm > 80 trump test
        self.assertLess(max(plain), 80)

    def test_trump_order(self):
        """Right bower > left bower > ace down to nine."""
        order = ["JS", "JC", "AS", "KS", "QS", "TS", "9S"]
        strengths = [k.suit_strength(k.CARD[n])[1] for n in order]
        self.assertEqual(strengths, sorted(strengths, reverse=True))
        self.assertEqual(strengths[0], 140)
        self.assertEqual(strengths[1], 135)

    def test_plain_suit_order(self):
        for suit in ("H", "D"):
            order = ["A", "K", "Q", "J", "T", "9"]
            strengths = [k.suit_strength(k.CARD[r + suit])[1] for r in order]
            self.assertEqual(strengths, sorted(strengths, reverse=True))

    def test_encode_hands_agrees_with_the_rules(self):
        """fast_search's njit encoder must match an independent reading."""
        cards = full_euchre_deck.reshape(4, 6, 2).astype(np.int64)
        suits, strs = encode_hands(np.ascontiguousarray(cards))
        for p in range(4):
            for c in range(6):
                want_s, want_v = k.suit_strength(cards[p, c])
                self.assertEqual((int(suits[p, c]), int(strs[p, c])), (want_s, want_v),
                                 "card %s encoded wrong" % k.name_of(cards[p, c]))

    def test_left_bower_encodes_as_trump(self):
        """The single most important consequence of the encoding."""
        cards = np.array([[k.CARD["JC"], k.CARD["AS"], k.CARD["JS"],
                           k.CARD["AC"], k.CARD["9C"]]], dtype=np.int64)
        suits, strs = encode_hands(cards)
        self.assertEqual(int(suits[0, 0]), k.TRUMP, "left bower is not trump")
        self.assertEqual(int(suits[0, 3]), k.CLUBS)
        # and it ranks between the ace of spades and the right bower
        self.assertLess(int(strs[0, 1]), int(strs[0, 0]))
        self.assertLess(int(strs[0, 0]), int(strs[0, 2]))

    def test_decode_round_trips_every_card(self):
        for c in full_euchre_deck:
            s, v = k.suit_strength(c)
            self.assertEqual(_decode(s, v), [int(c[0]), int(c[1])],
                             "round trip failed for %s" % k.name_of(c))


if __name__ == "__main__":
    unittest.main()
