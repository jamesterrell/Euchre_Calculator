"""
Unit tests for reference_solver.py.

reference_solver is the independent oracle that test_fast_search.py and
test_solver.py check the real solver against, so it needs testing in its own
right -- an oracle that agrees with a bug is worse than no oracle. It is itself
an alpha-beta search, so the checks here go one level further down and compare
it with euchre_testkit.full_minimax, which prunes nothing at all.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import euchre_testkit as k
from reference_solver import (
    card_suit_strength,
    hands_to_py,
    solve_py,
    trick_winner,
)

D, T, H, C = k.DIAMONDS, k.TRUMP, k.HEARTS, k.CLUBS


class TestCardSuitStrength(unittest.TestCase):
    def test_every_card_in_the_deck(self):
        for name, (x, y) in k.CARD.items():
            self.assertEqual(card_suit_strength(x, y), k.suit_strength((x, y)),
                             "disagreed on %s" % name)

    def test_the_four_suits(self):
        self.assertEqual(card_suit_strength(*k.CARD["AD"]), (D, 14))
        self.assertEqual(card_suit_strength(*k.CARD["AH"]), (H, 14))
        self.assertEqual(card_suit_strength(*k.CARD["AC"]), (C, 14))
        self.assertEqual(card_suit_strength(*k.CARD["AS"]), (T, 130))

    def test_both_bowers_are_trump(self):
        self.assertEqual(card_suit_strength(*k.CARD["JS"]), (T, 140))
        self.assertEqual(card_suit_strength(*k.CARD["JC"]), (T, 135))


class TestTrickWinner(unittest.TestCase):
    def test_highest_of_the_led_suit_wins(self):
        played = [(H, 9, 0), (H, 14, 1), (H, 12, 2), (H, 10, 3)]
        self.assertEqual(trick_winner(played), 1)

    def test_off_suit_cannot_win(self):
        """An ace of another plain suit loses to the nine that was led."""
        played = [(H, 9, 0), (D, 14, 1), (C, 14, 2), (D, 13, 3)]
        self.assertEqual(trick_winner(played), 0)

    def test_any_trump_beats_the_led_suit(self):
        played = [(H, 14, 0), (T, 90, 1), (H, 13, 2), (H, 12, 3)]
        self.assertEqual(trick_winner(played), 1)

    def test_highest_trump_wins_when_several_are_played(self):
        played = [(T, 90, 0), (T, 140, 1), (T, 135, 2), (T, 130, 3)]
        self.assertEqual(trick_winner(played), 1)

    def test_left_bower_beats_the_ace_of_trump(self):
        played = [(T, 130, 0), (T, 135, 1), (H, 14, 2), (D, 14, 3)]
        self.assertEqual(trick_winner(played), 1)

    def test_right_bower_beats_the_left(self):
        played = [(T, 135, 0), (T, 140, 1), (T, 130, 2), (T, 120, 3)]
        self.assertEqual(trick_winner(played), 1)

    def test_the_leader_can_win(self):
        played = [(T, 140, 2), (T, 135, 3), (T, 130, 0), (T, 120, 1)]
        self.assertEqual(trick_winner(played), 2)

    def test_agrees_with_the_testkit_oracle(self):
        rng = random.Random(404)
        cards = [k.suit_strength(v) for v in k.CARD.values()]
        for _ in range(500):
            four = rng.sample(cards, 4)
            played = [(s, v, i) for i, (s, v) in enumerate(four)]
            self.assertEqual(trick_winner(played), k.winner_of(played))


class TestHandsToPy(unittest.TestCase):
    def test_converts_a_whole_deal(self):
        deal = k.deal(
            ["JS", "JC", "9C", "9H", "9D"],
            ["KD", "AC", "TS", "QS", "TC"],
            ["9S", "TD", "AD", "QH", "JD"],
            ["JH", "AH", "QD", "KC", "TH"],
        )
        got = hands_to_py(deal)
        self.assertEqual(len(got), 4)
        self.assertTrue(all(len(h) == 5 for h in got))
        self.assertEqual(got[0][0], (T, 140))
        self.assertEqual(got[0][1], (T, 135))
        self.assertEqual(got[0][2], (C, 9))


class TestSolvePy(unittest.TestCase):
    def test_march(self):
        d = k.deal(
            ["JS", "JC", "AS", "KS", "QS"],
            ["AH", "KH", "QH", "JH", "TH"],
            ["AD", "KD", "QD", "JD", "TD"],
            ["AC", "KC", "QC", "TC", "9C"],
        )
        self.assertEqual(solve_py(hands_to_py(d), 0, 0), 2)
        self.assertEqual(solve_py(hands_to_py(d), 0, 1), -2)

    def test_canonical_fixture(self):
        d = k.deal(
            ["JS", "JC", "9C", "9H", "9D"],
            ["KD", "AC", "TS", "QS", "TC"],
            ["9S", "TD", "AD", "QH", "JD"],
            ["JH", "AH", "QD", "KC", "TH"],
        )
        self.assertEqual(solve_py(hands_to_py(d), 2, 0), 2)

    def test_does_not_consume_the_hands_it_is_given(self):
        """It removes and re-appends cards in place, so check it puts them back."""
        d = k.deal(
            ["JS", "JC", "9C", "9H", "9D"],
            ["KD", "AC", "TS", "QS", "TC"],
            ["9S", "TD", "AD", "QH", "JD"],
            ["JH", "AH", "QD", "KC", "TH"],
        )
        py = hands_to_py(d)
        before = [sorted(h) for h in py]
        solve_py(py, 2, 0)
        self.assertEqual([sorted(h) for h in py], before)

    def test_matches_exhaustive_minimax(self):
        """
        The oracle's own alpha-beta must return the same value as a search with
        no pruning whatsoever, on each of the pinned cheap fixtures.
        """
        for label, hands, sp, caller, expected in k.brute_forceable_deals():
            want, _ = k.full_minimax(hands, sp, caller)
            self.assertEqual(want, expected,
                             "%s: the fixture's pinned score is stale" % label)
            self.assertEqual(solve_py(hands_to_py(hands), sp, caller), want,
                             "%s: reference_solver disagreed with brute force" % label)


if __name__ == "__main__":
    unittest.main()
