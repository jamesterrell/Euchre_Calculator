"""
Unit tests for n_game_sim.generate_hands.

Every sweep in the notebook is built on this, so a silent dealing fault here
biases every EV number downstream without anything failing loudly. The tests
therefore check the deal itself -- 20 distinct cards, all from the deck, five
per player -- on every generated game rather than sampling one.

Includes the regression for `stack` and `up_card` naming the same player, which
used to drop the stacked cards out of the game entirely: they were removed from
the deck by the first call and then thrown away by the second.
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
from n_game_sim import generate_hands


def holds(game, player, name):
    want = k.CARD[name]
    return any((int(c[0]), int(c[1])) == want for c in game[player])


class TestShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2024)

    def test_shape_and_dtype(self):
        out = generate_hands(n_games=7)
        self.assertEqual(out.shape, (7, 4, 5, 2))
        self.assertEqual(out.dtype, np.int64)

    def test_default_size(self):
        self.assertEqual(generate_hands().shape[0], 100)

    def test_zero_games(self):
        self.assertEqual(generate_hands(n_games=0).shape, (0, 4, 5, 2))


class TestDealValidity(unittest.TestCase):
    def setUp(self):
        np.random.seed(31337)

    def test_every_game_is_a_legal_deal(self):
        for i, game in enumerate(generate_hands(n_games=250)):
            errs = k.deal_violations(game, full_euchre_deck)
            self.assertEqual(errs, [], "game %d: %s" % (i, errs))

    def test_games_are_not_all_identical(self):
        games = generate_hands(n_games=25)
        distinct = {game.tobytes() for game in games}
        self.assertGreater(len(distinct), 1)

    def test_four_cards_are_left_undealt(self):
        """20 of the 24 cards are dealt; the kitty is simply never returned."""
        for game in generate_hands(n_games=20):
            self.assertEqual(len(np.unique(game.reshape(-1, 2), axis=0)), 20)


class TestStacking(unittest.TestCase):
    def setUp(self):
        np.random.seed(808)

    def test_stack_is_honoured_and_the_deal_stays_legal(self):
        stack = k.hand("JS", "JC", "AS")
        for game in generate_hands(n_games=40, stack=stack, stack_player=1):
            self.assertEqual(k.deal_violations(game, full_euchre_deck), [])
            for name in ("JS", "JC", "AS"):
                self.assertTrue(holds(game, 1, name), "player 1 lost %s" % name)

    def test_a_full_five_card_stack_is_dealt_exactly(self):
        stack = k.hand("JS", "JC", "AS", "KS", "QS")
        for game in generate_hands(n_games=10, stack=stack, stack_player=0):
            np.testing.assert_array_equal(game[0], stack)
            self.assertEqual(k.deal_violations(game, full_euchre_deck), [])

    def test_up_card_is_honoured(self):
        up = k.hand("JS")
        for game in generate_hands(n_games=25, up_card=up, up_card_player=2):
            self.assertTrue(holds(game, 2, "JS"))
            self.assertEqual(k.deal_violations(game, full_euchre_deck), [])

    def test_up_card_may_be_a_bare_vector(self):
        for game in generate_hands(n_games=10,
                                   up_card=np.array(k.CARD["JS"]),
                                   up_card_player=3):
            self.assertTrue(holds(game, 3, "JS"))

    def test_stack_and_up_card_for_different_players(self):
        stack = k.hand("JS", "JC", "AS")
        for game in generate_hands(n_games=25, stack=stack, stack_player=0,
                                   up_card=k.hand("KS"), up_card_player=2):
            self.assertEqual(k.deal_violations(game, full_euchre_deck), [])
            for name in ("JS", "JC", "AS"):
                self.assertTrue(holds(game, 0, name))
            self.assertTrue(holds(game, 2, "KS"))

    def test_stack_and_up_card_for_the_same_player(self):
        """
        Regression: the up-card call replaced the stacked hand instead of adding
        to it, while the stacked cards stayed removed from the deck -- so they
        were dealt to nobody and the game ran with 17 live cards.
        """
        stack = k.hand("JS", "JC", "AS")
        for game in generate_hands(n_games=25, stack=stack, stack_player=0,
                                   up_card=k.hand("KS"), up_card_player=0):
            self.assertEqual(k.deal_violations(game, full_euchre_deck), [])
            for name in ("JS", "JC", "AS", "KS"):
                self.assertTrue(holds(game, 0, name),
                                "player 0 lost %s" % name)

    def test_stacked_cards_never_appear_twice(self):
        stack = k.hand("JS", "JC")
        for game in generate_hands(n_games=40, stack=stack, stack_player=3):
            flat = [(int(c[0]), int(c[1])) for c in game.reshape(-1, 2)]
            for name in ("JS", "JC"):
                self.assertEqual(flat.count(k.CARD[name]), 1)

    def test_rejects_a_stack_that_overfills_a_hand(self):
        with self.assertRaises(ValueError):
            generate_hands(n_games=1, stack=full_euchre_deck[:6], stack_player=0)

    def test_rejects_a_conflicting_stack_and_up_card(self):
        with self.assertRaises(ValueError):
            generate_hands(n_games=1, stack=k.hand("JS", "JC"), stack_player=0,
                           up_card=k.hand("JS"), up_card_player=1)


if __name__ == "__main__":
    unittest.main()
