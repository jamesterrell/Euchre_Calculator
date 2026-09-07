"""
Unit tests for rotation.py.

The property that matters is that rotation is a *bijection onto the canonical
deck* for every trump suit. If it is, then a rotated hand is indistinguishable
from a natively-canonical one and the solver's existing guarantees carry over
unchanged. If it is not -- if two real cards collide, or a card lands outside
the deck -- the solver will happily solve the wrong position without complaint,
because it has no way to know it was handed nonsense.

The left bower is where this goes wrong if it goes wrong: it is the jack of the
suit the same colour as trump, so which jack leaves its own suit changes with
the call.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import numpy as np

import rotation as r
from deck import full_euchre_deck
from fast_search import definitive_winner

CANONICAL = {(int(c[0]), int(c[1])) for c in full_euchre_deck}


def relabel_from_spades(card, trump):
    """
    The colour-preserving permutation carrying a spades-trump deal to `trump`:
    trump takes spades' place, its same-colour partner takes clubs' place, and
    the two off-colour suits follow in the same order.
    """
    plus_s, minus_s = r._plain_axes(r.SPADES)
    plus_t, minus_t = r._plain_axes(trump)
    mapping = {r.SPADES: trump,
               r.CLUBS: r.same_colour(trump),
               plus_s: plus_t,
               minus_s: minus_t}
    return r.Card(mapping[card.suit], card.rank)


class TestDeckAndColours(unittest.TestCase):
    def test_full_deck_is_24_distinct_real_cards(self):
        deck = r.full_deck()
        self.assertEqual(len(deck), 24)
        self.assertEqual(len(set(deck)), 24)

    def test_same_colour_pairs(self):
        self.assertEqual(r.same_colour(r.SPADES), r.CLUBS)
        self.assertEqual(r.same_colour(r.CLUBS), r.SPADES)
        self.assertEqual(r.same_colour(r.HEARTS), r.DIAMONDS)
        self.assertEqual(r.same_colour(r.DIAMONDS), r.HEARTS)

    def test_same_colour_is_an_involution(self):
        for suit in r.SUITS:
            self.assertEqual(r.same_colour(r.same_colour(suit)), suit)
            self.assertNotEqual(r.same_colour(suit), suit)


class TestBijection(unittest.TestCase):
    """The whole safety argument for rotation rests on this."""

    def test_every_trump_suit_maps_onto_the_canonical_deck(self):
        for trump in r.SUITS:
            vectors = [tuple(r.card_to_engine(c, trump)) for c in r.full_deck()]
            self.assertEqual(len(set(vectors)), 24,
                             "trump %s: two real cards collided" % r.suit_name(trump))
            self.assertEqual(set(vectors), CANONICAL,
                             "trump %s: did not reproduce full_euchre_deck"
                             % r.suit_name(trump))

    def test_round_trips_every_card_under_every_trump(self):
        for trump in r.SUITS:
            for card in r.full_deck():
                got = r.card_from_engine(r.card_to_engine(card, trump), trump)
                self.assertEqual(got, card,
                                 "%s did not survive a round trip under %s"
                                 % (r.card_name(card), r.suit_name(trump)))

    def test_spades_trump_is_the_identity(self):
        """
        deck.py is already written for spades, so rotating into spades must
        change nothing. Keeps the canonical constants and this module honest
        about each other.
        """
        for name, want in (("JS", [0, 140]), ("JC", [0, 135]), ("AS", [0, 130]),
                           ("9S", [0, 90]), ("QS", [0, 110]),
                           ("AC", [0, -14]), ("9C", [0, -9]),
                           ("AH", [-14, 0]), ("JH", [-11, 0]),
                           ("AD", [14, 0]), ("JD", [11, 0])):
            self.assertEqual(r.card_to_engine(r.parse_card(name), r.SPADES), want,
                             "%s misplaced under spades" % name)


class TestBowers(unittest.TestCase):
    def test_right_bower_is_the_jack_of_trump(self):
        for trump in r.SUITS:
            self.assertEqual(r.card_to_engine(r.Card(trump, r.JACK), trump),
                             [0, r.RIGHT_BOWER])

    def test_left_bower_is_the_jack_of_the_same_colour(self):
        for trump in r.SUITS:
            left = r.Card(r.same_colour(trump), r.JACK)
            self.assertEqual(r.card_to_engine(left, trump), [0, r.LEFT_BOWER],
                             "%s should be the left bower under %s"
                             % (r.card_name(left), r.suit_name(trump)))

    def test_the_named_bowers_are_the_ones_euchre_expects(self):
        expected = {r.SPADES: ("JS", "JC"), r.CLUBS: ("JC", "JS"),
                    r.HEARTS: ("JH", "JD"), r.DIAMONDS: ("JD", "JH")}
        for trump, (right, left) in expected.items():
            self.assertEqual(r.card_to_engine(r.parse_card(right), trump),
                             [0, r.RIGHT_BOWER])
            self.assertEqual(r.card_to_engine(r.parse_card(left), trump),
                             [0, r.LEFT_BOWER])

    def test_the_same_colour_suit_loses_its_jack(self):
        """
        Its jack became the left bower, so the suit is left with five cards --
        which is exactly why the canonical clubs axis has five.
        """
        for trump in r.SUITS:
            left_suit = r.same_colour(trump)
            on_axis = [r.card_to_engine(r.Card(left_suit, rank), trump)
                       for rank in r.RANKS if rank != r.JACK]
            self.assertEqual(len(on_axis), 5)
            for vec in on_axis:
                self.assertEqual(vec[0], 0)
                self.assertLess(vec[1], 0, "should sit on the clubs axis")

    def test_off_colour_suits_keep_all_six_ranks(self):
        for trump in r.SUITS:
            for suit in r._plain_axes(trump):
                vectors = [r.card_to_engine(r.Card(suit, rank), trump)
                           for rank in r.RANKS]
                self.assertEqual(len(set(map(tuple, vectors))), 6)
                for vec in vectors:
                    self.assertEqual(vec[1], 0, "should sit on a plain-suit axis")

    def test_bowers_outrank_the_ace_of_trump(self):
        for trump in r.SUITS:
            ace = r.card_to_engine(r.Card(trump, r.ACE), trump)[1]
            left = r.card_to_engine(r.Card(r.same_colour(trump), r.JACK), trump)[1]
            right = r.card_to_engine(r.Card(trump, r.JACK), trump)[1]
            self.assertLess(ace, left)
            self.assertLess(left, right)


class TestSolvingThroughRotation(unittest.TestCase):
    def test_a_called_suit_solves_the_same_as_spades(self):
        """
        The end-to-end claim. Relabel a deal by the colour-preserving
        permutation that moves trump from spades to each other suit; the
        canonical form, and so the score, must not budge.
        """
        rng = random.Random(0)
        deck = r.full_deck()
        for _ in range(40):
            rng.shuffle(deck)
            natural = [deck[i * 5:(i + 1) * 5] for i in range(4)]
            sp, caller = rng.randrange(4), rng.randrange(4)
            base = r.deal_to_engine(natural, r.SPADES)
            base_score = definitive_winner(base, sp, caller)

            for trump in r.SUITS:
                moved = [[relabel_from_spades(c, trump) for c in hand]
                         for hand in natural]
                arr = r.deal_to_engine(moved, trump)
                np.testing.assert_array_equal(
                    arr, base,
                    "trump %s did not reproduce the canonical deal"
                    % r.suit_name(trump))
                self.assertEqual(definitive_winner(arr, sp, caller), base_score)

    def test_a_worked_hearts_hand(self):
        """Both bowers plus the ace, hearts called: the top three cards."""
        hands = [r.parse_hand("JH JD AH KH QH"),
                 r.parse_hand("JS JC AS KS QS"),
                 r.parse_hand("AD KD QD TD 9D"),
                 r.parse_hand("AC KC QC TC 9C")]
        arr = r.deal_to_engine(hands, r.HEARTS)
        self.assertEqual(definitive_winner(arr, 0, 0), 2, "should be a march")
        self.assertEqual(arr[0].tolist(),
                         [[0, 140], [0, 135], [0, 130], [0, 120], [0, 110]])

    def test_deal_to_engine_output_is_solver_ready(self):
        rng = random.Random(11)
        deck = r.full_deck()
        rng.shuffle(deck)
        arr = r.deal_to_engine([deck[i * 5:(i + 1) * 5] for i in range(4)],
                               r.DIAMONDS)
        self.assertEqual(arr.shape, (4, 5, 2))
        self.assertEqual(arr.dtype, np.int64)
        self.assertIn(definitive_winner(arr, 0, 0), (-2, 1, 2))


class TestDealValidation(unittest.TestCase):
    def setUp(self):
        deck = r.full_deck()
        self.hands = [deck[i * 5:(i + 1) * 5] for i in range(4)]

    def test_accepts_a_good_deal(self):
        self.assertEqual(r.deal_to_engine(self.hands, r.SPADES).shape, (4, 5, 2))

    def test_rejects_a_duplicate_card(self):
        bad = [list(h) for h in self.hands]
        bad[1][0] = bad[0][0]
        with self.assertRaises(ValueError):
            r.deal_to_engine(bad, r.SPADES)

    def test_rejects_the_wrong_number_of_hands(self):
        with self.assertRaises(ValueError):
            r.deal_to_engine(self.hands[:3], r.SPADES)

    def test_rejects_a_short_hand(self):
        bad = [list(h) for h in self.hands]
        bad[2] = bad[2][:4]
        with self.assertRaises(ValueError):
            r.deal_to_engine(bad, r.SPADES)


class TestBadInput(unittest.TestCase):
    def test_rejects_a_bad_suit(self):
        with self.assertRaises(ValueError):
            r.card_to_engine(r.Card(9, r.ACE), r.SPADES)

    def test_rejects_a_bad_rank(self):
        with self.assertRaises(ValueError):
            r.card_to_engine(r.Card(r.SPADES, 2), r.SPADES)

    def test_rejects_a_bad_trump(self):
        with self.assertRaises(ValueError):
            r.card_to_engine(r.Card(r.SPADES, r.ACE), 7)

    def test_rejects_vectors_that_are_not_cards(self):
        for bad in ([0, 0], [0, 77], [3, 0]):
            with self.assertRaises(ValueError):
                r.card_from_engine(bad, r.SPADES)


class TestNotation(unittest.TestCase):
    def test_parses_every_card(self):
        for card in r.full_deck():
            self.assertEqual(r.parse_card(r.card_name(card)), card)

    def test_accepts_ten_written_either_way(self):
        self.assertEqual(r.parse_card("10H"), r.parse_card("TH"))
        self.assertEqual(r.parse_card("10h"), r.Card(r.HEARTS, r.TEN))

    def test_is_case_insensitive(self):
        self.assertEqual(r.parse_card("jh"), r.Card(r.HEARTS, r.JACK))

    def test_parses_a_hand_from_a_string_or_a_list(self):
        want = [r.Card(r.HEARTS, r.JACK), r.Card(r.DIAMONDS, r.JACK)]
        self.assertEqual(r.parse_hand("JH JD"), want)
        self.assertEqual(r.parse_hand(["JH", "JD"]), want)

    def test_hand_name_round_trips(self):
        text = "JH JD AH 9S TC"
        self.assertEqual(r.hand_name(r.parse_hand(text)), text)

    def test_rejects_nonsense(self):
        for bad in ("", "H", "1H", "JX", "ZZ"):
            with self.assertRaises(ValueError):
                r.parse_card(bad)

    def test_suit_notation(self):
        self.assertEqual(r.parse_suit("H"), r.HEARTS)
        self.assertEqual(r.parse_suit("hearts"), r.HEARTS)
        self.assertEqual(r.suit_name(r.HEARTS), "hearts")
        for suit in r.SUITS:
            self.assertEqual(r.parse_suit(r.suit_name(suit)), suit)

    def test_rejects_a_nonsense_suit(self):
        with self.assertRaises(ValueError):
            r.parse_suit("purple")


if __name__ == "__main__":
    unittest.main()
