"""
Unit tests for fast_search's partially-played-position entry points.

`solve` starts from a fresh deal, which is the one position a player choosing a
card is never in. `solve_position` and `position_moves` take the position as it
stands, and the danger in that is entirely in the bookkeeping: a card count
that disagrees with the trick number, a trick row filled in the wrong order, a
seat that has already played being asked to play again. None of those are
errors the search can detect -- it will solve whatever it is handed and return
a number -- so they are checked here instead.

The strongest test in this file is `TestReplay`. Along a double-dummy optimal
line the value of the position cannot change: both sides are already playing
the best they have, so nothing either does moves the number. Walking
`solve_line`'s own line one card at a time and re-solving from scratch at every
ply therefore has a known answer at all 20 plies, and it exercises mid-trick
entry, mid-hand entry and the trick-boundary handoff on every deal it touches.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import numpy as np

import fast_search as fs
import game
import rotation as r
from euchre_testkit import deal as kit_deal, name_of

TEST_HAND = kit_deal(
    ("JS", "JC", "AS", "KH", "9D"),
    ("QS", "TS", "AH", "TH", "9H"),
    ("9S", "AD", "KD", "QD", "TD"),
    ("KS", "AC", "KC", "QC", "TC"),
)


def engine_deal(seed, dealer=0):
    """A random deal, rotated into the solver's frame by its own up-card."""
    d = game.deal_random(rng=random.Random(seed), dealer=dealer)
    return r.deal_to_engine(d.hands, d.up_card.suit), d.first_bidder


def pad(hands):
    """Ragged per-seat card lists -> a (4, C, 2) array plus its counts."""
    counts = np.array([len(h) for h in hands], dtype=np.int64)
    width = max(1, int(counts.max()))
    arr = np.zeros((4, width, 2), dtype=np.int64)
    for seat, hand in enumerate(hands):
        for i, card in enumerate(hand):
            arr[seat, i] = card
    return arr, counts


def walk(dealt, starting_player, caller, alone):
    """
    Every position along solve_line's optimal line, one per card played.

    Yields (hands, trick_cards, trick_players, to_act, caller_tricks,
    trick_no) immediately before each of the 20 (or 15) cards is played, then
    finally the line's score.

    Every yield is a copy. The state here is mutated in place as the line is
    walked, so handing out references would give a caller that collects them --
    `list(walk(...))`, which is what the tests do -- twenty views of the
    finished position rather than twenty positions.
    """
    score, ps, pv, pp, winners = fs.solve_line(
        dealt, starting_player, caller, alone)
    width = ps.shape[1]
    sitting = (caller + 2) % 4 if alone else -1

    hands = [[[int(v) for v in card] for card in dealt[p]] for p in range(4)]
    if sitting >= 0:
        hands[sitting] = []

    caller_tricks = 0
    for trick_no in range(5):
        cards, seats = [], []
        for k in range(width):
            seat = int(pp[trick_no, k])
            card = fs._decode(ps[trick_no, k], pv[trick_no, k])
            yield ([list(h) for h in hands], list(cards), list(seats),
                   seat, caller_tricks, trick_no)
            hands[seat].remove(card)
            cards.append(card)
            seats.append(seat)
        if int(winners[trick_no]) % 2 == caller % 2:
            caller_tricks += 1
    yield int(score)


class TestFreshPositions(unittest.TestCase):
    """A position with nothing played yet is just `solve` by another route."""

    def check(self, dealt, starting_player, caller, alone):
        want = fs.definitive_winner(dealt, starting_player, caller, alone=alone)

        counts = np.full(4, 5, dtype=np.int64)
        leader = starting_player
        if alone:
            sitting = (caller + 2) % 4
            counts[sitting] = 0
            if leader == sitting:
                leader = (leader + 1) % 4

        got, _ = fs.solve_position(
            dealt, counts, np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.int64), leader, caller, 0, 0, alone)
        self.assertEqual(got, want)

    def test_test_hand(self):
        self.check(TEST_HAND, 2, 0, False)

    def test_test_hand_alone(self):
        self.check(TEST_HAND, 2, 0, True)

    def test_random_deals(self):
        for seed in range(12):
            dealt, leader = engine_deal(seed)
            for caller in range(4):
                for alone in (False, True):
                    with self.subTest(seed=seed, caller=caller, alone=alone):
                        self.check(dealt, leader, caller, alone)


class TestReplay(unittest.TestCase):
    """
    The value of a position is constant along an optimal line.

    If re-solving part-way through ever disagrees with the score the line ended
    on, the position was described to the search incorrectly -- which is the
    only way this can go wrong, since the search itself is already tested.
    """

    def check(self, dealt, starting_player, caller, alone):
        steps = list(walk(dealt, starting_player, caller, alone))
        score = steps.pop()

        for hands, cards, seats, to_act, caller_tricks, trick_no in steps:
            arr, counts = pad(hands)
            trick_cards = np.array(cards, dtype=np.int64).reshape(-1, 2)
            trick_players = np.array(seats, dtype=np.int64)

            got, _ = fs.solve_position(
                arr, counts, trick_cards, trick_players, to_act, caller,
                caller_tricks, trick_no, alone)
            self.assertEqual(
                got, score,
                "position value moved at trick %d, seat %d" % (trick_no, to_act))

    def test_test_hand(self):
        self.check(TEST_HAND, 2, 0, False)

    def test_test_hand_alone(self):
        self.check(TEST_HAND, 2, 0, True)

    def test_random_deals(self):
        for seed in range(8):
            dealt, leader = engine_deal(seed)
            for alone in (False, True):
                with self.subTest(seed=seed, alone=alone):
                    self.check(dealt, leader, 0, alone)


class TestPositionMoves(unittest.TestCase):
    """position_moves must agree with solve_position, card for card."""

    def test_best_move_matches_the_position_value(self):
        for seed in range(6):
            dealt, leader = engine_deal(seed)
            for alone in (False, True):
                caller = 0
                steps = list(walk(dealt, leader, caller, alone))
                score = steps.pop()

                for hands, cards, seats, to_act, ct, trick_no in steps:
                    arr, counts = pad(hands)
                    tc = np.array(cards, dtype=np.int64).reshape(-1, 2)
                    tp = np.array(seats, dtype=np.int64)
                    idx, vals, _ = fs.position_moves(
                        arr, counts, tc, tp, to_act, caller, ct, trick_no,
                        alone)

                    self.assertGreater(len(idx), 0)
                    # The seat to act maximises for its own team.
                    best = (max(vals) if to_act % 2 == caller % 2
                            else min(vals))
                    with self.subTest(seed=seed, alone=alone, trick=trick_no):
                        self.assertEqual(int(best), score)

    def test_a_void_seat_may_play_anything(self):
        # Seat 1 leads the ace of hearts. Seats 2 and 3 hold no hearts at all,
        # so nothing is forced on either of them.
        hands = [[[int(v) for v in c] for c in TEST_HAND[p]] for p in range(4)]
        trick, seats = [hands[1].pop(2)], [1]        # AH

        arr, counts = pad(hands)
        idx, _, _ = fs.position_moves(
            arr, counts, np.array(trick, dtype=np.int64),
            np.array(seats, dtype=np.int64), 2, 0, 0, 0, False)
        self.assertEqual(len(idx), 5)                # 9S AD KD QD TD

        trick.append(hands[2].pop(0))                # seat 2 ruffs with 9S
        seats.append(2)
        arr, counts = pad(hands)
        idx, _, _ = fs.position_moves(
            arr, counts, np.array(trick, dtype=np.int64),
            np.array(seats, dtype=np.int64), 3, 0, 0, 0, False)
        self.assertEqual(len(idx), 5)                # KS AC KC QC TC

    def test_following_suit_is_forced(self):
        # Seat 0 leads the nine of diamonds. Seat 2 holds four diamonds and the
        # nine of trump, and the nine of trump must not be on offer.
        hands = [[[int(v) for v in c] for c in TEST_HAND[p]] for p in range(4)]
        trick, seats = [hands[0].pop(4)], [0]        # 9D

        arr, counts = pad(hands)
        idx, _, _ = fs.position_moves(
            arr, counts, np.array(trick, dtype=np.int64),
            np.array(seats, dtype=np.int64), 1, 0, 0, 0, False)
        self.assertEqual(len(idx), 5)                # seat 1 has no diamonds

        trick.append(hands[1].pop(4))                # seat 1 pitches 9H
        seats.append(1)
        arr, counts = pad(hands)
        idx, _, _ = fs.position_moves(
            arr, counts, np.array(trick, dtype=np.int64),
            np.array(seats, dtype=np.int64), 2, 0, 0, 0, False)

        offered = sorted(name_of(hands[2][int(i)]) for i in idx)
        self.assertEqual(offered, ["AD", "KD", "QD", "TD"])

    def test_the_left_bower_cannot_follow_its_printed_suit(self):
        # Seat 3 leads the ace of clubs. Seat 0 holds the jack of clubs, which
        # is trump -- so it is not a club, and seat 0 is void. If the encoding
        # ever stopped saying so, this is where it would show: the search would
        # offer seat 0 exactly one card instead of five.
        hands = [[[int(v) for v in c] for c in TEST_HAND[p]] for p in range(4)]
        trick, seats = [hands[3].pop(1)], [3]        # AC

        arr, counts = pad(hands)
        idx, _, _ = fs.position_moves(
            arr, counts, np.array(trick, dtype=np.int64),
            np.array(seats, dtype=np.int64), 0, 0, 0, 0, False)
        self.assertEqual(len(idx), 5)                # JS JC AS KH 9D, all legal


class TestValidation(unittest.TestCase):
    """
    Bad positions must raise, not segfault.

    The solver has no bounds checking, so these used to be the difference
    between an exception and losing the interpreter -- the same reason
    `_validate` exists for `solve`.
    """

    def setUp(self):
        self.hands = TEST_HAND
        self.counts = np.full(4, 5, dtype=np.int64)
        self.empty = np.zeros((0, 2), dtype=np.int64)
        self.no_seats = np.zeros(0, dtype=np.int64)

    def solve(self, **kw):
        args = dict(hands=self.hands, counts=self.counts,
                    trick_cards=self.empty, trick_players=self.no_seats,
                    to_act=0, caller=0, caller_tricks=0, trick_no=0,
                    alone=False)
        args.update(kw)
        return fs.solve_position(**args)

    def test_a_sound_position_is_accepted(self):
        self.solve()

    def test_seat_out_of_range(self):
        for seat in (-1, 4, 99):
            with self.assertRaises(ValueError):
                self.solve(to_act=seat)
            with self.assertRaises(ValueError):
                self.solve(caller=seat)

    def test_trick_number_out_of_range(self):
        with self.assertRaises(ValueError):
            self.solve(trick_no=5)
        with self.assertRaises(ValueError):
            self.solve(trick_no=-1)

    def test_counts_must_match_the_trick_number(self):
        # Five cards each is trick 0's shape, not trick 2's.
        with self.assertRaises(ValueError):
            self.solve(trick_no=2, caller_tricks=1)

    def test_more_tricks_won_than_played(self):
        with self.assertRaises(ValueError):
            self.solve(caller_tricks=3)

    def test_a_full_trick_has_nobody_to_act(self):
        cards = np.array([[int(v) for v in TEST_HAND[p, 0]] for p in range(4)],
                         dtype=np.int64)
        with self.assertRaises(ValueError):
            self.solve(trick_cards=cards,
                       trick_players=np.arange(4, dtype=np.int64))

    def test_seats_must_act_in_turn(self):
        # Seat 1 has played its last card, 9H, so counts[1] covers the four
        # that are left; a played card still showing in the hand is a
        # duplicate, and caught separately below.
        counts = np.array([5, 4, 5, 5], dtype=np.int64)
        card = np.array([[int(v) for v in TEST_HAND[1, 4]]], dtype=np.int64)
        # Seat 1 played; seat 2 is next, not seat 3.
        with self.assertRaises(ValueError):
            self.solve(counts=counts, trick_cards=card,
                       trick_players=np.array([1], dtype=np.int64), to_act=3)
        self.solve(counts=counts, trick_cards=card,
                   trick_players=np.array([1], dtype=np.int64), to_act=2)

    def test_a_seat_cannot_play_twice_to_a_trick(self):
        counts = np.array([5, 4, 5, 5], dtype=np.int64)
        card = np.array([[int(v) for v in TEST_HAND[1, 4]]], dtype=np.int64)
        with self.assertRaises(ValueError):
            self.solve(counts=counts, trick_cards=card,
                       trick_players=np.array([1], dtype=np.int64), to_act=1)

    def test_duplicate_cards_are_rejected(self):
        hands = np.array(TEST_HAND, dtype=np.int64)
        hands[1, 0] = hands[0, 0]
        with self.assertRaises(ValueError):
            self.solve(hands=hands)

    def test_the_sitting_seat_holds_nothing(self):
        counts = np.full(4, 5, dtype=np.int64)
        with self.assertRaises(ValueError):
            self.solve(counts=counts, to_act=1, alone=True)
        counts[2] = 0
        self.solve(counts=counts, to_act=1, alone=True)

    def test_the_sitting_seat_cannot_act(self):
        counts = np.array([5, 5, 0, 5], dtype=np.int64)
        with self.assertRaises(ValueError):
            self.solve(counts=counts, to_act=2, alone=True)

    def test_a_loner_trick_is_three_cards(self):
        counts = np.array([4, 4, 0, 4], dtype=np.int64)
        cards = np.array([[int(v) for v in TEST_HAND[p, 0]]
                          for p in (0, 1, 3)], dtype=np.int64)
        with self.assertRaises(ValueError):
            fs.solve_position(TEST_HAND, counts, cards,
                              np.array([0, 1, 3], dtype=np.int64),
                              0, 0, 0, 0, True)


if __name__ == "__main__":
    unittest.main()
