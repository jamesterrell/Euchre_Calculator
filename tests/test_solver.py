"""
Unit tests for fast_search.py.

fast_search is checked three ways, none of which trusts the archived solver:

  1. hand-built positions whose value follows from the rules by inspection;
  2. against reference_solver.py and against euchre_testkit.full_minimax, an
     exhaustive no-pruning minimax -- the alpha-beta search must return the same
     value as a search that cannot prune anything away;
  3. by replaying the returned line and re-deriving seat order, card ownership,
     the follow-suit rule, each trick winner and the final score.

test_fast_search.py remains the broad randomised sweep; this file pins the
specific behaviours and the edge cases.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
import io
import random
import unittest

import numpy as np

import euchre_testkit as k
from fast_search import solve, solve_line, definitive_winner, _decode
from reference_solver import solve_py, hands_to_py

# The canonical fixture, also in test_hand.txt. True value 2 with
# starting_player=2, caller=0.
TEST_HAND = k.deal(
    ["JS", "JC", "9C", "9H", "9D"],
    ["KD", "AC", "TS", "QS", "TC"],
    ["9S", "TD", "AD", "QH", "JD"],
    ["JH", "AH", "QD", "KC", "TH"],
)

# p0 holds the five highest cards in the game and leads.
TOP_FIVE_TRUMP_P0 = k.deal(
    ["JS", "JC", "AS", "KS", "QS"],
    ["AH", "KH", "QH", "JH", "TH"],
    ["AD", "KD", "QD", "JD", "TD"],
    ["AC", "KC", "QC", "TC", "9C"],
)

# The same five cards, moved to p1.
TOP_FIVE_TRUMP_P1 = k.deal(
    ["AH", "KH", "QH", "JH", "TH"],
    ["JS", "JC", "AS", "KS", "QS"],
    ["AC", "KC", "QC", "TC", "9C"],
    ["AD", "KD", "QD", "JD", "TD"],
)


def random_deals(n, seed):
    """n (deal, starting_player, caller) triples, reproducibly."""
    rng = random.Random(seed)
    names = list(k.CARD)
    out = []
    for _ in range(n):
        rng.shuffle(names)
        out.append((k.deal(names[0:5], names[5:10], names[10:15], names[15:20]),
                    rng.randrange(4), rng.randrange(4)))
    return out


def played_card(ps, pv, trick, seat):
    """The card name fast_search reports at [trick, seat] of a returned line."""
    return k.name_of(np.array(_decode(ps[trick, seat], pv[trick, seat])))


class TestKnownPositions(unittest.TestCase):
    def test_canonical_fixture(self):
        self.assertEqual(int(solve(TEST_HAND, 2, 0)[0]), 2)

    def test_march_is_plus_two(self):
        """p0 holds the top five cards, so the calling team takes all five tricks."""
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P0, 0, 0)[0]), 2)

    def test_taking_no_tricks_is_minus_two(self):
        """The same layout read from the other team: nothing, so euchred."""
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P0, 0, 1)[0]), -2)

    def test_opponents_hold_everything(self):
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P1, 1, 0)[0]), -2)
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P1, 1, 1)[0]), 2)

    def test_exactly_three_tricks_is_plus_one(self):
        d = k.deal(
            ["TC", "QS", "KS", "QH", "JC"],
            ["JD", "9H", "AD", "KH", "TD"],
            ["TH", "JH", "9C", "QD", "KD"],
            ["KC", "AH", "9D", "TS", "AC"],
        )
        self.assertEqual(int(solve(d, 0, 0)[0]), 1)

    def test_score_is_always_from_the_calling_teams_side(self):
        for caller, want in ((0, 2), (2, 2), (1, -2), (3, -2)):
            self.assertEqual(int(solve(TOP_FIVE_TRUMP_P0, 0, caller)[0]), want,
                             "caller=%d" % caller)

    def test_only_the_three_scores_ever_occur(self):
        for hands, sp, caller in random_deals(120, seed=3):
            self.assertIn(int(solve(hands, sp, caller)[0]), (-2, 1, 2))


class TestLeftBower(unittest.TestCase):
    """The left bower is trump, which is the rule the encoding exists to enforce."""

    def test_forced_to_follow_a_trump_lead(self):
        """p1's only trump is the left bower, so a trump lead must force it out."""
        d = k.deal(
            ["JS", "9D", "TD", "JD", "QD"],
            ["JC", "9H", "TH", "JH", "QH"],
            ["9S", "TS", "QS", "KS", "AS"],
            ["KD", "AD", "KH", "AH", "9C"],
        )
        _, ps, pv, pp, _ = solve_line(d, 0, 0)
        self.assertEqual(played_card(ps, pv, 0, 0), "JS", "p0 should lead trump")
        self.assertEqual(int(pp[0, 1]), 1)
        got = played_card(ps, pv, 0, 1)
        self.assertEqual(got, "JC",
                         "the left bower must follow a trump lead, got %s" % got)

    def test_takes_a_trick_the_led_suit_cannot(self):
        """
        A club is led and p1 holds the left bower plus four hearts. The bower is
        trump, so p1 is void in clubs, is free to play it, and takes the trick.
        Treated as a club it would merely have to follow suit.
        """
        d = k.deal(
            ["AC", "KC", "QC", "TC", "JS"],
            ["JC", "9H", "TH", "JH", "QH"],
            ["9D", "TD", "JD", "QD", "KD"],
            ["9C", "9S", "TS", "QS", "KS"],
        )
        _, ps, pv, _, winners = solve_line(d, 0, 0)
        self.assertEqual(played_card(ps, pv, 0, 0), "AC")
        self.assertEqual(played_card(ps, pv, 0, 1), "JC")
        self.assertEqual(int(winners[0]), 1,
                         "the left bower should trump the ace of clubs")

    def test_beats_the_ace_of_trump(self):
        d = k.deal(
            ["AS", "9D", "TD", "JD", "QD"],
            ["JC", "9H", "TH", "JH", "QH"],
            ["9S", "TS", "QS", "KS", "9C"],
            ["KD", "AD", "KH", "AH", "TC"],
        )
        _, _, _, _, winners = solve_line(d, 0, 0)
        self.assertEqual(int(winners[0]), 1,
                         "the left bower should beat the ace of trump")


class TestSolveLineAgreesWithSolve(unittest.TestCase):
    def test_same_score(self):
        for i, (hands, sp, caller) in enumerate(random_deals(60, seed=17)):
            self.assertEqual(int(solve(hands, sp, caller)[0]),
                             int(solve_line(hands, sp, caller)[0]),
                             "hand %d disagreed" % i)

    def test_returned_line_obeys_the_rules(self):
        for i, (hands, sp, caller) in enumerate(random_deals(60, seed=23)):
            score, ps, pv, pp, winners = solve_line(hands, sp, caller)
            errs = k.replay_line(hands, sp, caller, score, ps, pv, pp, winners)
            self.assertEqual(errs, [], "hand %d: %s" % (i, errs))

    def test_line_shapes(self):
        score, ps, pv, pp, winners = solve_line(TEST_HAND, 2, 0)
        for arr in (ps, pv, pp):
            self.assertEqual(arr.shape, (5, 4))
        self.assertEqual(winners.shape, (5,))
        self.assertTrue(((winners >= 0) & (winners <= 3)).all())

    def test_first_trick_is_led_by_the_starting_player(self):
        for sp in range(4):
            _, _, _, pp, _ = solve_line(TEST_HAND, sp, 0)
            self.assertEqual(int(pp[0, 0]), sp)

    def test_each_trick_is_led_by_the_previous_winner(self):
        for hands, sp, caller in random_deals(20, seed=29):
            _, _, _, pp, winners = solve_line(hands, sp, caller)
            for t in range(1, 5):
                self.assertEqual(int(pp[t, 0]), int(winners[t - 1]))


class TestAgainstIndependentSolvers(unittest.TestCase):
    def test_matches_the_pure_python_reference(self):
        for i, (hands, sp, caller) in enumerate(random_deals(200, seed=41)):
            self.assertEqual(int(solve(hands, sp, caller)[0]),
                             solve_py(hands_to_py(hands), sp, caller),
                             "hand %d disagreed with reference_solver" % i)

    def test_matches_on_suit_concentrated_hands(self):
        """
        Random deals rarely produce heavy voids, which are exactly where the
        follow-suit rule and the per-trick buffer indexing get stressed.
        """
        rng = random.Random(97)
        names = list(k.CARD)
        for i in range(120):
            rng.shuffle(names)
            names.sort(key=lambda n: (k.suit_of(k.CARD[n]), rng.random()))
            hands = k.deal(names[0:5], names[5:10], names[10:15], names[15:20])
            sp, caller = i % 4, (i // 4) % 4
            self.assertEqual(int(solve(hands, sp, caller)[0]),
                             solve_py(hands_to_py(hands), sp, caller),
                             "suit-concentrated hand %d disagreed" % i)


class TestSymmetries(unittest.TestCase):
    def test_rotating_both_teams_two_seats_preserves_the_score(self):
        """Seats 0/2 and 1/3 are the teams, so a two-seat rotation only relabels."""
        rng = random.Random(5)
        names = list(k.CARD)
        for _ in range(80):
            rng.shuffle(names)
            a = k.deal(names[0:5], names[5:10], names[10:15], names[15:20])
            b = k.deal(names[10:15], names[15:20], names[0:5], names[5:10])
            sp, caller = rng.randrange(4), rng.randrange(4)
            self.assertEqual(int(solve(a, sp, caller)[0]),
                             int(solve(b, (sp + 2) % 4, (caller + 2) % 4)[0]))

    def test_partners_call_the_same_score(self):
        """Only the caller's team matters, never which of the two partners called."""
        for hands, sp, caller in random_deals(80, seed=61):
            self.assertEqual(int(solve(hands, sp, caller)[0]),
                             int(solve(hands, sp, (caller + 2) % 4)[0]))

    def test_reordering_a_hand_does_not_change_the_value(self):
        """A hand is a set; the order the cards sit in the array must not matter."""
        rng = random.Random(7)
        for hands, sp, caller in random_deals(40, seed=71):
            want = int(solve(hands, sp, caller)[0])
            shuffled = hands.copy()
            for p in range(4):
                idx = list(range(5))
                rng.shuffle(idx)
                shuffled[p] = hands[p][idx]
            self.assertEqual(int(solve(shuffled, sp, caller)[0]), want)


class TestSearchHygiene(unittest.TestCase):
    def test_does_not_mutate_its_input(self):
        """The search mutates state in place and restores it; prove it restores."""
        for hands, sp, caller in random_deals(20, seed=83):
            before = hands.copy()
            solve(hands, sp, caller)
            solve_line(hands, sp, caller)
            np.testing.assert_array_equal(hands, before)

    def test_is_deterministic(self):
        for hands, sp, caller in random_deals(20, seed=89):
            first = int(solve(hands, sp, caller)[0])
            for _ in range(3):
                self.assertEqual(int(solve(hands, sp, caller)[0]), first)

    def test_reports_a_node_count(self):
        self.assertGreater(int(solve(TEST_HAND, 2, 0)[1]), 0)


class TestInputValidation(unittest.TestCase):
    """
    Regression: none of this was checked. The hot loop has no bounds checking,
    so an out-of-range seat indexed straight into the (4,) / (4, 5) state
    arrays -- starting_player=6 segfaulted the interpreter -- and a hand with a
    card count other than five returned -1000 from solve or wrote past the end
    of solve_line's trick buffers.
    """

    def test_rejects_an_out_of_range_starting_player(self):
        for bad in (-1, -2, 4, 6):
            with self.assertRaises(ValueError, msg="starting_player=%d" % bad):
                solve(TEST_HAND, bad, 0)
            with self.assertRaises(ValueError, msg="starting_player=%d" % bad):
                solve_line(TEST_HAND, bad, 0)

    def test_rejects_an_out_of_range_caller(self):
        for bad in (-1, 4, 7):
            with self.assertRaises(ValueError, msg="caller=%d" % bad):
                solve(TEST_HAND, 0, bad)
            with self.assertRaises(ValueError, msg="caller=%d" % bad):
                solve_line(TEST_HAND, 0, bad)

    def test_rejects_hands_that_are_not_five_cards(self):
        for n in (3, 4, 6):
            wrong = np.ascontiguousarray(
                np.tile(TEST_HAND[:, :1, :], (1, n, 1)), dtype=np.int64)
            with self.assertRaises(ValueError, msg="%d cards" % n):
                solve(wrong, 0, 0)
            with self.assertRaises(ValueError, msg="%d cards" % n):
                solve_line(wrong, 0, 0)

    def test_rejects_a_table_that_is_not_four_players(self):
        with self.assertRaises(ValueError):
            solve(np.ascontiguousarray(TEST_HAND[:3]), 0, 0)

    def test_definitive_winner_rejects_a_badly_shaped_deal(self):
        with self.assertRaises(ValueError):
            definitive_winner(TEST_HAND.reshape(4, 10), 0, 0)


class TestDefinitiveWinner(unittest.TestCase):
    def test_matches_solve(self):
        for hands, sp, caller in random_deals(30, seed=101):
            self.assertEqual(definitive_winner(hands, sp, caller),
                             int(solve(hands, sp, caller)[0]))

    def test_returns_a_plain_int(self):
        self.assertIsInstance(definitive_winner(TEST_HAND, 2, 0), int)

    def test_accepts_lists_and_other_dtypes(self):
        want = definitive_winner(TEST_HAND, 2, 0)
        self.assertEqual(definitive_winner(TEST_HAND.tolist(), 2, 0), want)
        self.assertEqual(definitive_winner(TEST_HAND.astype(np.int32), 2, 0), want)

    def test_verbose_prints_the_line_and_returns_the_same_score(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score = definitive_winner(TEST_HAND, 2, 0, verbose=True)
        out = buf.getvalue()
        self.assertEqual(score, definitive_winner(TEST_HAND, 2, 0))
        for t in range(1, 6):
            self.assertIn("Trick %d:" % t, out)
            self.assertIn("Trick %d winner:" % t, out)


class TestAgainstExhaustiveMinimax(unittest.TestCase):
    """
    The slow tier. full_minimax has no alpha-beta and no forced-outcome cutoffs,
    so it cannot be wrong in the ways a pruning search can -- but it is far too
    branchy to point at an arbitrary hand, so these run against the pinned
    BRUTE_FORCEABLE fixtures whose cost is known.
    """

    def test_alpha_beta_returns_the_exact_minimax_value(self):
        for label, hands, sp, caller, expected in k.brute_forceable_deals():
            want, _ = k.full_minimax(hands, sp, caller)
            self.assertEqual(want, expected,
                             "%s: the fixture's pinned score is stale" % label)
            self.assertEqual(int(solve(hands, sp, caller)[0]), want,
                             "%s: alpha-beta disagreed with exhaustive minimax" % label)

    def test_the_cutoffs_actually_prune(self):
        """
        Documents the claim that the cutoffs skip almost the whole tree. Uses
        the canonical fixture rather than a cheap one: pruning pays off in
        proportion to tree size, and on the small fixtures the saving is only
        1-5%. Costs ~3M exhaustive nodes, a little over three seconds.
        """
        ab_nodes = int(solve(TEST_HAND, 2, 0)[1])
        _, full_nodes = k.full_minimax(TEST_HAND, 2, 0)
        self.assertLess(ab_nodes, full_nodes // 100,
                        "expected under 1%% of %d nodes, visited %d"
                        % (full_nodes, ab_nodes))


if __name__ == "__main__":
    unittest.main()
