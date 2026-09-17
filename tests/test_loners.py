"""
Unit tests for loners -- a call made with the caller's partner sitting out.

A loner is three structural changes, and each one is tested on its own here:

  * the partner's hand leaves the game. Its cards are dealt but unplayable, so
    they are as out of play as the kitty, and nothing may reach them.
  * a trick is three cards. Turn order steps over the sitting seat, and so does
    the opening lead when the eldest hand is the one sitting out.
  * a march pays 4 instead of 2. Three or four tricks is still 1, and being
    euchred alone still hands over only 2, so the reachable values are
    {-2, 1, 4} and never 2.

`fast_search._search_alone` is a deliberate copy of `_search` (see the comment
above it), so it is checked against the same two independent oracles the
four-handed search is: `reference_solver.solve_py` and the no-pruning minimax in
`euchre_testkit.full_minimax`. That cross-check is what keeps the copy honest.
"""
import os
import sys

# Also runnable directly (python tests/test_loners.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import numpy as np

import bidding as b
import euchre_testkit as k
import game
import rotation as r
from fast_search import (definitive_winner, sitting_seat, solve, solve_line,
                         LONE_MARCH, MARCH)
from reference_solver import hands_to_py, solve_py

# The canonical fixture. Four-handed it is a march for team 0; alone, seat 0
# holds both bowers and three losers, so it takes two tricks and is euchred.
TEST_HAND = k.deal(
    ["JS", "JC", "9C", "9H", "9D"],
    ["KD", "AC", "TS", "QS", "TC"],
    ["9S", "TD", "AD", "QH", "JD"],
    ["JH", "AH", "QD", "KC", "TH"],
)

# Seat 0 holds the five top trumps, so nothing can be done about it whoever is
# playing: all five tricks, and alone that is worth 4.
TOP_FIVE_TRUMP_P0 = k.deal(
    ["JS", "JC", "AS", "KS", "QS"],
    ["AH", "KH", "QH", "JH", "TH"],
    ["AD", "KD", "QD", "JD", "TD"],
    ["AC", "KC", "QC", "TC", "9C"],
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


class TestLoneScoring(unittest.TestCase):
    def test_a_lone_march_pays_four(self):
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P0, 0, 0, True)[0]), LONE_MARCH)

    def test_the_same_hand_four_handed_pays_two(self):
        self.assertEqual(int(solve(TOP_FIVE_TRUMP_P0, 0, 0)[0]), MARCH)

    def test_being_euchred_alone_still_costs_two(self):
        self.assertEqual(int(solve(TEST_HAND, 2, 0, True)[0]), -2)

    def test_pinned_outcomes(self):
        for label, hands, sp, caller, score in k.brute_forceable_lone_deals():
            with self.subTest(label):
                self.assertEqual(int(solve(hands, sp, caller, True)[0]), score)

    def test_only_three_values_are_reachable_and_two_is_not_one_of_them(self):
        seen = set()
        for hands, sp, caller in random_deals(120, seed=4):
            seen.add(int(solve(hands, sp, caller, True)[0]))
        self.assertEqual(seen, {-2, 1, LONE_MARCH})

    def test_scoring_helper_agrees_with_the_solver(self):
        self.assertEqual(k.score_from_tricks(5, alone=True), LONE_MARCH)
        self.assertEqual(k.score_from_tricks(5, alone=False), MARCH)
        for tricks in (3, 4):
            self.assertEqual(k.score_from_tricks(tricks, alone=True), 1)
        for tricks in (0, 1, 2):
            self.assertEqual(k.score_from_tricks(tricks, alone=True), -2)


class TestSittingPartner(unittest.TestCase):
    def test_the_sitting_seat_is_the_callers_partner(self):
        for caller in range(4):
            self.assertEqual(sitting_seat(caller), (caller + 2) % 4)
        self.assertEqual(sitting_seat(0, alone=False), -1)

    def test_tricks_are_three_cards_wide(self):
        _, ps, pv, pp, _ = solve_line(TEST_HAND, 2, 0, True)
        for arr in (ps, pv, pp):
            self.assertEqual(arr.shape, (5, 3))

    def test_the_partner_never_plays_a_card(self):
        for hands, sp, caller in random_deals(25, seed=5):
            _, _, _, pp, _ = solve_line(hands, sp, caller, True)
            with self.subTest(caller=caller):
                self.assertNotIn(sitting_seat(caller), set(pp.ravel().tolist()))

    def test_the_lead_steps_over_a_sitting_eldest_hand(self):
        # caller 0 plays alone, so seat 2 sits out; it cannot lead trick one.
        _, _, _, pp, _ = solve_line(TEST_HAND, 2, 0, True)
        self.assertEqual(int(pp[0, 0]), 3)

    def test_the_lead_is_unchanged_when_the_eldest_hand_plays(self):
        _, _, _, pp, _ = solve_line(TEST_HAND, 1, 0, True)
        self.assertEqual(int(pp[0, 0]), 1)

    def test_the_partners_cards_order_does_not_matter(self):
        """The sitting hand is out of play, so shuffling it changes nothing."""
        rng = random.Random(6)
        for hands, sp, caller in random_deals(15, seed=7):
            base = int(solve(hands, sp, caller, True)[0])
            shuffled = hands.copy()
            order = list(range(5))
            rng.shuffle(order)
            shuffled[sitting_seat(caller)] = hands[sitting_seat(caller)][order]
            with self.subTest(caller=caller):
                self.assertEqual(int(solve(shuffled, sp, caller, True)[0]), base)


class TestAgainstIndependentSolvers(unittest.TestCase):
    def test_matches_the_pure_python_reference(self):
        for hands, sp, caller in random_deals(120, seed=8):
            with self.subTest(sp=sp, caller=caller):
                self.assertEqual(
                    solve_py(hands_to_py(hands), sp, caller, alone=True),
                    int(solve(hands, sp, caller, True)[0]))

    def test_matches_exhaustive_minimax_on_the_pinned_deals(self):
        """
        Alpha-beta must return the value a search that prunes nothing returns.

        These three total about 37k nodes. A loner tree is far smaller than the
        four-handed one -- a whole hand has left the game -- so this is cheap
        where the four-handed version of the same check is not.
        """
        for label, hands, sp, caller, score in k.brute_forceable_lone_deals():
            with self.subTest(label):
                value, _ = k.full_minimax(hands, sp, caller, alone=True)
                self.assertEqual(value, score)
                self.assertEqual(int(solve(hands, sp, caller, True)[0]), value)

    def test_the_pinned_node_counts_are_still_accurate(self):
        for label, hands, sp, caller, _score, nodes in k.BRUTE_FORCEABLE_ALONE:
            with self.subTest(label):
                _, visited = k.full_minimax(k.deal(*hands), sp, caller,
                                            alone=True)
                self.assertEqual(visited, nodes)


class TestReturnedLine(unittest.TestCase):
    def test_solve_line_agrees_with_solve(self):
        for hands, sp, caller in random_deals(40, seed=9):
            with self.subTest(sp=sp, caller=caller):
                self.assertEqual(int(solve_line(hands, sp, caller, True)[0]),
                                 int(solve(hands, sp, caller, True)[0]))

    def test_the_returned_line_obeys_the_rules(self):
        for hands, sp, caller in random_deals(40, seed=10):
            score, ps, pv, pp, winners = solve_line(hands, sp, caller, True)
            errs = k.replay_line(hands, sp, caller, score, ps, pv, pp, winners,
                                 alone=True)
            with self.subTest(sp=sp, caller=caller):
                self.assertEqual(errs, [])

    def test_each_trick_is_led_by_the_previous_winner(self):
        for hands, sp, caller in random_deals(15, seed=11):
            _, _, _, pp, winners = solve_line(hands, sp, caller, True)
            for t in range(1, 5):
                with self.subTest(t=t):
                    self.assertEqual(int(pp[t, 0]), int(winners[t - 1]))

    def test_replay_rejects_a_line_of_the_wrong_width(self):
        """The four-handed line is not a legal loner line, and vice versa."""
        score, ps, pv, pp, winners = solve_line(TEST_HAND, 1, 0)
        errs = k.replay_line(TEST_HAND, 1, 0, score, ps, pv, pp, winners,
                             alone=True)
        self.assertTrue(errs)


class TestSearchHygiene(unittest.TestCase):
    def test_does_not_mutate_its_input(self):
        before = TEST_HAND.copy()
        solve(TEST_HAND, 1, 0, True)
        solve_line(TEST_HAND, 1, 0, True)
        np.testing.assert_array_equal(TEST_HAND, before)

    def test_is_deterministic(self):
        first = [int(solve(h, sp, c, True)[0]) for h, sp, c in random_deals(20, 12)]
        again = [int(solve(h, sp, c, True)[0]) for h, sp, c in random_deals(20, 12)]
        self.assertEqual(first, again)

    def test_loners_are_cheaper_to_solve_in_aggregate(self):
        """
        A hand leaving the game is the cheapest pruning there is -- about 3x
        fewer nodes over a sweep. It is *not* true deal by deal: a four-handed
        position often trips a forced-outcome cutoff early that the same layout
        alone does not, so individual counts go either way.
        """
        lone = four = 0
        for hands, sp, caller in random_deals(20, seed=13):
            lone += int(solve(hands, sp, caller, True)[1])
            four += int(solve(hands, sp, caller)[1])
        self.assertLess(lone, four)

    def test_definitive_winner_takes_alone_as_a_keyword(self):
        self.assertEqual(definitive_winner(TOP_FIVE_TRUMP_P0, 0, 0, alone=True),
                         LONE_MARCH)
        self.assertEqual(definitive_winner(TOP_FIVE_TRUMP_P0, 0, 0), MARCH)

    def test_verbose_names_the_sitting_seat(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score = definitive_winner(TEST_HAND, 2, 0, verbose=True, alone=True)
        text = buf.getvalue()
        self.assertEqual(score, -2)
        self.assertIn("seat 2 sits out", text)


# ------------------------------------------------------------------- bidding

# Deals where allowing loners changes the auction. Both are lone marches: on the
# first, team 0 gains; on the second, team 1 does, which is why `allow_loners`
# is not monotone for either team.
LONE_MARCH_FOR_TEAM0 = (1, 75)      # (dealer, seed): seat 2 orders up alone
LONE_MARCH_FOR_TEAM1 = (0, 70)      # (dealer, seed): seat 1 orders up alone


def a_deal(seed=0, dealer=0):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


class TestContract(unittest.TestCase):
    def test_alone_defaults_to_false(self):
        _, contract = b.order_up(a_deal(), 1)
        self.assertFalse(contract.alone)
        self.assertIsNone(contract.sitting)

    def test_a_lone_contract_records_the_sitting_seat(self):
        _, contract = b.order_up(a_deal(), 1, alone=True)
        self.assertTrue(contract.alone)
        self.assertEqual(contract.sitting, 3)
        self.assertIn("alone", str(contract))
        self.assertIn("seat 3 sits out", str(contract))

    def test_contract_solve_matches_the_reported_value(self):
        for seat in range(4):
            value, contract = b.order_up(a_deal(3), seat, alone=True)
            with self.subTest(seat=seat):
                self.assertEqual(b.net_to_team0(contract.solve(), seat), value)

    def test_play_value_passes_alone_through(self):
        base = a_deal(3)
        d = base.pick_up(discard=base.hands[base.dealer][0])
        for caller in range(4):
            hands = r.deal_to_engine(d.hands, d.up_card.suit)
            with self.subTest(caller=caller):
                self.assertEqual(
                    b.play_value(d, d.up_card.suit, caller, alone=True),
                    definitive_winner(hands, d.first_bidder, caller, alone=True))


class TestOrderUpAlone(unittest.TestCase):
    def test_every_discard_is_worth_the_same_when_the_dealer_sits_out(self):
        """
        Why order_up short-circuits. If the caller's partner is the dealer, the
        dealer picks up into a hand that never plays, so which card it pitches
        is both unobservable and worth nothing -- all five must agree, and the
        function solves one of them rather than five.
        """
        for seed in range(6):
            d = a_deal(seed, dealer=0)
            caller = 2                      # partner of seat 0, the dealer
            values = set()
            for card in d.hands[0]:
                after = d.pick_up(discard=card)
                values.add(b.play_value(after, d.up_card.suit, caller,
                                        alone=True))
            with self.subTest(seed=seed):
                self.assertEqual(len(values), 1)
                self.assertEqual(b.order_up(d, caller, alone=True)[0],
                                 b.net_to_team0(values.pop(), caller))

    def test_the_dealer_still_chooses_against_a_lone_opponent(self):
        """With an opponent calling alone the discard is a real decision."""
        d = a_deal(20, dealer=2)
        caller = 1                          # dealer's opponent, seat 3 sits out
        value, contract = b.order_up(d, caller, alone=True)
        worst_for_dealer = max(
            b.play_value(d.pick_up(discard=c), d.up_card.suit, caller,
                         alone=True)
            for c in d.hands[2])
        self.assertLessEqual(contract.solve(), worst_for_dealer)
        self.assertEqual(b.net_to_team0(contract.solve(), caller), value)

    def test_a_loner_can_beat_the_same_call_four_handed(self):
        dealer, seed = LONE_MARCH_FOR_TEAM0
        d = a_deal(seed, dealer)
        seat = d.first_bidder
        self.assertGreater(b.value_to(seat, b.order_up(d, seat, alone=True)[0]),
                           b.value_to(seat, b.order_up(d, seat)[0]))

    def test_but_almost_never_does(self):
        """
        Going alone is the exception, and this is why: it only gains when the
        caller can take all five tricks unaided, since three or four tricks is
        worth 1 either way and being euchred costs the same 2. Measured over
        these 32 deals at the eldest seat: better on 0, worse on 8, equal on 24.
        Same story as the ~1% of auctions `allow_loners` changes.
        """
        better = worse = 0
        for dealer in range(4):
            for seed in range(8):
                d = a_deal(seed, dealer=dealer)
                seat = d.first_bidder
                four = b.value_to(seat, b.order_up(d, seat)[0])
                lone = b.value_to(seat, b.order_up(d, seat, alone=True)[0])
                better += lone > four
                worse += lone < four
        self.assertLessEqual(better, 3,
                             "gaining by going alone should be rare, got %d/32"
                             % better)
        self.assertGreater(worse, 0, "sitting the partner down should cost "
                                     "something on some of these deals")


class TestNameSuitAlone(unittest.TestCase):
    def test_records_alone_and_no_discard(self):
        d = a_deal()
        trump = (d.up_card.suit + 1) % 4
        value, contract = b.name_suit(d, 0, trump, alone=True)
        self.assertTrue(contract.alone)
        self.assertIsNone(contract.discard)
        self.assertEqual(contract.bidding_round, b.ROUND_TWO)
        self.assertEqual(b.net_to_team0(contract.solve(), 0), value)

    def test_still_rejects_the_turned_suit(self):
        d = a_deal()
        with self.assertRaises(ValueError):
            b.name_suit(d, 0, d.up_card.suit, alone=True)


class TestSolveBiddingWithLoners(unittest.TestCase):
    def test_loners_are_off_by_default(self):
        for dealer in range(4):
            for seed in range(6):
                out = b.solve_bidding(a_deal(seed, dealer))
                with self.subTest(dealer=dealer, seed=seed):
                    self.assertFalse(out.passed_out and out.value != 0)
                    self.assertFalse(out.contract.alone)

    def test_a_lone_march_is_found_for_team_zero(self):
        dealer, seed = LONE_MARCH_FOR_TEAM0
        d = a_deal(seed, dealer)
        out = b.solve_bidding(d, allow_loners=True)
        self.assertTrue(out.contract.alone)
        self.assertEqual(out.value, LONE_MARCH)
        self.assertEqual(b.team_of(out.contract.caller), 0)
        self.assertEqual(b.solve_bidding(d).value, MARCH)

    def test_a_lone_march_is_found_for_team_one(self):
        dealer, seed = LONE_MARCH_FOR_TEAM1
        d = a_deal(seed, dealer)
        out = b.solve_bidding(d, allow_loners=True)
        self.assertTrue(out.contract.alone)
        self.assertEqual(out.value, -LONE_MARCH)
        self.assertEqual(b.team_of(out.contract.caller), 1)

    def test_the_reported_value_matches_the_contract(self):
        for dealer in range(4):
            for seed in range(4):
                out = b.solve_bidding(a_deal(seed, dealer), allow_loners=True)
                with self.subTest(dealer=dealer, seed=seed):
                    self.assertEqual(
                        out.value,
                        b.net_to_team0(out.contract.solve(), out.contract.caller))

    def test_values_stay_on_the_team_zero_scale(self):
        allowed = {-4, -2, -1, 1, 2, 4}
        for dealer in range(4):
            for seed in range(6):
                out = b.solve_bidding(a_deal(seed, dealer), allow_loners=True)
                with self.subTest(dealer=dealer, seed=seed):
                    self.assertIn(out.value, allowed)

    def test_a_lone_contract_is_worth_four_one_or_minus_two_to_its_caller(self):
        for dealer in range(4):
            for seed in range(20):
                out = b.solve_bidding(a_deal(seed, dealer), allow_loners=True)
                if out.contract.alone:
                    with self.subTest(dealer=dealer, seed=seed):
                        self.assertIn(out.contract.solve(), (LONE_MARCH, 1, -2))

    def test_the_line_says_alone(self):
        dealer, seed = LONE_MARCH_FOR_TEAM0
        out = b.solve_bidding(a_deal(seed, dealer), allow_loners=True)
        self.assertTrue(any("alone" in step for step in out.line))

    def test_ties_resolve_against_going_alone(self):
        """
        A loner worth no more than the same call four-handed is declined.

        Both are the same contract as far as the score goes, so choosing the
        loner would report a hand played alone for no reason -- the same
        nonsense as ordering up a hand you know will be euchred.
        """
        for dealer in range(4):
            for seed in range(15):
                d = a_deal(seed, dealer)
                out = b.solve_bidding(d, allow_loners=True)
                if not out.contract.alone:
                    continue
                seat = out.contract.caller
                same_call_four_handed = (
                    b.order_up(d, seat)[0]
                    if out.contract.bidding_round == b.ROUND_ONE
                    else b.name_suit(d, seat, out.contract.trump)[0])
                with self.subTest(dealer=dealer, seed=seed):
                    self.assertGreater(b.value_to(seat, out.value),
                                       b.value_to(seat, same_call_four_handed))

    def test_stick_the_dealer_still_works_with_loners(self):
        for dealer in range(4):
            for seed in range(4):
                out = b.solve_bidding(a_deal(seed, dealer),
                                      stick_the_dealer=True, allow_loners=True)
                with self.subTest(dealer=dealer, seed=seed):
                    self.assertFalse(out.passed_out)


class TestFirstBidOptions(unittest.TestCase):
    def test_keys(self):
        opts = b.first_bid_options(a_deal())
        self.assertEqual(set(opts), {"pass", "order", "order alone"})
        self.assertEqual(set(b.first_bid_options(a_deal(), allow_loners=False)),
                         {"pass", "order"})

    def test_order_matches_order_up(self):
        for seed in range(5):
            d = a_deal(seed, dealer=3)
            seat = d.first_bidder
            opts = b.first_bid_options(d)
            with self.subTest(seed=seed):
                self.assertEqual(opts["order"],
                                 b.value_to(seat, b.order_up(d, seat)[0]))
                self.assertEqual(
                    opts["order alone"],
                    b.value_to(seat, b.order_up(d, seat, alone=True)[0]))

    def test_agrees_with_first_bid_choice(self):
        for seed in range(5):
            d = a_deal(seed, dealer=3)
            ordered, passed = b.first_bid_choice(d)
            opts = b.first_bid_options(d, allow_loners=False)
            with self.subTest(seed=seed):
                self.assertEqual((opts["order"], opts["pass"]), (ordered, passed))

    def test_values_are_from_the_eldest_hands_own_side(self):
        """Seat 1 is on team 1, so its own-side values are the negated scale."""
        d = a_deal(seed=0, dealer=0)
        self.assertEqual(d.first_bidder, 1)
        opts = b.first_bid_options(d)
        self.assertEqual(opts["order"],
                         -b.order_up(d, 1)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
