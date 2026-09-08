"""
Unit tests for bidding.py.

Two things here are easy to get subtly wrong and are tested hardest:

  * the **scoring scale**. `definitive_winner` answers from the calling team's
    side, so comparing a call by seat 1 against a call by seat 2 means putting
    both on one scale first. A sign error there is invisible on any deal where
    the two teams happen to agree.
  * **who chooses the discard**. It is the dealer, not the caller. When the
    opponents order it up, the dealer is picking up a card for a contract they
    want to fail, and will pitch whatever hurts the caller most.

Perfect knowledge also does something worth knowing about: it essentially never
passes out. Over 1600 solved auctions no deal was thrown in, because somebody
can nearly always find a call that is at worst harmless to their own team.
"""
import os
import sys

# Also runnable directly (python tests/test_x.py), not just under
# `python -m unittest`, which gets the path from tests/__init__.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import bidding as b
import game
import rotation as r
from fast_search import definitive_winner

# Seat 0 holds the five best hearts; the other suits are scattered so no rival
# call is worth as much. Ordering up hearts is a march for team 0.
HEARTS_MONSTER = r.parse_hand(
    "JH AH KH QH TH  AD KS QC JD TC  TD AS KC QS 9D  JS AC KD QD 9C  9H  JC TS 9S")

# Seed 20 dealt by seat 2: seat 1 (team 1) orders up, so the dealer is picking
# up for the opposition. Pitching JD would hand the caller a march; the dealer
# pitches AH instead and holds them to +1.
HOSTILE_DISCARD_SEED, HOSTILE_DISCARD_DEALER = 20, 2


def a_deal(seed=0, dealer=0):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


def deals(n, dealer=0, start=0):
    return [a_deal(seed, dealer) for seed in range(start, start + n)]


class TestScoringScale(unittest.TestCase):
    def test_team_of(self):
        self.assertEqual([b.team_of(s) for s in range(4)], [0, 1, 0, 1])

    def test_net_to_team0_keeps_team_0_calls_as_they_are(self):
        for caller in (0, 2):
            for score in (-2, 1, 2):
                self.assertEqual(b.net_to_team0(score, caller), score)

    def test_net_to_team0_flips_team_1_calls(self):
        for caller in (1, 3):
            self.assertEqual(b.net_to_team0(2, caller), -2)
            self.assertEqual(b.net_to_team0(1, caller), -1)
            self.assertEqual(b.net_to_team0(-2, caller), 2)

    def test_a_euchre_is_worth_two_to_the_other_side(self):
        """Being euchred hands the opponents 2, whichever team called."""
        self.assertEqual(b.net_to_team0(-2, 0), -2)   # team 0 called, team 1 scores
        self.assertEqual(b.net_to_team0(-2, 1), 2)    # team 1 called, team 0 scores

    def test_value_to_is_its_own_inverse(self):
        for seat in range(4):
            for v in (-2, -1, 0, 1, 2):
                self.assertEqual(b.value_to(seat, b.value_to(seat, v)), v)

    def test_value_to_agrees_with_net_to_team0(self):
        for caller in range(4):
            for score in (-2, 1, 2):
                self.assertEqual(
                    b.value_to(caller, b.net_to_team0(score, caller)), score)


class TestPlayValue(unittest.TestCase):
    def test_play_begins_to_the_dealers_left(self):
        for dealer in range(4):
            d = a_deal(dealer=dealer)
            trump = d.up_card.suit
            want = definitive_winner(r.deal_to_engine(d.hands, trump),
                                     (dealer + 1) % 4, 0)
            self.assertEqual(b.play_value(d, trump, 0), want)

    def test_returns_a_real_score(self):
        for d in deals(10):
            self.assertIn(b.play_value(d, d.up_card.suit, 0), (-2, 1, 2))


class TestOrderUp(unittest.TestCase):
    def test_contract_shape(self):
        d = a_deal()
        value, contract = b.order_up(d, d.first_bidder)
        self.assertEqual(contract.trump, d.up_card.suit)
        self.assertEqual(contract.bidding_round, b.ROUND_ONE)
        self.assertTrue(contract.deal.picked_up)
        self.assertIsNotNone(contract.discard)
        contract.deal.check()

    def test_the_dealer_ends_with_five_cards(self):
        for d in deals(10):
            _, contract = b.order_up(d, d.first_bidder)
            self.assertEqual(len(contract.deal.hands[d.dealer]), 5)
            self.assertIn(contract.discard, contract.deal.buried)

    def test_value_matches_the_contract(self):
        for d in deals(10):
            value, contract = b.order_up(d, d.first_bidder)
            self.assertEqual(value, b.net_to_team0(contract.solve(), contract.caller))

    def test_the_dealer_discards_for_its_own_team(self):
        """Exhaustively: the chosen pitch is the dealer-team-optimal one."""
        for d in deals(8):
            for caller in range(4):
                value, _ = b.order_up(d, caller)
                everything = [
                    b.net_to_team0(
                        b.play_value(d.pick_up(card), d.up_card.suit, caller), caller)
                    for card in list(d.hands[d.dealer]) + [d.up_card]
                ]
                want = (max(everything) if b.team_of(d.dealer) == 0
                        else min(everything))
                self.assertEqual(value, want,
                                 "dealer %d did not pitch for its own team"
                                 % d.dealer)

    def test_the_dealer_will_hold_an_opposing_caller_down(self):
        """
        The headline case: the dealer is forced to pick up for the opposition
        and pitches to hurt them. A naive implementation that lets the *caller*
        choose the discard reports a march here instead of a bare make.
        """
        d = a_deal(HOSTILE_DISCARD_SEED, HOSTILE_DISCARD_DEALER)
        caller = 1
        self.assertNotEqual(b.team_of(caller), b.team_of(d.dealer),
                            "fixture should pit the dealer against the caller")

        value, contract = b.order_up(d, caller)
        got = b.value_to(caller, value)

        best_if_caller_chose = max(
            b.value_to(caller,
                       b.net_to_team0(
                           b.play_value(d.pick_up(card), d.up_card.suit, caller),
                           caller))
            for card in list(d.hands[d.dealer]) + [d.up_card])

        self.assertEqual(got, 1)
        self.assertEqual(best_if_caller_chose, 2)
        self.assertLess(got, best_if_caller_chose,
                        "the dealer should have denied the caller a march")


class TestNameSuit(unittest.TestCase):
    def test_rejects_the_turned_suit(self):
        d = a_deal()
        with self.assertRaises(ValueError):
            b.name_suit(d, d.first_bidder, d.up_card.suit)

    def test_contract_shape(self):
        d = a_deal()
        trump = next(s for s in r.SUITS if s != d.up_card.suit)
        value, contract = b.name_suit(d, d.first_bidder, trump)
        self.assertEqual(contract.bidding_round, b.ROUND_TWO)
        self.assertFalse(contract.deal.picked_up)
        self.assertIsNone(contract.discard)
        self.assertEqual(value, b.net_to_team0(contract.solve(), contract.caller))


class TestSolveBidding(unittest.TestCase):
    def test_value_matches_the_surviving_contract(self):
        for dealer in range(4):
            for d in deals(5, dealer=dealer):
                out = b.solve_bidding(d)
                if out.passed_out:
                    self.assertEqual(out.value, 0)
                    continue
                self.assertEqual(
                    out.value,
                    b.net_to_team0(out.contract.solve(), out.contract.caller))

    def test_contract_deal_is_sound(self):
        for d in deals(12):
            out = b.solve_bidding(d)
            if out.contract:
                out.contract.deal.check()

    def test_round_one_calls_take_the_turned_suit(self):
        for d in deals(12):
            out = b.solve_bidding(d)
            if out.contract and out.contract.bidding_round == b.ROUND_ONE:
                self.assertEqual(out.contract.trump, d.up_card.suit)
                self.assertTrue(out.contract.deal.picked_up)

    def test_round_two_calls_never_take_the_turned_suit(self):
        for dealer in range(4):
            for d in deals(6, dealer=dealer):
                out = b.solve_bidding(d)
                if out.contract and out.contract.bidding_round == b.ROUND_TWO:
                    self.assertNotEqual(out.contract.trump, d.up_card.suit)
                    self.assertFalse(out.contract.deal.picked_up)

    def test_is_deterministic(self):
        d = a_deal(5)
        first = b.solve_bidding(d)
        for _ in range(3):
            again = b.solve_bidding(d)
            self.assertEqual(again.value, first.value)
            self.assertEqual(str(again.contract), str(first.contract))

    def test_rejects_a_deal_already_picked_up(self):
        d = a_deal()
        after = d.pick_up(discard=d.hands[d.dealer][0])
        with self.assertRaises(ValueError):
            b.solve_bidding(after)

    def test_a_monster_hand_is_ordered_up_for_a_march(self):
        d = game.deal_from_order(HEARTS_MONSTER, dealer=3)
        out = b.solve_bidding(d)
        self.assertEqual(out.value, 2)
        self.assertEqual(out.contract.bidding_round, b.ROUND_ONE)
        self.assertEqual(out.contract.trump, d.up_card.suit)
        self.assertEqual(b.team_of(out.contract.caller), 0)

    def test_ordering_that_monster_is_worth_a_march_to_the_holder(self):
        d = game.deal_from_order(HEARTS_MONSTER, dealer=3)
        self.assertEqual(b.order_up(d, 0)[0], 2)

    def test_line_records_every_bid(self):
        d = a_deal()
        out = b.solve_bidding(d)
        self.assertTrue(out.line)
        self.assertTrue(all(isinstance(step, str) for step in out.line))


class TestFirstBidChoice(unittest.TestCase):
    def test_values_are_from_the_eldest_hands_own_side(self):
        for d in deals(10):
            ordered, _ = b.first_bid_choice(d)
            raw, _ = b.order_up(d, d.first_bidder)
            self.assertEqual(ordered, b.value_to(d.first_bidder, raw))

    def test_the_eldest_orders_only_when_it_is_strictly_better(self):
        """Ties resolve to passing, so an equal-value call is declined."""
        for dealer in range(4):
            for d in deals(5, dealer=dealer):
                ordered, passed = b.first_bid_choice(d)
                out = b.solve_bidding(d)
                eldest_ordered = (out.contract is not None
                                  and out.contract.bidding_round == b.ROUND_ONE
                                  and out.contract.caller == d.first_bidder)
                if eldest_ordered:
                    self.assertGreater(ordered, passed)
                else:
                    self.assertLessEqual(ordered, passed)


class TestStickTheDealer(unittest.TestCase):
    def test_never_passes_out(self):
        for dealer in range(4):
            for d in deals(5, dealer=dealer):
                out = b.solve_bidding(d, stick_the_dealer=True)
                self.assertFalse(out.passed_out,
                                 "the dealer may not pass under stick-the-dealer")

    def test_is_never_better_for_the_dealers_team_than_a_free_pass(self):
        """
        Being forced to call can only cost the dealer's team, never help.

        Not vacuous: the rule changes the result on about 4% of deals (21 of
        480 measured). It bites even though perfect knowledge rarely passes
        out, because removing the dealer's pass changes the value of that node
        and the earlier seats bid differently against the threat.
        """
        for dealer in range(4):
            for d in deals(4, dealer=dealer):
                free = b.solve_bidding(d, stick_the_dealer=False)
                stuck = b.solve_bidding(d, stick_the_dealer=True)
                self.assertLessEqual(b.value_to(dealer, stuck.value),
                                     b.value_to(dealer, free.value))


class TestOutcome(unittest.TestCase):
    def test_passed_out_outcome(self):
        out = b.Outcome(None, 0)
        self.assertTrue(out.passed_out)
        self.assertEqual(out.value, 0)
        self.assertIn("passed out", str(out))

    def test_contract_string_mentions_the_caller_and_suit(self):
        d = a_deal()
        _, contract = b.order_up(d, d.first_bidder)
        text = str(contract)
        self.assertIn("seat %d" % d.first_bidder, text)
        self.assertIn(r.suit_name(d.up_card.suit), text)
        self.assertIn("pitched", text)


if __name__ == "__main__":
    unittest.main()
