"""
The two functions a front end calls.

`api.py` computes almost nothing itself -- `solve` assembles `bidding` and
`fast_search`, `evaluate` drives `hand_ev.run_fast` three times -- so this is
not about whether the Euchre is right. `test_bitcore.py`, `test_fastsim.py`
and `test_hand_ev.py` own that. What is tested here is the four things a thin
layer can still get wrong on its own:

**The line it reports is a real line.** `solve` decodes `solve_line`'s planes
back into cards, through two conversions and a private-until-now helper. A
decode that is subtly wrong produces a plausible-looking hand of cards that
nobody could have played, so the line is replayed against the rules -- every
card came from the hand that played it, follow-suit was respected, and each
trick went to the seat the rules say.

**The numbers agree with themselves.** An `ActionEV`'s mean has to be the mean
of the outcomes its own trick breakdown describes. That catches a summary
built from the wrong subset, which is the easy mistake here and is invisible
on any hand where the subsets happen to agree.

**The population is the one that was promised.** `evaluate` reports over the
deals the auction reached the seat on, not all of them. The two differ, and
reversing them reverses the advice about loners -- see `notes/front_end.md`.

**It survives contact with a web layer.** `as_dict()` has to be JSON, all the
way down, with no `Card` namedtuples left in it.
"""
import json
import random
import unittest

import api
import bidding as b
import game
import rotation as r
import table as t
from fast_search import decode_card, encode_hands

# Small enough to be a unit test. Nothing here depends on a PIMC player
# choosing *well*, only on the bookkeeping around it.
DEALS = 40
SIMS = 4

FOUR_HANDS = ("AS KD JS JH AD", "TC KH AC KS TH",
              "QS TS 9S JD 9D", "9C AH JC KC QH")
UP = "9H"


def a_deal(seed, dealer=3):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


class TestSolve(unittest.TestCase):
    """The exact answer, and the line of play under it."""

    def test_auction_matches_solve_bidding(self):
        for seed in range(12):
            deal = a_deal(seed)
            want = b.solve_bidding(deal, allow_loners=True)
            got = api.solve(deal.hands, deal.up_card, dealer=deal.dealer)
            self.assertEqual(got.passed_out, want.passed_out, seed)
            self.assertEqual(got.value, want.value, seed)
            self.assertEqual(tuple(got.auction), tuple(want.line), seed)
            if not want.passed_out:
                self.assertEqual(got.caller, want.contract.caller, seed)
                self.assertEqual(got.trump, want.contract.trump, seed)
                self.assertEqual(got.alone, want.contract.alone, seed)
                self.assertEqual(got.discard, want.contract.discard, seed)

    def test_the_line_is_one_that_could_be_played(self):
        """
        Replay it against the rules rather than trusting the decode.

        Every card has to come out of the hand that played it, follow-suit has
        to hold, and each trick has to go to the seat the rules give it. A
        decode that lands on the wrong card usually still looks like a card.
        """
        for seed in range(12):
            deal = a_deal(seed)
            got = api.solve(deal.hands, deal.up_card, dealer=deal.dealer)
            if got.passed_out:
                continue

            settled = deal
            if got.discard is not None:
                settled = deal.pick_up(discard=got.discard)
            held = [list(h) for h in settled.hands]
            sitting = (got.caller + 2) % game.PLAYERS if got.alone else None
            width = 3 if got.alone else 4

            leader = settled.first_bidder
            if leader == sitting:
                leader = t.next_seat(leader, sitting)

            for trick in got.tricks:
                self.assertEqual(len(trick.plays), width, seed)
                self.assertEqual(trick.plays[0][0], leader,
                                 "trick %d led by the wrong seat, deal %d"
                                 % (trick.number, seed))
                led = trick.plays[0][1]
                seat = leader
                for played_by, card in trick.plays:
                    self.assertEqual(played_by, seat, seed)
                    self.assertIn(card, held[seat],
                                  "seat %d played %s it does not hold"
                                  % (seat, r.card_name(card)))
                    legal = t.legal_cards(held[seat],
                                          None if card is led else led,
                                          got.trump)
                    self.assertIn(card, legal,
                                  "seat %d played %s with %s led"
                                  % (seat, r.card_name(card),
                                     r.card_name(led)))
                    held[seat].remove(card)
                    seat = t.next_seat(seat, sitting)
                self.assertEqual(trick.winner,
                                 t.trick_winner(list(trick.plays), got.trump),
                                 "trick %d of deal %d" % (trick.number, seed))
                leader = trick.winner

            for seat in range(game.PLAYERS):
                self.assertEqual(held[seat], [] if seat != sitting
                                 else list(settled.hands[seat]), seed)

    def test_tricks_and_score_agree(self):
        for seed in range(12):
            deal = a_deal(seed)
            got = api.solve(deal.hands, deal.up_card, dealer=deal.dealer)
            if got.passed_out:
                self.assertEqual(got.tricks, ())
                continue
            won = sum(1 for x in got.tricks
                      if x.winner % 2 == got.caller % 2)
            self.assertEqual(got.caller_tricks, won, seed)
            if won == game.HAND_SIZE:
                want = 4 if got.alone else 2
            elif won >= 3:
                want = 1
            else:
                want = -2
            self.assertEqual(got.caller_score, want, seed)
            self.assertEqual(got.value,
                             b.net_to_team0(got.caller_score, got.caller), seed)

    def test_loners_play_three_card_tricks(self):
        """A lone hand somewhere in the sweep, played out three-handed."""
        seen = 0
        for seed in range(60):
            deal = a_deal(seed)
            got = api.solve(deal.hands, deal.up_card, dealer=deal.dealer)
            if got.passed_out or not got.alone:
                continue
            seen += 1
            sitting = (got.caller + 2) % game.PLAYERS
            for trick in got.tricks:
                self.assertEqual(len(trick.plays), 3)
                for seat, _ in trick.plays:
                    self.assertNotEqual(seat, sitting)
            self.assertIn(got.caller_score, (-2, 1, 4))
        self.assertGreater(seen, 0, "no loner turned up to test")

    def test_options_are_bidding_s_own(self):
        deal = a_deal(3)
        for seat in range(game.PLAYERS):
            got = api.solve(deal.hands, deal.up_card, dealer=deal.dealer,
                            seat=seat)
            self.assertEqual(got.options, b.bid_options(deal, seat, True))
            self.assertEqual(got.options_seat, seat)
        self.assertIsNone(api.solve(deal.hands, deal.up_card,
                                    dealer=deal.dealer).options)

    def test_it_does_not_sample(self):
        deal = a_deal(5)
        first = api.solve(deal.hands, deal.up_card, dealer=deal.dealer, seat=1)
        for _ in range(3):
            self.assertEqual(api.solve(deal.hands, deal.up_card,
                                       dealer=deal.dealer, seat=1), first)

    def test_it_takes_strings_or_cards(self):
        from_text = api.solve(FOUR_HANDS, UP, dealer=3)
        from_cards = api.solve([r.parse_hand(h) for h in FOUR_HANDS],
                               r.parse_card(UP), dealer=3)
        self.assertEqual(from_text, from_cards)


class TestEvaluate(unittest.TestCase):
    """The sampled answer, and whether its summary describes its own deals."""

    def _one(self, **kw):
        kw.setdefault("deals", DEALS)
        kw.setdefault("play_sims", SIMS)
        return api.evaluate("TH AS AD KD JD", "9H", seat=2, dealer=0, **kw)

    def test_every_action_is_priced(self):
        got = self._one()
        self.assertEqual([a.action for a in got.actions], list(api.ACTIONS))
        for action in api.ACTIONS:
            self.assertEqual(got[action].action, action)
        self.assertIn(got.best().action, api.ACTIONS)

    def test_a_subset_of_actions(self):
        got = self._one(actions=(api.ORDER, api.PASS))
        self.assertEqual([a.action for a in got.actions],
                         [api.ORDER, api.PASS])
        with self.assertRaises(KeyError):
            got[api.ORDER_ALONE]

    def test_the_mean_is_the_mean_of_its_own_tricks(self):
        """
        Where this seat's team holds every contract, the trick breakdown
        determines the EV exactly. That pins the summary to the right subset.
        """
        got = self._one()
        for action in (api.ORDER, api.ORDER_ALONE):
            cell = got[action]
            self.assertEqual(cell.tricks["contracts"], cell.reached,
                             "%s: the seat should hold every contract" % action)
            march = 4 if action == api.ORDER_ALONE else 2
            total = (cell.tricks["march"] * march + cell.tricks["made"]
                     - 2 * cell.tricks["euchred"])
            self.assertAlmostEqual(cell.ev, total / cell.reached, places=9,
                                   msg=action)

    def test_the_population_is_the_deals_that_reached_the_seat(self):
        """
        From the eldest seat the auction always reaches you; from a later one
        it need not, and the reported mean is over the deals it did.
        """
        eldest = api.evaluate("TH AS AD KD JD", "9H", seat=0, dealer=3,
                              deals=DEALS, play_sims=SIMS,
                              actions=(api.ORDER,))[api.ORDER]
        self.assertEqual(eldest.reached, eldest.deals)
        self.assertEqual(eldest.reach_rate, 1.0)

        later = self._one(actions=(api.ORDER,))[api.ORDER]
        self.assertLess(later.reached, later.deals)
        self.assertGreater(later.reached, 0)
        self.assertEqual(sum(later.roles.values()), later.reached)

    def test_the_same_seed_gives_the_same_answer(self):
        first = self._one(actions=(api.ORDER,))[api.ORDER]
        again = self._one(actions=(api.ORDER,))[api.ORDER]
        self.assertEqual(first, again)

    def test_the_thread_count_does_not_change_it(self):
        one = self._one(actions=(api.ORDER,), workers=1)[api.ORDER]
        many = self._one(actions=(api.ORDER,), workers=4)[api.ORDER]
        self.assertEqual(one, many)

    def test_more_deals_narrows_the_interval(self):
        small = self._one(actions=(api.ORDER,), deals=40)[api.ORDER]
        big = self._one(actions=(api.ORDER,), deals=400)[api.ORDER]
        self.assertLess(big.interval, small.interval)


class TestRoundTwo(unittest.TestCase):
    """The up-card turned down: what each suit is worth when it comes back."""

    def _one(self, **kw):
        kw.setdefault("deals", 120)
        kw.setdefault("play_sims", SIMS)
        return api.evaluate("TH AS AD KD JD", "9C", seat=2, dealer=0,
                            bidding_round=api.ROUND_TWO, **kw)

    def test_the_turned_down_suit_is_not_on_offer(self):
        self.assertEqual(api.round_two_actions("9C"),
                         ("spades", "spades alone", "hearts", "hearts alone",
                          "diamonds", "diamonds alone", api.PASS))
        self.assertEqual(api.round_two_actions("9C", allow_loners=False),
                         ("spades", "hearts", "diamonds", api.PASS))

    def test_every_action_sees_the_same_deals(self):
        got = self._one(actions=("diamonds", "hearts alone", api.PASS))
        self.assertEqual(got.bidding_round, api.ROUND_TWO)
        reached = {a.reached for a in got.actions}
        self.assertEqual(len(reached), 1, "actions saw different deals")
        self.assertGreater(reached.pop(), 0)

    def test_a_named_suit_is_always_the_seats_contract(self):
        """
        On every counted deal the seat named the suit, so its team holds the
        contract and the trick breakdown fixes the mean exactly.
        """
        got = self._one(actions=("diamonds", "diamonds alone"))
        for cell, march in ((got["diamonds"], 2), (got["diamonds alone"], 4)):
            self.assertLess(cell.reached, cell.deals)
            self.assertEqual(cell.roles, {"you called": cell.reached})
            self.assertEqual(cell.tricks["contracts"], cell.reached)
            total = (cell.tricks["march"] * march + cell.tricks["made"]
                     - 2 * cell.tricks["euchred"])
            self.assertAlmostEqual(cell.ev, total / cell.reached, places=9)

    def test_passing_again_never_makes_the_seat_the_caller(self):
        cell = self._one(actions=(api.PASS,))[api.PASS]
        self.assertNotIn("you called", cell.roles)
        self.assertEqual(sum(cell.roles.values()), cell.reached)

    def test_a_threshold_keeps_dealing_until_it_is_met(self):
        got = self._one(actions=("diamonds", api.PASS), deals=20,
                        min_reached=30)
        self.assertGreater(got.deals, 20)
        self.assertEqual(got.min_reached, 30)
        for cell in got.actions:
            self.assertGreaterEqual(cell.reached, 30)
            self.assertEqual(cell.deals, got.deals)
        # One batch is the same sweep as many: the first N deals either way.
        flat = self._one(actions=("diamonds",), deals=got.deals)["diamonds"]
        self.assertEqual(flat, got["diamonds"])

    def test_later_actions_replay_only_the_deals_that_came_back(self):
        """
        After the first action, round two plays only the deals that came
        back -- `hand_ev.run_fast_listed`. Priced second, an action must
        come out exactly as it does priced first over the whole sweep.
        """
        first = self._one(actions=("diamonds",))["diamonds"]
        second = self._one(actions=(api.PASS, "diamonds"))["diamonds"]
        self.assertEqual(first, second)

    def test_a_threshold_stops_at_the_cap(self):
        got = self._one(actions=("spades",), deals=8, min_reached=10 ** 6,
                        max_deals=50)
        self.assertEqual(got.deals, 50)
        self.assertLess(got["spades"].reached, 10 ** 6)

    def test_it_serialises_with_its_round(self):
        got = self._one(actions=("spades",), deals=20).as_dict()
        self.assertEqual(got["round"], 2)
        self.assertEqual(json.loads(json.dumps(got)), got)

    def test_it_rejects_what_round_two_cannot_do(self):
        for kw, why in ((dict(actions=("clubs",)), "the turned-down suit"),
                        (dict(actions=("order",)), "a round-one action"),
                        (dict(actions=("hearts alone",), allow_loners=False),
                         "a loner with loners off"),
                        (dict(bidding_round=3), "no third round")):
            with self.subTest(why=why), self.assertRaises(ValueError):
                kw.setdefault("bidding_round", api.ROUND_TWO)
                api.evaluate("TH AS AD KD JD", "9C", seat=2, dealer=0,
                             deals=4, play_sims=SIMS, **kw)


class TestValidation(unittest.TestCase):
    """What a web layer will send on its first day."""

    def test_evaluate_rejects_bad_input(self):
        for kw, why in (
                (dict(hand="TH AS AD KD", up_card="9H"), "four cards"),
                (dict(hand="TH TH AD KD JD", up_card="9H"), "a repeat"),
                (dict(hand="TH AS AD KD JD", up_card="TH"), "up-card in hand"),
                (dict(hand="TH AS AD KD JD", up_card="9H", seat=4), "seat"),
                (dict(hand="TH AS AD KD JD", up_card="9H", dealer=-1), "dealer"),
                (dict(hand="TH AS AD KD JD", up_card="9H", deals=0), "deals"),
                (dict(hand="TH AS AD KD JD", up_card="9H",
                      actions=("shove",)), "action"),
        ):
            with self.assertRaises(ValueError, msg=why):
                api.evaluate(**kw)

    def test_solve_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            api.solve(FOUR_HANDS[:3], UP)
        with self.assertRaises(ValueError):
            api.solve(("AS KD JS JH", ) + FOUR_HANDS[1:], UP)
        with self.assertRaises(ValueError):
            api.solve(FOUR_HANDS, "AS")            # already in seat 0's hand
        with self.assertRaises(ValueError):
            api.solve(FOUR_HANDS, UP, dealer=9)


class TestSerialisation(unittest.TestCase):
    """`as_dict()` is JSON all the way down, or the web layer finds out."""

    def _plain(self, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.assertIsInstance(k, str)
                self._plain(v)
        elif isinstance(value, list):
            for v in value:
                self._plain(v)
        else:
            self.assertIsInstance(value, (str, int, float, bool, type(None)),
                                  "%r is not JSON" % (value,))

    def test_evaluation_round_trips(self):
        got = api.evaluate("TH AS AD KD JD", "9H", seat=2, dealer=0,
                           deals=DEALS, play_sims=SIMS).as_dict()
        self._plain(got)
        self.assertEqual(json.loads(json.dumps(got)), got)
        self.assertEqual(got["hand"], ["TH", "AS", "AD", "KD", "JD"])
        self.assertEqual(got["up_card"], "9H")

    def test_solution_round_trips(self):
        got = api.solve(FOUR_HANDS, UP, dealer=3, seat=0).as_dict()
        self._plain(got)
        self.assertEqual(json.loads(json.dumps(got)), got)
        self.assertEqual(got["trump"], "hearts")
        self.assertEqual(len(got["tricks"]), game.HAND_SIZE)

    def test_a_passed_out_solution_still_serialises(self):
        """Rare -- 0 of 1600 measured -- but it must not be a crash."""
        made = api.Solution(
            hands=tuple(tuple(r.parse_hand(h)) for h in FOUR_HANDS),
            up_card=r.parse_card(UP), dealer=3, auction=("all pass",),
            caller=None, trump=None, alone=False, discard=None, value=0,
            caller_score=0, caller_tricks=0, tricks=())
        got = made.as_dict()
        self._plain(got)
        self.assertTrue(got["passed_out"])
        self.assertIsNone(got["trump"])


class TestDecodeCard(unittest.TestCase):
    """The helper `solve_line`'s output needed, now that it is public."""

    def test_it_inverts_encode_hands(self):
        import numpy as np
        for trump in r.SUITS:
            deck = r.full_deck()
            vectors = np.array([r.card_to_engine(c, trump) for c in deck],
                               dtype=np.int64).reshape(1, -1, 2)
            suits, strengths = encode_hands(vectors)
            for i, card in enumerate(deck):
                back = decode_card(suits[0, i], strengths[0, i])
                self.assertEqual(r.card_from_engine(back, trump), card,
                                 "%s under %s" % (r.card_name(card),
                                                  r.suit_name(trump)))


if __name__ == "__main__":
    unittest.main()
