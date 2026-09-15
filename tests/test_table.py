"""
Unit tests for table.py and players.py.

Two things are checked, in very different ways.

**The referee** is checked against the rules directly. Its trick-winner rule is
a second implementation of `fast_search._resolve`, so it is run against that and
against `euchre_testkit.winner_of` -- the no-shared-code oracle -- over every
trick a random table produces. Its legality rule is checked by seating
`RandomPlayer` everywhere, which tries every legal line including the silly
ones, and asserting card conservation over the whole hand.

**The players** are checked against the baseline they generalise. A table of
four God Mode players is `bidding.solve_bidding` followed by
`fast_search.definitive_winner`, one decision at a time instead of all at once,
so it must give the same contract and score on every deal. That is the test
that matters most here: referee and God Mode player between them re-derive the
existing answer through entirely new code, and `PIMCPlayer` rides on it.

The PIMC sim cannot be tested for *quality* -- there is no ground truth for
"what should an honest player bid", which is why the project exists. What is
tested is that it is honest: it never touches `turn.deal`, it only returns
legal options, and a full sweep played by it stays structurally sound.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import numpy as np

import bidding as b
import euchre_testkit as kit
import fast_search as fs
import game
import observation as ob
import players
import rotation as r
import table as t


def a_deal(seed=0, dealer=0):
    return game.deal_random(rng=random.Random(seed), dealer=dealer)


def god_mode_table():
    return [players.GodModePlayer() for _ in range(4)]


def random_table(seed=0):
    return [players.RandomPlayer(random.Random(seed * 10 + s))
            for s in range(4)]


def pimc_table(seed=0, samples=4, bid_samples=2, **kw):
    return [players.PIMCPlayer(samples=samples, bid_samples=bid_samples,
                               rng=random.Random(seed * 10 + s), **kw)
            for s in range(4)]


class TestCardOrder(unittest.TestCase):
    """The referee's reading of a card must match the solver's."""

    def test_trump_outranks_everything_plain(self):
        for trump in r.SUITS:
            for rank in r.RANKS:
                plain = [s for s in r.SUITS
                         if s != trump and s != r.same_colour(trump)]
                trump_card = t.card_order(r.Card(trump, rank), trump)
                self.assertEqual(trump_card[0], 1)
                for suit in plain:
                    self.assertNotEqual(
                        t.card_order(r.Card(suit, rank), trump)[0], 1)

    def test_the_bowers_are_the_top_two_trumps(self):
        for trump in r.SUITS:
            right = t.card_order(r.Card(trump, r.JACK), trump)
            left = t.card_order(r.Card(r.same_colour(trump), r.JACK), trump)
            ace = t.card_order(r.Card(trump, r.ACE), trump)
            self.assertEqual(right[0], 1)
            self.assertEqual(left[0], 1)
            self.assertGreater(right[1], left[1])
            self.assertGreater(left[1], ace[1])


class TestTrickWinner(unittest.TestCase):
    """
    Checked against two independent readings of the same rule.

    `fast_search._resolve` is the rule as the search applies it;
    `euchre_testkit.winner_of` is the rule written from scratch for the tests.
    Agreement with both, over every trick of a lot of random hands, is what
    makes a third implementation safe to have.
    """

    def resolve_with_solver(self, trick, trump):
        width = len(trick)
        t_suit = np.zeros((1, width), dtype=np.int64)
        t_str = np.zeros((1, width), dtype=np.int64)
        t_player = np.zeros((1, width), dtype=np.int64)
        for k, (seat, card) in enumerate(trick):
            suit, strength = kit.suit_strength(r.card_to_engine(card, trump))
            t_suit[0, k] = suit
            t_str[0, k] = strength
            t_player[0, k] = seat
        return int(fs._resolve(t_suit, t_str, t_player, 0, width))

    def resolve_with_oracle(self, trick, trump):
        played = [kit.suit_strength(r.card_to_engine(card, trump)) + (seat,)
                  for seat, card in trick]
        return kit.winner_of(played)

    def test_against_both_oracles_over_random_play(self):
        checked = 0
        for seed in range(8):
            deal = a_deal(seed, dealer=seed % 4)
            result = t.play_deal(deal, random_table(seed), allow_loners=True)
            if result.passed_out:
                continue
            trump = result.contract.trump
            width = 3 if result.contract.alone else 4
            for i in range(0, len(result.plays), width):
                trick = result.plays[i:i + width]
                want = t.trick_winner(trick, trump)
                self.assertEqual(want, self.resolve_with_solver(trick, trump))
                self.assertEqual(want, self.resolve_with_oracle(trick, trump))
                checked += 1
        self.assertGreater(checked, 20)

    def test_the_left_bower_wins_as_trump(self):
        C = r.parse_card
        # Diamonds are trump, so the jack of hearts is the left bower and beats
        # the ace of trump. Reading the printed suit would score it as a heart
        # discard and award the trick to the ace.
        trick = ((0, C("AD")), (1, C("JH")), (2, C("9D")), (3, C("KD")))
        self.assertEqual(t.trick_winner(trick, r.DIAMONDS), 1)

    def test_a_ruff_beats_the_led_suit(self):
        C = r.parse_card
        trick = ((2, C("AH")), (3, C("KH")), (0, C("9S")), (1, C("QH")))
        self.assertEqual(t.trick_winner(trick, r.SPADES), 0)

    def test_highest_of_the_led_suit_when_nobody_trumps(self):
        C = r.parse_card
        trick = ((1, C("9H")), (2, C("KH")), (3, C("TH")), (0, C("QH")))
        self.assertEqual(t.trick_winner(trick, r.SPADES), 2)


class TestLegalCards(unittest.TestCase):
    def test_anything_goes_on_the_lead(self):
        hand = r.parse_hand("AH KH 9S TD")
        self.assertEqual(t.legal_cards(hand, None, r.SPADES), tuple(hand))

    def test_follow_suit_when_you_can(self):
        hand = r.parse_hand("AH KH 9S TD")
        legal = t.legal_cards(hand, r.parse_card("QH"), r.SPADES)
        self.assertEqual(set(legal), set(r.parse_hand("AH KH")))

    def test_anything_goes_when_void(self):
        hand = r.parse_hand("AH KH 9S TD")
        legal = t.legal_cards(hand, r.parse_card("QC"), r.SPADES)
        self.assertEqual(set(legal), set(hand))

    def test_the_left_bower_must_follow_trump(self):
        # Spades trump: the jack of clubs is trump, so a trump lead forces it
        # and a club lead does not.
        hand = r.parse_hand("JC AH KH")
        self.assertEqual(t.legal_cards(hand, r.parse_card("9S"), r.SPADES),
                         (r.parse_card("JC"),))
        self.assertEqual(set(t.legal_cards(hand, r.parse_card("9C"), r.SPADES)),
                         set(hand))


class TestGodModeTableMatchesTheBaseline(unittest.TestCase):
    """
    Four God Mode players are solve_bidding, one decision at a time.

    If these ever disagree, either the referee walks the auction differently
    from the way `bidding.py` searches it, or the player breaks ties on a
    different rule. Both are silent failures everywhere else.
    """

    def check(self, deal, stick=False, loners=False):
        want = b.solve_bidding(deal, stick_the_dealer=stick,
                               allow_loners=loners)
        got = t.play_deal(deal, god_mode_table(), stick_the_dealer=stick,
                          allow_loners=loners)
        self.assertEqual(got.value, want.value)
        self.assertEqual(got.passed_out, want.passed_out)
        if not want.passed_out:
            self.assertEqual(got.contract.trump, want.contract.trump)
            self.assertEqual(got.contract.caller, want.contract.caller)
            self.assertEqual(got.contract.alone, want.contract.alone)
            self.assertEqual(got.contract.discard, want.contract.discard)

    def test_plain_auctions(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self.check(a_deal(seed, dealer=seed % 4))

    def test_with_loners(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self.check(a_deal(seed, dealer=seed % 4), loners=True)

    def test_with_stick_the_dealer(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self.check(a_deal(seed, dealer=seed % 4), stick=True)

    def test_the_play_reaches_the_god_mode_score(self):
        # The auction picks a contract; playing it out in God Mode has to end
        # on the score the solver gives that contract.
        for seed in range(6):
            deal = a_deal(seed, dealer=seed % 4)
            got = t.play_deal(deal, god_mode_table(), allow_loners=True)
            if got.passed_out:
                continue
            want = b.play_value(got.contract.deal, got.contract.trump,
                                got.contract.caller, got.contract.alone)
            with self.subTest(seed=seed):
                self.assertEqual(got.caller_score, want)


class TestPlayIsLegal(unittest.TestCase):
    """
    Structural soundness of a played-out hand, whoever is playing it.

    RandomPlayer is the useful one here: it reaches positions no thoughtful
    player would, which is exactly where a referee bug lives.
    """

    def check(self, result, deal):
        if result.passed_out:
            self.assertEqual(result.value, 0)
            self.assertEqual(result.plays, ())
            return

        contract = result.contract
        width = 3 if contract.alone else 4
        self.assertEqual(len(result.plays), 5 * width)
        self.assertEqual(len(result.winners), 5)

        # Every card played came from the hand that played it, once.
        played = {}
        for seat, card in result.plays:
            played.setdefault(seat, []).append(card)
        for seat, cards in played.items():
            self.assertEqual(len(cards), len(set(cards)))
            self.assertEqual(len(cards), 5)
            self.assertTrue(set(cards) <= set(contract.deal.hands[seat]))

        if contract.alone:
            self.assertNotIn(contract.sitting, played)
            self.assertEqual(len(played), 3)
        else:
            self.assertEqual(len(played), 4)

        # Following suit was respected on every trick.
        held = {s: list(contract.deal.hands[s]) for s in range(4)}
        for i in range(0, len(result.plays), width):
            trick = result.plays[i:i + width]
            led = trick[0][1]
            for seat, card in trick:
                legal = t.legal_cards(held[seat],
                                      None if seat == trick[0][0] else led,
                                      contract.trump)
                self.assertIn(card, legal,
                              "seat %d played %s illegally"
                              % (seat, r.card_name(card)))
                held[seat].remove(card)

        # Tricks, winners and the score all tell the same story.
        caller_tricks = sum(1 for w in result.winners
                            if w % 2 == contract.caller % 2)
        self.assertEqual(caller_tricks, result.caller_tricks)
        self.assertEqual(result.caller_score,
                         int(fs._final(caller_tricks, contract.alone)))
        self.assertEqual(result.value,
                         b.net_to_team0(result.caller_score, contract.caller))

    def test_random_players(self):
        for seed in range(20):
            deal = a_deal(seed, dealer=seed % 4)
            with self.subTest(seed=seed):
                self.check(t.play_deal(deal, random_table(seed),
                                       allow_loners=True), deal)

    def test_god_mode_players(self):
        for seed in range(6):
            deal = a_deal(seed, dealer=seed % 4)
            with self.subTest(seed=seed):
                self.check(t.play_deal(deal, god_mode_table(),
                                       allow_loners=True), deal)

    def test_a_lone_hand_scores_one_of_three_values(self):
        # -2, +1 or +4, and never +2: a march alone pays double and nothing
        # else about the scoring changes.
        seen = set()
        for seed in range(30):
            deal = a_deal(seed, dealer=seed % 4)
            result = t.play_deal(deal, random_table(seed), allow_loners=True)
            if result.passed_out or not result.contract.alone:
                continue
            seen.add(result.caller_score)
        self.assertTrue(seen <= {-2, 1, 4}, seen)


class TestReferee(unittest.TestCase):
    def test_an_illegal_card_is_refused(self):
        class Cheat:
            def bid(self, turn):
                for option in turn.options:
                    if option.action != t.PASS:
                        return option
                return turn.options[0]

            def discard(self, turn):
                return turn.options[0]

            def play(self, turn):
                held = turn.observation.hand
                outside = [c for c in held if c not in turn.legal]
                return outside[0] if outside else turn.legal[0]

        deal = a_deal(0, dealer=0)
        table = [Cheat() for _ in range(4)]
        with self.assertRaises(ValueError):
            for seed in range(12):
                t.play_deal(a_deal(seed, dealer=seed % 4), table)

    def test_a_bid_that_was_not_offered_is_refused(self):
        class Wild:
            def bid(self, turn):
                return t.Bid(t.NAME, turn.deal.up_card.suit, False)

            def discard(self, turn):
                return turn.options[0]

            def play(self, turn):
                return turn.legal[0]

        with self.assertRaises(ValueError):
            t.play_deal(a_deal(0), [Wild() for _ in range(4)])

    def test_a_table_needs_four_players(self):
        with self.assertRaises(ValueError):
            t.play_deal(a_deal(0), god_mode_table()[:3])


class TestPIMC(unittest.TestCase):
    """
    PIMC is checked for honesty and soundness, not for strength.

    There is no oracle for "what should an honest player bid" -- producing one
    is the point of the project -- so quality is left to the sweep in
    `pimc_sweep.py`, where it is measured rather than asserted.
    """

    def test_it_plays_legal_hands(self):
        checker = TestPlayIsLegal("test_random_players")
        for seed in range(3):
            deal = a_deal(seed, dealer=seed % 4)
            result = t.play_deal(deal, pimc_table(seed), allow_loners=True)
            with self.subTest(seed=seed):
                checker.check(result, deal)

    def test_it_never_reads_the_true_deal(self):
        """
        The referee hands every player the truth; an honest one must not look.

        `turn.deal` is the whole table's cards. This wraps it in something that
        screams if it is touched, and does it for all three decisions -- a PIMC
        player that peeked would otherwise just look like a very good PIMC
        player, which is not a thing any other test would notice.
        """
        tripped = []

        class Tripwire:
            def __init__(self, real):
                object.__setattr__(self, "_real", real)

            def __getattr__(self, name):
                tripped.append(name)
                return getattr(object.__getattribute__(self, "_real"), name)

        player = players.PIMCPlayer(samples=2, bid_samples=1,
                                    rng=random.Random(0))
        deal = a_deal(1, dealer=3)

        bid_turn = t.BidTurn(
            deal=Tripwire(deal),
            observation=ob.observe(deal, 0),
            seat=0, bidding_round=b.ROUND_ONE, index=0,
            order=tuple(deal.bidding_order()),
            options=t._bid_options(deal, b.ROUND_ONE, False, False, True),
            stick_the_dealer=False, allow_loners=False)
        self.assertIn(player.bid(bid_turn), bid_turn.options)

        trump = deal.up_card.suit
        six = tuple(deal.hands[3]) + (deal.up_card,)
        taken = game.Deal(
            hands=tuple(six if s == 3 else tuple(h)
                        for s, h in enumerate(deal.hands)),
            up_card=deal.up_card, buried=deal.buried, dealer=3, picked_up=True)
        discard_turn = t.DiscardTurn(
            deal=Tripwire(taken), before=Tripwire(deal),
            observation=ob.Observation(
                seat=3, hand=six, dealer=3, up_card=deal.up_card,
                up_state=ob.PICKED_UP, trump=trump, caller=0,
                pending_discard=True).check(),
            seat=3, caller=0, trump=trump, alone=False, options=six)
        self.assertIn(player.discard(discard_turn), six)

        settled = deal.pick_up(discard=deal.hands[3][0])
        contract = b.Contract(trump, 0, b.ROUND_ONE, settled,
                              deal.hands[3][0], False)
        lead = settled.hands[settled.first_bidder][0]
        play_turn = t.PlayTurn(
            deal=Tripwire(settled),
            observation=ob.observe(
                settled, (settled.first_bidder + 1) % 4,
                plays=((settled.first_bidder, lead),), trump=trump, caller=0,
                up_state=ob.PICKED_UP),
            seat=(settled.first_bidder + 1) % 4, contract=contract,
            legal=tuple(settled.hands[(settled.first_bidder + 1) % 4]),
            plays=((settled.first_bidder, lead),),
            caller_tricks=0, trick_no=0)
        self.assertIn(player.play(play_turn), play_turn.legal)

        self.assertEqual(tripped, [])

    def test_the_pass_models_both_run(self):
        for model in (players.PASS_GOD_MODE, players.PASS_ZERO):
            with self.subTest(model=model):
                result = t.play_deal(
                    a_deal(2, dealer=1),
                    pimc_table(1, pass_model=model))
                self.assertIn(result.value, (-4, -2, -1, 0, 1, 2, 4))

    def test_a_forced_card_costs_no_search(self):
        # With one legal card there is nothing to decide, and deciding it
        # anyway would cost a few hundred solves per trick.
        player = players.PIMCPlayer(samples=50, rng=random.Random(0))
        deal = a_deal(3, dealer=0)
        contract = b.Contract(r.SPADES, 0, b.ROUND_TWO, deal, None, False)
        turn = t.PlayTurn(
            deal=deal, observation=ob.observe(deal, 1, trump=r.SPADES,
                                              caller=0),
            seat=1, contract=contract, legal=(deal.hands[1][0],),
            plays=(), caller_tricks=0, trick_no=0)
        self.assertEqual(player.play(turn), deal.hands[1][0])
        self.assertEqual(player.solves, 0)

    def test_an_unmissable_loner_is_found(self):
        # Seat 1 holds both bowers plus ace-king of trump and an outside ace,
        # with hearts turned up: five tricks with no help needed. An honest
        # player cannot see the other hands, but it does not need to.
        deal = game.deal_from_order(
            r.parse_hand("AD KD QD TD 9D  JH JD AH KH AS  "
                         "AC KC QC TC 9C  KS QS TS 9S QH  "
                         "9H  TH JS JC"),
            dealer=0)
        self.assertEqual(deal.up_card, r.parse_card("9H"))
        table = pimc_table(4, samples=6, bid_samples=6)
        result = t.play_deal(deal, table, allow_loners=True)
        self.assertFalse(result.passed_out)
        self.assertEqual(result.contract.caller, 1)
        self.assertTrue(result.contract.alone)
        self.assertEqual(result.caller_score, 4)


if __name__ == "__main__":
    unittest.main()
