"""
The compiled sweep against the readable one.

`fastsim.py` is `table.py`, `players.py`, `observation.py` and `bidding.py`
again with nothing in them but integers. Two implementations of one model is a
liability unless something holds them together, and this is that something.

The two do **not** agree deal by deal, and cannot: `random.Random` does not
exist inside njit, so the compiled engine carries its own splitmix64 and draws
different worlds. What is checked instead, from the bottom up:

  * the pieces that involve no randomness at all -- the auction in God Mode,
    the trick rules, what a seat can infer -- are checked for **exact
    equality** against `bidding.py` and `observation.py`;
  * the decisions, which do involve randomness, are checked by handing *both*
    implementations the same sampled worlds and requiring the same card. That
    pins the stopping rule, the averaging and the tie-breaks together, and it
    is the test that would catch a plausible-looking player that quietly
    differs;
  * the sampler itself is checked against the observation it came from: every
    world it draws must be one the seat could actually be in;
  * and the sweep as a whole is checked for the invariant that made the old
    process pool trustworthy -- the answer does not depend on how many threads
    computed it.

`tests/test_bitcore.py` covers the search underneath all of this.
"""
import random
import unittest

import numpy as np

import bidding as b
import bitcore
import fastsim as F
import game
import observation as obs
import players
import rotation as r
import table as t
from fast_search import position_moves


def card_id(card):
    return card.suit * 6 + (card.rank - r.NINE)


def to_mask(cards):
    mask = 0
    for card in cards:
        mask |= 1 << card_id(card)
    return mask


def to_cards(mask):
    return tuple(r.Card(i // 6, i % 6 + r.NINE)
                 for i in range(24) if mask >> i & 1)


def a_deal(rng, dealer):
    return game.deal_random(rng=random.Random(rng.randrange(10 ** 9)),
                            dealer=dealer)


class TestCardRules(unittest.TestCase):
    """The trick rules, which everything above them assumes."""

    def test_trick_winner_matches_table(self):
        rng = random.Random(11)
        for _ in range(400):
            trump = rng.randrange(4)
            deck = r.full_deck()
            rng.shuffle(deck)
            size = rng.randrange(2, 5)
            trick = [(i, deck[i]) for i in range(size)]
            want = t.trick_winner(trick, trump)
            seat, best = trick[0]
            for other, card in trick[1:]:
                if F.beats(card_id(card), card_id(best), trump):
                    seat, best = other, card
            self.assertEqual(seat, want, "%s trump %s"
                             % (r.hand_name([c for _, c in trick]),
                                r.suit_name(trump)))

    def test_legal_cards_match_table(self):
        rng = random.Random(12)
        for _ in range(400):
            trump = rng.randrange(4)
            deck = r.full_deck()
            rng.shuffle(deck)
            hand = tuple(deck[:rng.randrange(1, 6)])
            led = deck[10] if rng.random() < 0.8 else None
            want = set(t.legal_cards(hand, led, trump))
            got = to_cards(F.legal_mask(to_mask(hand),
                                        -1 if led is None else card_id(led),
                                        trump))
            self.assertEqual(set(got), want)

    def test_tie_rank_matches_card_order(self):
        """
        The order `players._pick` throws a tied card in, reproduced as a table.

        Two runs that disagree here disagree on the card played, so this is
        checked against `table.card_order` directly rather than assumed.
        """
        for trump in range(4):
            by_table = sorted(r.full_deck(),
                              key=lambda c: t.card_order(c, trump))
            by_fast = sorted(r.full_deck(),
                             key=lambda c: int(F.TIE_RANK[trump, card_id(c)]))
            self.assertEqual(by_fast, by_table, r.suit_name(trump))


class TestAuction(unittest.TestCase):
    """The God Mode auction: same value, same contract, every time."""

    def setUp(self):
        self.tt = bitcore.new_tt(18)
        self.mask = bitcore.tt_mask(self.tt)
        self.nodes = np.zeros(F.COUNTERS, dtype=np.int64)
        self.stack = bitcore.new_stack()
        self.out = np.zeros((1, F.RECORD), dtype=np.int64)

    def test_matches_solve_bidding(self):
        rng = random.Random(4242)
        hands = np.zeros(4, dtype=np.int64)
        for _ in range(60):
            dealer = rng.randrange(4)
            deal = a_deal(rng, dealer)
            loners = rng.random() < 0.5
            stick = rng.random() < 0.3
            want = b.solve_bidding(deal, stick_the_dealer=stick,
                                   allow_loners=loners)
            for seat in range(4):
                hands[seat] = to_mask(deal.hands[seat])
            F.solve_bidding(hands, card_id(deal.up_card), dealer,
                            1 if stick else 0, 1 if loners else 0, self.tt,
                            self.mask, self.nodes, self.stack, self.out, 0)
            where = "dealer %d stick %s loners %s" % (dealer, stick, loners)
            self.assertEqual(int(self.out[0, F.R_VALUE]), int(want.value),
                             where)
            self.assertEqual(int(self.out[0, F.R_CALLER]),
                             -1 if want.passed_out else want.contract.caller,
                             where)
            self.assertEqual(int(self.out[0, F.R_TRUMP]),
                             -1 if want.passed_out else want.contract.trump,
                             where)
            self.assertEqual(bool(self.out[0, F.R_ALONE]),
                             False if want.passed_out
                             else bool(want.contract.alone), where)

    def test_rest_of_auction_matches(self):
        """
        What passing is worth, which is nearly the whole cost of `pass_model
        "god"` and the only thing that prices a decline.

        Checked from every index of both rounds, because `bidding` recurses
        forwards down the chain and `fastsim` solves it backwards, and the two
        only agree if the option order and the tie rule agree at every step.
        """
        rng = random.Random(3131)
        hands = np.zeros(4, dtype=np.int64)
        for _ in range(25):
            dealer = rng.randrange(4)
            deal = a_deal(rng, dealer)
            loners = rng.random() < 0.5
            stick = rng.random() < 0.4
            order = deal.bidding_order()
            for seat in range(4):
                hands[seat] = to_mask(deal.hands[seat])
            for bidding_round in (b.ROUND_ONE, b.ROUND_TWO):
                for index in range(5):
                    want = b.rest_of_auction(deal, index, order, stick, loners,
                                             bidding_round)
                    got = F.rest_of_auction(
                        hands, card_id(deal.up_card), dealer, index,
                        bidding_round, 1 if stick else 0, 1 if loners else 0,
                        self.tt, self.mask, self.nodes, self.stack)
                    self.assertEqual(int(got), int(want.value),
                                     "round %d index %d stick %s loners %s"
                                     % (bidding_round, index, stick, loners))

    def test_order_up_matches(self):
        """
        The dealer's pick-up, which is where the sign conventions go wrong.

        `order_up` is scored for the *dealer's* team, not the caller's, so an
        opposing dealer pitches to hurt the contract. It is also where the
        alpha-beta window and the equivalent-discard reduction are applied, and
        neither may move the number.
        """
        rng = random.Random(98)
        hands = np.zeros(4, dtype=np.int64)
        for _ in range(60):
            dealer = rng.randrange(4)
            deal = a_deal(rng, dealer)
            for seat in range(4):
                hands[seat] = to_mask(deal.hands[seat])
            for caller in range(4):
                for alone in (0, 1):
                    want, _ = b.order_up(deal, caller, bool(alone))
                    got = F.order_up_value(hands, card_id(deal.up_card),
                                           dealer, caller, alone, self.tt,
                                           self.mask, self.nodes, self.stack)
                    self.assertEqual(int(got), int(want),
                                     "dealer %d caller %d alone %d"
                                     % (dealer, caller, alone))


class TestDealing(unittest.TestCase):
    """`deal_around`: the pinned cards stay pinned, and none go missing."""

    def test_every_card_is_somewhere_exactly_once(self):
        """
        `dealer.py` shipped a card-losing bug twice, which is why `game.Deal`
        checks this rather than assuming it. Nothing inside the compiled engine
        tracks the kitty -- every observation derives how deep it is -- so this
        is where the arithmetic gets asserted.
        """
        rng = np.zeros(1, dtype=np.int64)
        hands = np.zeros(4, dtype=np.int64)
        pin = to_mask(r.parse_hand("JS AS 9H 9D TC"))
        up = card_id(r.parse_card("9S"))
        for i in range(200):
            F.seed_stream(rng, i)
            turned, kitty = F.deal_around(pin, 2, up, 0, rng, hands)
            self.assertEqual(turned, up)
            self.assertEqual(int(hands[2]), pin)
            seen = kitty | (1 << turned)
            for seat in range(4):
                self.assertEqual(F.popcount(int(hands[seat])), 5)
                self.assertEqual(int(hands[seat]) & seen, 0)
                seen |= int(hands[seat])
            self.assertEqual(seen, (1 << 24) - 1)

    def test_an_unpinned_up_card_is_dealt_too(self):
        rng = np.zeros(1, dtype=np.int64)
        hands = np.zeros(4, dtype=np.int64)
        turned_up = set()
        for i in range(60):
            F.seed_stream(rng, i)
            turned, kitty = F.deal_around(0, 0, -1, 3, rng, hands)
            turned_up.add(turned)
            seen = kitty | (1 << turned)
            for seat in range(4):
                seen |= int(hands[seat])
            self.assertEqual(seen, (1 << 24) - 1)
        self.assertGreater(len(turned_up), 5, "the up-card never moved")


class TestInference(unittest.TestCase):
    """What a seat can work out, and the worlds that are consistent with it."""

    def _state(self, rng, acting=False):
        """
        A random game state, plus the Observation one seat has of it.

        `acting` picks the seat whose turn it is rather than one at random,
        which is the only seat a card decision can be asked of.
        """
        dealer = rng.randrange(4)
        deal = a_deal(rng, dealer)
        picked = rng.random() < 0.6
        discard = None
        settled = deal
        if picked:
            discard = rng.choice(deal.hands[dealer])
            settled = deal.pick_up(discard=discard)
            trump = deal.up_card.suit
        else:
            trump = rng.randrange(4)
        caller = rng.randrange(4)
        alone = rng.random() < 0.25
        sitting = (caller + 2) % 4 if alone else None
        width = 3 if alone else 4

        live = [list(h) for h in settled.hands]
        if sitting is not None:
            live[sitting] = []
        seat = (dealer + 1) % 4
        if seat == sitting:
            seat = (seat + 1) % 4
        plays = []
        for _ in range(rng.randrange(0, width * 4)):
            led = plays[len(plays) // width * width][1] \
                if len(plays) % width else None
            card = rng.choice(t.legal_cards(live[seat], led, trump))
            live[seat].remove(card)
            plays.append((seat, card))
            if len(plays) % width == 0:
                seat = t.trick_winner(plays[-width:], trump)
            else:
                seat = t.next_seat(seat, sitting)

        observer = seat if acting else rng.randrange(4)
        if observer == sitting:
            return None
        view = obs.Observation(
            seat=observer, hand=tuple(live[observer]), dealer=dealer,
            up_card=deal.up_card,
            up_state=obs.PICKED_UP if picked else obs.TURNED_DOWN,
            plays=tuple(plays), trump=trump, caller=caller, alone=alone,
            discard=(discard if observer == dealer and picked else None))
        try:
            view.check()
        except ValueError:
            return None
        return (deal, live, plays, view, trump, sitting, width, picked,
                discard)

    def _setup(self, state):
        """`fastsim.obs_setup` run on a state from `_state`."""
        deal, live, plays, view, trump, sitting, width, picked, discard = state
        hands = np.zeros(4, dtype=np.int64)
        for seat in range(4):
            hands[seat] = to_mask(live[seat])
        played_seats = np.zeros(20, dtype=np.int64)
        played_cards = np.zeros(20, dtype=np.int64)
        for i, (seat, card) in enumerate(plays):
            played_seats[i] = seat
            played_cards[i] = card_id(card)
        caps = np.zeros(5, dtype=np.int64)
        voids = np.zeros(4, dtype=np.int64)
        pool, forced = F.obs_setup(
            view.seat, hands, deal.dealer, card_id(deal.up_card),
            F.PICKED_UP if picked else F.TURNED_DOWN, trump,
            -1 if sitting is None else sitting,
            card_id(discard) if view.discard is not None else -1,
            False, played_seats, played_cards, len(plays), width, caps, voids)
        return hands, pool, forced, caps, voids

    def test_matches_observation(self):
        """Same unseen pool, same room per slot, same voids, same up-card."""
        rng = random.Random(9090)
        checked = 0
        for _ in range(400):
            state = self._state(rng)
            if state is None:
                continue
            view = state[3]
            want_pool, want_caps, want_voids, want_forced, _, _ = \
                obs._draw_setup(view)
            _, pool, forced, caps, voids = self._setup(state)
            checked += 1

            self.assertEqual(set(to_cards(pool)), set(want_pool))
            self.assertEqual([int(c) for c in caps], list(want_caps))
            for seat in range(4):
                shown = {s for s in range(4) if int(voids[seat]) >> s & 1}
                self.assertEqual(shown, set(want_voids[seat]),
                                 "voids for seat %d" % seat)
            self.assertEqual(int(forced),
                             card_id(want_forced[0][0]) if want_forced else -1)
        self.assertGreater(checked, 200, "too few states survived to be a test")

    def test_sampled_worlds_are_consistent(self):
        """
        Every world the sampler draws is one the seat could really be in.

        The same four things `tests/test_observation.py` asks of the Python
        sampler: the observer's own hand is its real one, every seat holds the
        number of cards it must, nobody is dealt a suit they have shown out of,
        and every card is somewhere exactly once.
        """
        rng = random.Random(555)
        stream = np.zeros(1, dtype=np.int64)
        world = np.zeros(5, dtype=np.int64)
        checked = 0
        for _ in range(200):
            state = self._state(rng)
            if state is None:
                continue
            deal, live, plays, view, trump, sitting, width, picked, _ = state
            hands, pool, forced, caps, voids = self._setup(state)
            F.seed_stream(stream, rng.randrange(10 ** 9))
            for _ in range(8):
                ok = F.draw_world(pool, caps, voids,
                                  -1 if sitting is None else sitting, trump,
                                  hands[view.seat], view.seat, deal.dealer,
                                  forced, stream, world)
                self.assertTrue(ok, "the sampler gave up on a real position")
                checked += 1

                self.assertEqual(int(world[view.seat]), int(hands[view.seat]))
                seen = 0
                for slot in range(5):
                    self.assertEqual(int(world[slot]) & seen, 0,
                                     "a card was dealt to two places")
                    seen |= int(world[slot])
                known = seen | to_mask([c for _, c in plays])
                known |= to_mask(view.known_kitty())
                self.assertEqual(known, (1 << 24) - 1,
                                 "a card ended up nowhere")

                counts = view.counts()
                for seat in range(4):
                    self.assertEqual(F.popcount(int(world[seat])),
                                     counts[seat],
                                     "seat %d holds the wrong number" % seat)
                shown = view.voids()
                for seat in range(4):
                    if seat == sitting:
                        continue
                    for card in to_cards(int(world[seat])):
                        self.assertNotIn(
                            obs.effective_suit(card, trump), shown[seat],
                            "seat %d was dealt a suit it showed out of" % seat)
        self.assertGreater(checked, 400)


class TestDecisions(unittest.TestCase):
    """
    The compiled decisions against `players.py`, on identical worlds.

    Each test drives both implementations from the same splitmix64 stream, so
    they imagine the same layouts in the same order, and then insists they take
    the same option. What that covers is everything downstream of the sampler:
    the values, the paired-difference stopping rule and its constants, and the
    tie-break. A player that differed anywhere in there would look like nothing
    worse than a slightly different player, which is exactly the failure this
    is here to catch.
    """

    def setUp(self):
        self.tt = bitcore.new_tt(18)
        self.mask = bitcore.tt_mask(self.tt)
        self.nodes = np.zeros(F.COUNTERS, dtype=np.int64)
        self.stack = bitcore.new_stack()

    def _python_play(self, observer, hands, dealer, up_card, up_state, trump,
                     caller, alone, plays, n_in_trick, led_card, win_card,
                     win_seat, caller_tricks, trick_no, budget, epsilon,
                     stream, pool, caps, voids, forced, sitting):
        """
        `PIMCPlayer.play`, fed by the compiled sampler and solved by
        `fast_search`. Nothing of `fastsim`'s decision code is used.
        """
        legal = to_cards(F.legal_mask(int(hands[observer]),
                                      -1 if led_card < 0 else led_card, trump))
        if len(legal) == 1:
            return legal[0]
        world = np.zeros(5, dtype=np.int64)

        def draw(_active):
            ok = F.draw_world(pool, caps, voids,
                              -1 if sitting is None else sitting, trump,
                              hands[observer], observer, dealer, forced,
                              stream, world)
            assert ok
            held = [list(to_cards(int(world[s]))) for s in range(4)]
            counts = np.array(
                [0 if s == sitting else len(held[s]) for s in range(4)],
                dtype=np.int64)
            width = max(1, int(counts.max()))
            arr = np.zeros((4, width, 2), dtype=np.int64)
            for seat in range(4):
                if seat == sitting:
                    continue        # those cards are as out of play as the kitty
                for i, card in enumerate(held[seat]):
                    arr[seat, i] = r._vec(card, trump)
            trick_cards = np.array([r._vec(c, trump) for _, c in plays],
                                   dtype=np.int64).reshape(-1, 2)
            trick_seats = np.array([s for s, _ in plays], dtype=np.int64)
            idx, vals, _ = position_moves(arr, counts, trick_cards,
                                          trick_seats, observer, caller,
                                          caller_tricks, trick_no, bool(alone))
            return {held[observer][int(i)]:
                    players.to_seat(int(v), caller, observer)
                    for i, v in zip(idx, vals)}

        sums, counts, alive = players._race(legal, draw, budget, epsilon,
                                            players.MIN_WORLDS)
        scores = players._means(legal, sums, counts)
        live = tuple(legal[i] for i in alive)
        return players._pick(scores, live, tie_break=players.LOW, trump=trump)

    def test_play_agrees_on_the_same_worlds(self):
        rng = random.Random(606)
        inference = TestInference()
        checked = 0
        for _ in range(60):
            state = inference._state(rng, acting=True)
            if state is None:
                continue
            deal, live, plays, view, trump, sitting, width, picked, discard = \
                state
            if not view.hand:
                continue
            trick = list(view.current_trick)
            observer = view.seat
            hands, pool, forced, caps, voids = inference._setup(state)
            if len(to_cards(int(hands[observer]))) < 2:
                continue

            if trick:
                led_card = card_id(trick[0][1])
                best = trick[0]
                for play in trick[1:]:
                    if F.beats(card_id(play[1]), card_id(best[1]), trump):
                        best = play
                win_card = card_id(best[1])
                win_seat = best[0]
            else:
                led_card = win_card = -1
                win_seat = -1

            caller_tricks = 0
            for i in range(view.trick_no):
                won = t.trick_winner(list(plays[i * width:(i + 1) * width]),
                                     trump)
                if won % 2 == view.caller % 2:
                    caller_tricks += 1

            played_seats = np.zeros(20, dtype=np.int64)
            played_cards = np.zeros(20, dtype=np.int64)
            for i, (seat, card) in enumerate(plays):
                played_seats[i] = seat
                played_cards[i] = card_id(card)

            seed = rng.randrange(10 ** 9)
            budget = 40
            epsilon = 0.05

            stream = np.zeros(1, dtype=np.int64)
            F.seed_stream(stream, seed)
            got = F.pimc_play(
                observer, hands, deal.dealer, card_id(deal.up_card),
                F.PICKED_UP if picked else F.TURNED_DOWN, trump, view.caller,
                1 if view.alone else 0,
                card_id(discard) if view.discard is not None else -1,
                played_seats, played_cards, len(plays), width, len(trick),
                led_card, win_card, win_seat, caller_tricks, view.trick_no,
                budget, epsilon, True, players.MIN_WORLDS, stream, self.tt,
                self.mask, self.nodes, self.stack)

            stream2 = np.zeros(1, dtype=np.int64)
            F.seed_stream(stream2, seed)
            want = self._python_play(
                observer, hands, deal.dealer, card_id(deal.up_card),
                F.PICKED_UP if picked else F.TURNED_DOWN, trump, view.caller,
                1 if view.alone else 0, trick, len(trick), led_card, win_card,
                win_seat, caller_tricks, view.trick_no, budget, epsilon,
                stream2, pool, caps, voids, forced, sitting)
            checked += 1
            self.assertEqual(to_cards(1 << int(got))[0], want,
                             "trick %d, seat %d, trump %s"
                             % (view.trick_no, observer, r.suit_name(trump)))
        self.assertGreater(checked, 20, "too few positions survived")


class TestNames(unittest.TestCase):
    """The strings the command line speaks, against the integers njit speaks."""

    def test_every_option_has_a_code(self):
        import hand_ev

        pass_models, assumptions = hand_ev.codes(F)
        self.assertEqual(set(pass_models), set(players.PASS_MODELS))
        self.assertEqual(set(assumptions), set(hand_ev.ASSUMPTIONS))
        self.assertEqual(len(set(pass_models.values())), len(pass_models))
        self.assertEqual(len(set(assumptions.values())), len(assumptions))


class TestSweep(unittest.TestCase):
    """The whole sweep, and the invariants that make its number readable."""

    def _run(self, deals, chunks, seed=0, tt_bits=18, **kw):
        pin = to_mask(r.parse_hand("JS AS 9H 9D TC"))
        up = card_id(r.parse_card("9S"))
        tt = bitcore.new_tt(tt_bits)
        out = np.zeros((deals, F.RECORD), dtype=np.int64)
        god = np.zeros((deals, F.RECORD), dtype=np.int64)
        counters = np.zeros((chunks, F.COUNTERS), dtype=np.int64)
        F.run_deals(0, deals, seed, pin, kw.get("seat", 0), up,
                    kw.get("dealer", 3), 20, 12, 12, 0.05, True,
                    players.MIN_WORLDS, F.PASS_ZERO,
                    kw.get("assume", F.ASSUME_AUCTION), 1, 0, 1, chunks, tt,
                    out, god, kw.get("both", 0), counters)
        return out, god

    def test_the_answer_does_not_depend_on_the_thread_count(self):
        """
        The cheapest available check that the parallel path is sound.

        Every deal seeds its own streams from its own index, so one thread and
        twelve must produce the same records in the same order -- not the same
        mean, the same records.
        """
        one, _ = self._run(40, 1)
        many, _ = self._run(40, 7)
        np.testing.assert_array_equal(one, many)

    def test_a_bigger_table_does_not_change_the_answer(self):
        """The transposition table is a cache. Resizing it may only cost time."""
        small, _ = self._run(30, 3, tt_bits=14)
        large, _ = self._run(30, 3, tt_bits=20)
        np.testing.assert_array_equal(small, large)

    def test_records_are_on_the_asking_seats_scale(self):
        """
        Every reported number is the asking seat's team's, derived from the
        caller's rather than compared against the conversion that produced it.
        A sign error is invisible on any deal where the two teams agree.
        """
        out, _ = self._run(40, 2, seat=1, dealer=3)
        for row in out:
            caller = int(row[F.R_CALLER])
            if caller < 0:
                self.assertEqual(int(row[F.R_VALUE]), 0)
                continue
            score = int(row[F.R_SCORE])
            tricks = int(row[F.R_TRICKS])
            alone = bool(row[F.R_ALONE])
            if tricks == 5:
                self.assertEqual(score, 4 if alone else 2)
            elif tricks >= 3:
                self.assertEqual(score, 1)
            else:
                self.assertEqual(score, -2)
            want = score if (caller % 2) == (1 % 2) else -score
            self.assertEqual(int(row[F.R_VALUE]), want)

    def test_a_pinned_bid_is_recorded_when_it_fires(self):
        """
        `--assume order` pins the opening bid, and only when it is reached.

        From the eldest seat it always is; the flag exists for the seats where
        an earlier caller can take the option away, and a sweep that averaged
        those in would be answering a different question.
        """
        out, _ = self._run(40, 2, seat=0, dealer=3, assume=F.ASSUME_ORDER)
        for row in out:
            self.assertEqual(int(row[F.R_FORCED]), 1)
            self.assertEqual(int(row[F.R_CALLER]), 0)

        out, _ = self._run(40, 2, seat=2, dealer=3, assume=F.ASSUME_ORDER)
        fired = sum(int(row[F.R_FORCED]) for row in out)
        self.assertGreater(fired, 0)
        for row in out:
            if int(row[F.R_FORCED]):
                self.assertEqual(int(row[F.R_CALLER]), 2)

    def test_god_mode_rows_match_solve_bidding(self):
        """`--both` solves the same layout it played, not a different one."""
        out, god = self._run(20, 2, both=1)
        rng = np.zeros(1, dtype=np.int64)
        hands = np.zeros(4, dtype=np.int64)
        pin = to_mask(r.parse_hand("JS AS 9H 9D TC"))
        up = card_id(r.parse_card("9S"))
        for i in range(20):
            F.seed_stream(rng, i)
            turned, buried = F.deal_around(pin, 0, up, 3, rng, hands)
            deal = game.Deal(
                hands=tuple(to_cards(int(hands[s])) for s in range(4)),
                up_card=r.Card(turned // 6, turned % 6 + r.NINE),
                buried=to_cards(buried), dealer=3).check()
            want = b.solve_bidding(deal, allow_loners=True)
            self.assertEqual(int(god[i, F.R_VALUE]), int(want.value))


if __name__ == "__main__":
    unittest.main()
