"""
The bitboard solver against the one every other test is written against.

`bitcore` is not a second opinion -- it is meant to return `fast_search`'s
number, every time, and the whole point of the equivalence theorems in
`notes/equivalence.md` is that it may skip work without changing the answer. So
this module asserts equality rather than agreement within something: on whole
deals, four-handed and alone, and on positions reached by walking a random
legal line partway through a hand.

The layering matters, the same way it does for `test_solver.py`. This checks
`bitcore` against `fast_search`; `test_solver.py` checks `fast_search` against
`reference_solver`; and `test_reference_solver.py` checks *that* against the
no-pruning minimax in `euchre_testkit`, which cannot be wrong the way an
alpha-beta search can. Do not collapse the layers by checking `bitcore`
against the oracle directly and calling it done -- what that would miss is a
change of convention that moves both ends at once.
"""
import random
import unittest

import numpy as np

import bitcore
import rotation as r
from fast_search import definitive_winner, position_moves

# A sweep this size takes a couple of seconds and has caught every bug the
# rewrite produced. Raise it when changing the search, not for a routine run.
DEALS = 120
POSITIONS = 400


def card_id(card):
    return card.suit * 6 + (card.rank - r.NINE)


def canon_mask(cards, trump):
    mask = 0
    for card in cards:
        mask |= 1 << int(bitcore.CANON[trump, card_id(card)])
    return mask


def effective_suit(card, trump):
    if card.rank == r.JACK and card.suit == r.same_colour(trump):
        return trump
    return card.suit


def strength(card, trump):
    if card.suit == trump and card.rank == r.JACK:
        return 16
    if effective_suit(card, trump) == trump and card.rank == r.JACK:
        return 15
    return card.rank


def beats(card, winner, trump):
    if effective_suit(card, trump) == trump:
        return (effective_suit(winner, trump) != trump
                or strength(card, trump) > strength(winner, trump))
    if effective_suit(card, trump) != effective_suit(winner, trump):
        return False
    return strength(card, trump) > strength(winner, trump)


def deal_out(rng):
    deck = r.full_deck()
    rng.shuffle(deck)
    return [tuple(deck[i * 5:(i + 1) * 5]) for i in range(4)]


class SolverCase(unittest.TestCase):
    """Shared fixtures: one table, one stack, reused across the sweep."""

    def setUp(self):
        self.tt = bitcore.new_tt(16)
        self.mask = bitcore.tt_mask(self.tt)
        self.nodes = np.zeros(1, dtype=np.int64)
        self.stack = bitcore.new_stack()


class TestWholeDeals(SolverCase):

    def test_matches_fast_search(self):
        """Every deal, four-handed and alone, same value as the reference."""
        rng = random.Random(1234)
        for _ in range(DEALS):
            hands = deal_out(rng)
            trump = rng.randrange(4)
            leader = rng.randrange(4)
            caller = rng.randrange(4)
            engine = r.deal_to_engine(hands, trump)
            masks = [canon_mask(hand, trump) for hand in hands]
            for alone in (0, 1):
                want = definitive_winner(engine, leader, caller,
                                         alone=bool(alone))
                got = bitcore.solve_canon(masks[0], masks[1], masks[2],
                                          masks[3], leader, caller, alone,
                                          self.tt, self.mask, self.nodes,
                                          self.stack)
                self.assertEqual(
                    int(got), int(want),
                    "%s trump=%s leader=%d caller=%d alone=%d"
                    % ([r.hand_name(h) for h in hands], r.suit_name(trump),
                       leader, caller, alone))

    def test_lone_value_is_never_two(self):
        """A loner scores -2, 1 or 4. Two is not on the list."""
        rng = random.Random(77)
        for _ in range(40):
            hands = deal_out(rng)
            trump = rng.randrange(4)
            masks = [canon_mask(hand, trump) for hand in hands]
            value = bitcore.solve_canon(masks[0], masks[1], masks[2], masks[3],
                                        rng.randrange(4), rng.randrange(4), 1,
                                        self.tt, self.mask, self.nodes,
                                        self.stack)
            self.assertIn(int(value), (-2, 1, 4))

    def test_a_cold_table_gives_the_same_answers(self):
        """
        The table is a cache, so emptying it may cost time and nothing else.

        This is the property that lets a sweep keep one table across every deal
        it plays: an entry is keyed by everything its value depends on, so it is
        as true later as it was when it was written.
        """
        rng = random.Random(31415)
        for _ in range(30):
            hands = deal_out(rng)
            trump = rng.randrange(4)
            leader = rng.randrange(4)
            caller = rng.randrange(4)
            masks = [canon_mask(hand, trump) for hand in hands]
            warm = bitcore.solve_canon(masks[0], masks[1], masks[2], masks[3],
                                       leader, caller, 0, self.tt, self.mask,
                                       self.nodes, self.stack)
            cold_tt = bitcore.new_tt(16)
            cold = bitcore.solve_canon(masks[0], masks[1], masks[2], masks[3],
                                       leader, caller, 0, cold_tt,
                                       bitcore.tt_mask(cold_tt), self.nodes,
                                       self.stack)
            self.assertEqual(int(warm), int(cold))


class TestPositions(SolverCase):
    """Part-played positions, which is what a PIMC player actually asks for."""

    def _walk(self, rng):
        """
        A random legal line, stopped partway. Returns the position, or None.

        None means the line ran to the end of the hand, which is not a position
        anybody has to choose a card in.
        """
        hands = deal_out(rng)
        trump = rng.randrange(4)
        caller = rng.randrange(4)
        alone = rng.random() < 0.3
        sitting = (caller + 2) % 4 if alone else None
        width = 3 if alone else 4

        live = [list(h) for h in hands]
        if sitting is not None:
            live[sitting] = []
        seat = rng.randrange(4)
        while seat == sitting:
            seat = rng.randrange(4)

        trick = []
        caller_tricks = 0
        trick_no = 0
        for _ in range(rng.randrange(1, width * 4)):
            if trick:
                led = effective_suit(trick[0][1], trump)
                follow = [c for c in live[seat]
                          if effective_suit(c, trump) == led]
                legal = follow or live[seat]
            else:
                legal = live[seat]
            card = rng.choice(legal)
            live[seat].remove(card)
            trick.append((seat, card))
            seat = (seat + 1) % 4
            if seat == sitting:
                seat = (seat + 1) % 4
            if len(trick) == width:
                best = trick[0]
                for play in trick[1:]:
                    if beats(play[1], best[1], trump):
                        best = play
                if best[0] % 2 == caller % 2:
                    caller_tricks += 1
                seat = best[0]
                trick = []
                trick_no += 1
                if trick_no == 5:
                    return None
        return (hands, live, trick, seat, trump, caller, alone, caller_tricks,
                trick_no, sitting, width)

    def test_matches_position_moves(self):
        """Every legal card gets the value `fast_search` gives it."""
        rng = random.Random(2718)
        checked = 0
        while checked < POSITIONS:
            walked = self._walk(rng)
            if walked is None:
                continue
            (_, live, trick, seat, trump, caller, alone, caller_tricks,
             trick_no, sitting, _) = walked
            checked += 1

            counts = np.array(
                [0 if s == sitting else len(live[s]) for s in range(4)],
                dtype=np.int64)
            width = max(1, int(counts.max()))
            arr = np.zeros((4, width, 2), dtype=np.int64)
            for s in range(4):
                for i, card in enumerate(live[s]):
                    arr[s, i] = r._vec(card, trump)
            trick_cards = np.array([r._vec(c, trump) for _, c in trick],
                                   dtype=np.int64).reshape(-1, 2)
            trick_seats = np.array([s for s, _ in trick], dtype=np.int64)
            idx, vals, _ = position_moves(arr, counts, trick_cards, trick_seats,
                                          seat, caller, caller_tricks,
                                          trick_no, alone)
            want = {live[seat][int(i)]: int(v) for i, v in zip(idx, vals)}

            hands = np.zeros(4, dtype=np.int64)
            for s in range(4):
                hands[s] = canon_mask(live[s], trump)
            if trick:
                led_field = int(bitcore.FIELD_OF[
                    int(bitcore.CANON[trump, card_id(trick[0][1])])])
                best = trick[0]
                for play in trick[1:]:
                    if beats(play[1], best[1], trump):
                        best = play
                win_card = int(bitcore.CANON[trump, card_id(best[1])])
                win_seat = best[0]
            else:
                led_field = win_card = win_seat = 0

            out_bits = np.zeros(8, dtype=np.int64)
            out_vals = np.zeros(8, dtype=np.int64)
            found = bitcore.position_canon(
                hands, seat, len(trick), led_field, win_card, win_seat, caller,
                caller_tricks, trick_no, 1 if alone else 0, self.tt, self.mask,
                self.nodes, self.stack, out_bits, out_vals)
            got = {}
            for i in range(found):
                nat = int(bitcore.DECANON[trump, int(out_bits[i])])
                got[r.Card(nat // 6, nat % 6 + r.NINE)] = int(out_vals[i])

            self.assertEqual(got, want,
                             "trump=%s caller=%d alone=%s trick=%d hands=%s"
                             % (r.suit_name(trump), caller, alone, trick_no,
                                [r.hand_name(h) for h in live]))


class TestCanonicalForm(unittest.TestCase):
    """The card-space permutation, which everything else is expressed in."""

    def test_canon_is_a_bijection_per_trump(self):
        for trump in range(4):
            bits = sorted(int(b) for b in bitcore.CANON[trump])
            self.assertEqual(bits, list(range(24)))

    def test_decanon_inverts_canon(self):
        for trump in range(4):
            for card in range(24):
                bit = int(bitcore.CANON[trump, card])
                self.assertEqual(int(bitcore.DECANON[trump, bit]), card)

    def test_the_left_bower_lands_in_trump_under_the_right(self):
        for trump in range(4):
            left = r.Card(r.same_colour(trump), r.JACK)
            right = r.Card(trump, r.JACK)
            ace = r.Card(trump, r.ACE)
            bits = [int(bitcore.CANON[trump, card_id(c)])
                    for c in (ace, left, right)]
            self.assertTrue(bits[0] < bits[1] < bits[2],
                            "trump %s: ace %d, left %d, right %d"
                            % (r.suit_name(trump), bits[0], bits[1], bits[2]))
            for bit in bits:
                self.assertEqual(int(bitcore.FIELD_OF[bit]),
                                 bitcore.TRUMP_FIELD)

    def test_the_same_colour_suit_has_lost_its_jack(self):
        """Five cards in that field, because its jack is a trump."""
        for trump in range(4):
            same = r.same_colour(trump)
            field = [card for card in range(24)
                     if int(bitcore.FIELD_OF[int(bitcore.CANON[trump, card])])
                     == 1]
            self.assertEqual(len(field), 5)
            self.assertTrue(all(c // 6 == same for c in field))


class TestEquivalence(unittest.TestCase):
    """
    Theorem 1: the move reduction, checked against what it claims.

    `notes/equivalence.md` says two cards of one suit with no live card between
    them are interchangeable. These are the two halves of that: the reduction
    drops exactly the cards it should, and the values it copies onto them are
    the values a full search gives.
    """

    def _reps(self, hand, live, trump):
        got = bitcore._moves(canon_mask(hand, trump), canon_mask(live, trump),
                             0, 0)
        out = []
        for bit in range(24):
            if got >> bit & 1:
                nat = int(bitcore.DECANON[trump, bit])
                out.append(r.Card(nat // 6, nat % 6 + r.NINE))
        return set(out)

    def test_adjacent_cards_collapse(self):
        """9H and TH are one run; the queen is live, so the king is not in it."""
        hand = r.parse_hand("9H TH KH AS")
        live = list(hand) + list(r.parse_hand("QH 9D"))
        self.assertEqual(self._reps(hand, live, r.HEARTS),
                         set(r.parse_hand("9H KH AS")))

    def test_a_whole_run_collapses_to_one(self):
        """With the queen dead as well, 9-10-K is a single run."""
        hand = r.parse_hand("9H TH KH AS")
        live = list(hand) + list(r.parse_hand("QC KC 9D"))
        self.assertEqual(self._reps(hand, live, r.HEARTS),
                         set(r.parse_hand("9H AS")))

    def test_a_dead_card_between_them_collapses_them_too(self):
        """With the jack and queen out of play, 10-K is one run."""
        hand = r.parse_hand("TH KH AS 9C")
        live = list(hand) + list(r.parse_hand("9D TD"))
        self.assertEqual(self._reps(hand, live, r.HEARTS),
                         set(r.parse_hand("TH AS 9C")))

    def test_a_live_card_between_them_keeps_them_apart(self):
        """The same hand, but somebody else still holds the queen."""
        hand = r.parse_hand("TH KH AS 9C")
        live = list(hand) + list(r.parse_hand("QH 9D"))
        self.assertEqual(self._reps(hand, live, r.HEARTS),
                         set(r.parse_hand("TH KH AS 9C")))

    def test_the_left_bower_is_not_adjacent_to_its_printed_neighbours(self):
        """JD next to TD is two suits, not one run, when hearts are trump."""
        hand = r.parse_hand("TD JD 9S")
        live = list(hand)
        self.assertEqual(self._reps(hand, live, r.HEARTS),
                         set(r.parse_hand("TD JD 9S")))


if __name__ == "__main__":
    unittest.main()
