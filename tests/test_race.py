"""
Unit tests for players._race -- the sequential elimination behind `epsilon`.

The racing rule is the one place in the repo that trades an exact answer for a
cheaper one, so it is tested on both halves of the bargain.

**What must not change.** With `epsilon=None` the rule is off, and off has to
mean bit-for-bit off: exactly `budget` worlds drawn, every option kept, the
same rng consumed. A player that quietly drew fewer worlds would hand every
later decision in the deal a different set of layouts, so an unraced sweep
would stop reproducing even though nothing about it was meant to change.

**What may change, and by how much.** With `epsilon` set, an option is dropped
once the paired difference says it is losing, or once it is within `epsilon` of
the leader. The guarantee is only ever "the survivor is within epsilon of the
best", so the tests assert that shape -- a dominated option loses, a dominant
one is never dropped, two options that always agree tie out rather than running
the full budget -- rather than pinning an exact choice.

`draw` here returns fixed numbers rather than solving anything: the stopping
rule is the subject, and a real solver would only make it slow and non-obvious.
The one test that does drive a real table is the loner, which is there because
no budget and no indifference band should be able to talk a player out of a
hand that takes all five tricks unaided.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import unittest

import bidding as b
import game
import observation as ob
import players
import rotation as r
import table as t

from test_table import a_deal


def counting_draw(values_for):
    """A draw that records what it was asked for, and how often."""
    calls = []

    def draw(active):
        calls.append(tuple(active))
        return {o: values_for(o, len(calls) - 1) for o in active}

    return draw, calls


class TestRaceOff(unittest.TestCase):
    """epsilon=None has to be indistinguishable from never having raced."""

    def test_draws_exactly_the_budget(self):
        draw, calls = counting_draw(lambda o, i: 0)
        sums, counts, alive = players._race(("a", "b", "c"), draw, 40,
                                            epsilon=None)
        self.assertEqual(len(calls), 40)
        self.assertEqual(counts, [40, 40, 40])
        self.assertEqual(alive, [0, 1, 2])

    def test_keeps_hopeless_options(self):
        # "b" loses in every world and is still carried to the end, because
        # with racing off there is nothing to carry it out.
        draw, calls = counting_draw(lambda o, i: 0 if o == "b" else 5)
        sums, counts, alive = players._race(("a", "b"), draw, 60, epsilon=None)
        self.assertEqual(len(calls), 60)
        self.assertEqual(alive, [0, 1])

    def test_a_single_option_still_draws(self):
        # Nothing to decide, but the rng consumption has to match the unraced
        # player anyway -- a short draw here would desynchronise the deal.
        draw, calls = counting_draw(lambda o, i: 1)
        players._race(("only",), draw, 12, epsilon=None)
        self.assertEqual(len(calls), 12)


class TestRaceOn(unittest.TestCase):

    def test_a_dominated_option_is_dropped_early(self):
        draw, calls = counting_draw(lambda o, i: 0 if o == "b" else 4)
        sums, counts, alive = players._race(("a", "b"), draw, 5000,
                                            epsilon=0.05)
        self.assertEqual(alive, [0])
        self.assertLess(len(calls), 200,
                        "a four-point gap should not need hundreds of worlds")
        # Both end on the same count: dropping "b" leaves one option, and a
        # race with nothing left to compare stops rather than sampling on.
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0], len(calls))

    def test_the_winner_is_never_dropped(self):
        # The leader is only ever the reference the others are measured
        # against, so the survivor of a one-sided race is the dominant option.
        for winner in ("a", "b", "c"):
            draw, _ = counting_draw(lambda o, i, w=winner: 2 if o == w else -1)
            _, _, alive = players._race(("a", "b", "c"), draw, 2000,
                                        epsilon=0.05)
            self.assertEqual(alive, [("a", "b", "c").index(winner)])

    def test_options_that_always_agree_tie_out(self):
        # The expensive case: options the search cannot separate. Exact
        # averaging runs the whole budget to split them; the indifference band
        # calls it a tie, and that is the saving epsilon buys.
        draw, calls = counting_draw(lambda o, i: 1)
        _, _, alive = players._race(("a", "b"), draw, 10000, epsilon=0.05)
        self.assertEqual(len(alive), 1)
        self.assertLess(len(calls), 300,
                        "identical options should not run the full budget")

    def test_a_real_gap_survives_the_band(self):
        # A whole point apart is far more than epsilon, so the band must not
        # collapse it -- only the losing test may drop "b", and it must.
        draw, _ = counting_draw(lambda o, i: 1 if o == "a" else 0)
        _, _, alive = players._race(("a", "b"), draw, 4000, epsilon=0.05)
        self.assertEqual(alive, [0])

    def test_nothing_is_drawn_when_there_is_nothing_to_choose(self):
        draw, calls = counting_draw(lambda o, i: 1)
        _, counts, alive = players._race(("only",), draw, 500, epsilon=0.05)
        self.assertEqual(calls, [])
        self.assertEqual(alive, [0])

    def test_the_budget_is_still_a_ceiling(self):
        # Two options that never separate, with epsilon 0 so the band cannot
        # end it either: the race has to stop at the budget regardless.
        rng = random.Random(0)
        draw, calls = counting_draw(lambda o, i: rng.choice((-2, 1, 2)))
        players._race(("a", "b"), draw, 150, epsilon=0.0)
        self.assertLessEqual(len(calls), 150)

    def test_a_wider_band_stops_sooner(self):
        # The knob has to be monotone, since that is how it gets tuned: the
        # cost of calling two noisy options tied goes as (sd / epsilon)^2.
        def cost(epsilon):
            rng = random.Random(11)
            draw, calls = counting_draw(
                lambda o, i: rng.choice((-2, 1, 2)))
            players._race(("a", "b"), draw, 20000, epsilon=epsilon)
            return len(calls)

        wide, narrow = cost(0.5), cost(0.1)
        self.assertLess(wide, narrow)


class TestRacedPlayer(unittest.TestCase):
    """The stopping rule, driven through a real table rather than a stub."""

    def test_an_unmissable_loner_is_still_found(self):
        # Seat 1 holds both bowers plus ace-king of trump and an outside ace:
        # five tricks with no help needed, and no stopping rule may lose it.
        deal = game.deal_from_order(
            r.parse_hand("AD KD QD TD 9D  JH JD AH KH AS  "
                         "AC KC QC TC 9C  KS QS TS 9S QH  "
                         "9H  TH JS JC"),
            dealer=0)
        table = [players.PIMCPlayer(samples=60, bid_samples=60, epsilon=0.05,
                                    pass_model=players.PASS_ZERO,
                                    rng=random.Random(4 + s))
                 for s in range(game.PLAYERS)]
        result = t.play_deal(deal, table, allow_loners=True)
        self.assertFalse(result.passed_out)
        self.assertEqual(result.contract.caller, 1)
        self.assertTrue(result.contract.alone)
        self.assertEqual(result.caller_score, 4)

    def test_racing_costs_fewer_solves_than_exact(self):
        def spend(epsilon):
            table = [players.PIMCPlayer(samples=400, bid_samples=400,
                                        epsilon=epsilon,
                                        pass_model=players.PASS_ZERO,
                                        rng=random.Random(7 + s))
                     for s in range(game.PLAYERS)]
            t.play_deal(a_deal(5, dealer=2), table, allow_loners=True)
            return sum(p.solves for p in table)

        self.assertLess(spend(0.25), spend(None) / 2,
                        "racing should more than halve the work at 400 worlds")

    def test_a_raced_player_still_never_peeks(self):
        # test_table.py wraps turn.deal in a tripwire and asserts PIMCPlayer
        # never reads it. Repeated here with racing on, because an early stop
        # that reached for the truth would look like a very good player.
        seen = []

        class Tripwire:
            def __init__(self, real):
                object.__setattr__(self, "_real", real)

            def __getattr__(self, name):
                seen.append(name)
                return getattr(object.__getattribute__(self, "_real"), name)

        deal = a_deal(2, dealer=3)
        player = players.PIMCPlayer(samples=30, bid_samples=30, epsilon=0.05,
                                    pass_model=players.PASS_ZERO,
                                    rng=random.Random(0))
        turn = t.BidTurn(
            deal=Tripwire(deal), observation=ob.observe(deal, 0),
            seat=0, bidding_round=b.ROUND_ONE, index=0,
            order=tuple(deal.bidding_order()),
            options=t._bid_options(deal, b.ROUND_ONE, False, False, True),
            stick_the_dealer=False, allow_loners=True)
        player.bid(turn)
        self.assertEqual(seen, [], "a raced PIMC player read turn.deal")


if __name__ == "__main__":
    unittest.main()
