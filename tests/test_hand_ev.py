"""
Unit tests for hand_ev.py.

`hand_ev` computes no Euchre of its own -- it pins a hand with
`game.deal_around`, hands each sampled layout to `table.play_deal`, and adds up
what comes back. So there is nothing here about whether a card was well played;
`test_table.py` and `test_solver.py` own that. What is tested is the three
things this module can get wrong on its own:

**The question stays pinned.** Every sampled deal has to hold the asked-about
hand at the asked-about seat, with the asked-about up-card and dealer. A sweep
that quietly re-deals the hand it was asked about answers a different question
and looks completely normal doing it.

**A deal is a function of its index.** `Setup.deal(i)` seeds from `i` alone.
That is what makes the parallel path safe, and it is the load-bearing claim in
this module's docstring -- so it is checked directly, and then checked again end
to end by running the same sweep across two processes and demanding the same
records back. That last test spawns a pool and pays the JIT warmup twice, which
is most of this file's runtime; it earns it, because an unordered pool silently
scrambling records is exactly the bug that would survive every other test here.

**The seat's scale.** Every number reported is on the asking seat's own team's
scale, converted from the referee's net-to-team-0. A sign error there is
invisible on any deal where the two teams agree, which is why it gets its own
invariant rather than a spot check.

The sims are deliberately tiny -- one or two worlds per decision. Nothing here
depends on a PIMC player choosing *well*, only on the bookkeeping around it
being right.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib
import io
import unittest

import bidding as b
import game
import hand_ev as h
import rotation as r

# Seat 0 holds both bowers, the ace of trump and two side cards, with the nine
# of spades up. It gets ordered up nearly every deal, so a sweep over it always
# reaches the play rather than passing out and testing nothing.
HAND = r.parse_hand("JS AS 9H 9D TC")
UP = r.parse_card("9S")


def a_setup(seat=0, dealer=3, up_card=UP, hand=None, player_eval_sims=1,
            bid_eval_sims=None, pass_model=h.players.PASS_ZERO,
            allow_loners=True, stick=False, seed=0):
    return h.Setup(
        hand=tuple(HAND if hand is None else hand),
        up_card=up_card, seat=seat, dealer=dealer,
        player_eval_sims=player_eval_sims,
        bid_eval_sims=(player_eval_sims if bid_eval_sims is None
                       else bid_eval_sims),
        pass_model=pass_model, allow_loners=allow_loners, stick=stick,
        seed=seed)


def quietly(fn, *args, **kwargs):
    """Run something that prints, and give back what it returned."""
    with contextlib.redirect_stdout(io.StringIO()) as out:
        value = fn(*args, **kwargs)
    return value, out.getvalue()


class TestTheQuestionStaysPinned(unittest.TestCase):
    """A sampled deal must be the question that was asked, plus randomness."""

    def test_the_hand_lands_at_the_asking_seat(self):
        for seat in range(game.PLAYERS):
            setup = a_setup(seat=seat)
            for i in range(6):
                deal = setup.deal(i)
                self.assertEqual(set(deal.hands[seat]), set(HAND),
                                 "seat %d lost its pinned hand on deal %d"
                                 % (seat, i))

    def test_the_up_card_and_dealer_are_the_ones_asked_about(self):
        for dealer in range(game.PLAYERS):
            setup = a_setup(dealer=dealer)
            for i in range(4):
                deal = setup.deal(i)
                self.assertEqual(deal.up_card, UP)
                self.assertEqual(deal.dealer, dealer)

    def test_every_sampled_deal_is_a_sound_deal(self):
        # game.Deal.check() is the card-conservation invariant; deal_around
        # already calls it, so this is a guard against that ever being relaxed.
        setup = a_setup()
        for i in range(10):
            setup.deal(i).check()

    def test_the_other_seats_really_do_move(self):
        setup = a_setup()
        others = {tuple(sorted(setup.deal(i).hands[1])) for i in range(10)}
        self.assertGreater(len(others), 1, "the other seats were never re-dealt")

    def test_an_absent_up_card_is_dealt_rather_than_pinned(self):
        setup = a_setup(up_card=None)
        turned = {setup.deal(i).up_card for i in range(12)}
        self.assertGreater(len(turned), 1)
        for card in turned:
            self.assertNotIn(card, HAND)

    def test_the_seat_speaks_where_the_dealer_says_it_does(self):
        # dealer 3 makes seat 0 eldest; dealer 0 makes it last to speak.
        self.assertEqual(a_setup(dealer=3).deal(0).bidding_order()[0], 0)
        self.assertEqual(a_setup(dealer=0).deal(0).bidding_order()[-1], 0)


class TestADealIsAFunctionOfItsIndex(unittest.TestCase):
    """
    The property the parallel path rests on.

    Nothing is carried between deals: deal `i` is derived from `i` and the seed
    alone, so a worker that sees only deal 7 produces exactly what a serial run
    produces at deal 7.
    """

    def test_the_same_index_gives_the_same_deal(self):
        setup = a_setup()
        for i in range(5):
            self.assertEqual(setup.deal(i).hands, setup.deal(i).hands)

    def test_different_indices_give_different_deals(self):
        setup = a_setup()
        seen = {setup.deal(i).hands for i in range(12)}
        self.assertEqual(len(seen), 12)

    def test_the_seed_moves_the_whole_sweep(self):
        self.assertNotEqual(a_setup(seed=0).deal(0).hands,
                            a_setup(seed=1000).deal(0).hands)

    def test_a_played_deal_is_reproducible(self):
        setup = a_setup()
        self.assertEqual(h.play_one(setup, 3), h.play_one(setup, 3))

    def test_a_solved_deal_is_reproducible(self):
        setup = a_setup()
        self.assertEqual(h.solve_one(setup, 3), h.solve_one(setup, 3))

    def test_a_sweep_is_reproducible(self):
        setup = a_setup()
        first, _ = h.run(setup, 4)
        second, _ = h.run(setup, 4)
        self.assertEqual(first, second)


class TestRoles(unittest.TestCase):
    """Who ended up with the contract, said relative to the asking seat."""

    def test_from_an_even_seat(self):
        self.assertEqual(h.role_of(0, 0), "you called")
        self.assertEqual(h.role_of(2, 0), "partner called")
        self.assertEqual(h.role_of(1, 0), "opponent called")
        self.assertEqual(h.role_of(3, 0), "opponent called")

    def test_from_an_odd_seat(self):
        self.assertEqual(h.role_of(1, 1), "you called")
        self.assertEqual(h.role_of(3, 1), "partner called")
        self.assertEqual(h.role_of(0, 1), "opponent called")
        self.assertEqual(h.role_of(2, 1), "opponent called")

    def test_no_caller_is_a_passed_out_deal(self):
        self.assertEqual(h.role_of(-1, 0), "passed out")
        self.assertTrue(h.PASSED_OUT.passed_out)
        self.assertEqual(h.PASSED_OUT.value, 0)

    def test_every_role_is_one_of_the_four_reported(self):
        for caller in range(-1, game.PLAYERS):
            for seat in range(game.PLAYERS):
                self.assertIn(h.role_of(caller, seat), h.ROLES)


class TestTheSeatsScale(unittest.TestCase):
    """
    Every reported number is on the asking seat's own team's scale.

    The referee answers in net points to team 0. Re-expressing that for an odd
    seat is a sign flip, and a sign error survives every deal where the two
    teams happen to agree -- so it is derived here from the caller's score
    rather than compared against the same conversion that produced it.
    """

    def test_a_played_record_agrees_with_the_caller_score(self):
        for seat in range(game.PLAYERS):
            setup = a_setup(seat=seat)
            for i in range(3):
                rec = h.play_one(setup, i)
                if rec.passed_out:
                    continue
                same_team = rec.caller % 2 == seat % 2
                want = rec.caller_score if same_team else -rec.caller_score
                self.assertEqual(
                    rec.value, want,
                    "seat %d, deal %d: caller %d scored %+d, seat read %+d"
                    % (seat, i, rec.caller, rec.caller_score, rec.value))

    def test_a_played_record_is_structurally_sound(self):
        setup = a_setup()
        for i in range(4):
            rec = h.play_one(setup, i)
            if rec.passed_out:
                continue
            self.assertIn(rec.caller_tricks, range(game.HAND_SIZE + 1))
            self.assertIn(rec.caller_score, (-2, 1, 2, 4))
            if rec.alone:
                # A loner is worth one of {-2, 1, 4} and never 2.
                self.assertNotEqual(rec.caller_score, 2)
            self.assertIn(rec.trump, r.SUITS)

    def test_solve_one_is_the_god_mode_auction_on_the_seats_scale(self):
        # Recomputed from solve_bidding rather than trusted, which also pins
        # that setup.deal(i) hands back the same deal it did inside solve_one.
        for seat in (0, 1):
            setup = a_setup(seat=seat)
            for i in range(3):
                rec = h.solve_one(setup, i)
                out = b.solve_bidding(setup.deal(i), allow_loners=True)
                self.assertEqual(rec.passed_out, out.passed_out)
                self.assertEqual(rec.value, b.value_to(seat, out.value))
                if not out.passed_out:
                    self.assertEqual(rec.caller, out.contract.caller)
                    self.assertEqual(rec.trump, out.contract.trump)


class TestInterval(unittest.TestCase):
    """The reported mean and its 95% half-width."""

    def test_nothing_at_all(self):
        self.assertEqual(h.interval([]), (0.0, 0.0))

    def test_one_reading_has_no_interval(self):
        self.assertEqual(h.interval([2]), (2.0, 0.0))

    def test_a_constant_has_no_spread(self):
        mean, half = h.interval([1] * 20)
        self.assertEqual(mean, 1.0)
        self.assertEqual(half, 0.0)

    def test_the_mean_and_half_width(self):
        mean, half = h.interval([-2, -2, 2, 2])
        self.assertEqual(mean, 0.0)
        # sample sd is 2.309..., so the half-width is 1.96 * sd / sqrt(4).
        self.assertAlmostEqual(half, 1.96 * (16 / 3.0 / 4) ** 0.5)

    def test_the_interval_tightens_with_more_readings(self):
        _, few = h.interval([-2, 2] * 5)
        _, many = h.interval([-2, 2] * 500)
        self.assertLess(many, few)


class TestReporting(unittest.TestCase):
    """The printed report, checked for the numbers rather than the layout."""

    def setUp(self):
        self.setup = a_setup()
        self.pimc, self.god = h.run(self.setup, 4, both=True)

    def test_report_gives_back_the_mean_of_the_records(self):
        (mean, half), text = quietly(h.report, "x", self.pimc, self.setup)
        self.assertEqual((mean, half),
                         h.interval([rec.value for rec in self.pimc]))
        self.assertIn("EV to your team", text)

    def test_the_roles_account_for_every_deal(self):
        _, text = quietly(h.report, "x", self.pimc, self.setup)
        counted = sum(int(line.split("(")[0].split()[-1])
                      for line in text.splitlines()
                      if any(line.strip().startswith(role) for role in h.ROLES))
        self.assertEqual(counted, len(self.pimc))

    def test_paired_is_the_difference_deal_by_deal(self):
        (mean, _), _ = quietly(h.paired, self.pimc, self.god)
        want = sum(a.value - c.value for a, c in zip(self.pimc, self.god))
        self.assertAlmostEqual(mean, want / len(self.pimc))

    def test_both_sweeps_cover_the_same_layouts(self):
        self.assertEqual(len(self.pimc), len(self.god))

    def test_one_sweep_without_both_solves_nothing(self):
        pimc, god = h.run(self.setup, 2)
        self.assertEqual(len(pimc), 2)
        self.assertEqual(god, [])


class TestTheCommandLine(unittest.TestCase):
    """Argument handling, including the rejections that protect the sweep."""

    def test_the_defaults(self):
        args = h.parse_args(["JS AS 9H 9D TC"])
        self.assertEqual(args.pass_model, h.players.PASS_ZERO)
        self.assertEqual(args.deals, h.DEALS)
        self.assertEqual(args.player_eval_sims, h.PLAYER_EVAL_SIMS)

    def test_bid_eval_sims_follows_player_eval_sims(self):
        setup = h.setup_from(h.parse_args(
            ["JS AS 9H 9D TC", "--player-eval-sims", "7"]))
        self.assertEqual(setup.player_eval_sims, 7)
        self.assertEqual(setup.bid_eval_sims, 7)

    def test_bid_eval_sims_can_be_set_apart(self):
        setup = h.setup_from(h.parse_args(
            ["JS AS 9H 9D TC", "--player-eval-sims", "7",
             "--bid-eval-sims", "2"]))
        self.assertEqual(setup.player_eval_sims, 7)
        self.assertEqual(setup.bid_eval_sims, 2)

    def test_the_hand_and_up_card_are_parsed(self):
        setup = h.setup_from(h.parse_args(["JS AS 9H 9D TC", "--up", "9S"]))
        self.assertEqual(setup.hand, tuple(HAND))
        self.assertEqual(setup.up_card, UP)

    def test_loners_are_on_unless_turned_off(self):
        self.assertTrue(h.setup_from(h.parse_args(["JS AS 9H 9D TC"]))
                        .allow_loners)
        self.assertFalse(h.setup_from(h.parse_args(
            ["JS AS 9H 9D TC", "--no-loners"])).allow_loners)

    def test_a_short_hand_is_refused(self):
        with self.assertRaises(SystemExit):
            h.setup_from(h.parse_args(["JS AS 9H"]))

    def test_an_up_card_already_in_hand_is_refused(self):
        # deal_around would raise "the same card was pinned twice" deals later;
        # catching it here names the actual mistake.
        with self.assertRaises(SystemExit):
            h.setup_from(h.parse_args(["JS AS 9H 9D TC", "--up", "JS"]))

    def test_a_seat_off_the_table_is_refused(self):
        for bad in (["--seat", "4"], ["--dealer", "-1"]):
            with self.assertRaises(SystemExit):
                h.setup_from(h.parse_args(["JS AS 9H 9D TC"] + bad))

    def test_an_unknown_pass_model_is_refused(self):
        # argparse prints its usage to stderr on the way out; swallowed so a
        # passing suite stays quiet.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                h.parse_args(["JS AS 9H 9D TC", "--pass-model", "vibes"])


class TestWorkersDoNotChangeTheAnswer(unittest.TestCase):
    """
    A sweep across processes gives the same records, in deal order.

    The end-to-end form of "a deal is a function of its index". Both tests here
    check the records against `play_one` index by index rather than against each
    other: the pool hands results back as they finish, so comparing two runs
    only catches a scramble if one of them happened to be scrambled and the
    other did not. Anchoring to `play_one(setup, i)` catches it either way.

    This costs a process pool and a JIT warmup per worker, which is most of this
    file's runtime. It earns it -- every cheaper test in this file passes
    happily while `run` returns the right records in the wrong order.
    """

    DEALS = 12          # several chunks, so the pool can interleave them

    def test_serial_records_are_in_deal_order(self):
        setup = a_setup()
        records, _ = h.run(setup, self.DEALS, workers=1)
        self.assertEqual(records,
                         [h.play_one(setup, i) for i in range(self.DEALS)])

    def test_parallel_records_are_in_deal_order(self):
        setup = a_setup()
        records, _ = h.run(setup, self.DEALS, workers=3)
        self.assertEqual(records,
                         [h.play_one(setup, i) for i in range(self.DEALS)])


if __name__ == "__main__":
    unittest.main()
