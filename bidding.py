"""
Bidding, solved in God Mode.

The baseline, not the destination: every seat sees all four hands and picks the
bid that is genuinely best, which no real player can do. What it gives you is a
*correct* answer to "was this hand worth ordering up", against which heuristic
bidders can be measured. Replace the decision rule, keep the machinery.

The tree is small enough to solve exactly. Round one is a chain of four
order-or-pass decisions and an order ends it; round two is four
name-a-suit-or-pass decisions. Each leaf is one God Mode trick-play solve, so
at most 36 solves per deal -- 24 for round one, since ordering up forces the
dealer to choose among six discards, plus 12 for round two. Well under a tenth
of a second.

Scoring is **net points to team 0** (seats 0 and 2) throughout, so that one
number can be maximised and minimised on a single scale:

    caller on team 0, makes it   ->  +1, or +2 for a march (+4 alone)
    caller on team 0, euchred    ->  -2   (team 1 scores 2)
    caller on team 1, makes it   ->  -1, or -2 (-4 alone)
    caller on team 1, euchred    ->  +2
    passed out                   ->   0

**Loners** are off by default: `allow_loners=True` adds "and alone" beside
every call, doubling the tree to ~72 solves per auction. Lone solves are far
cheaper, so wall-clock cost rises well under double. Defending alone is not
modelled.

Three details a looser implementation gets wrong:

  * The **dealer** chooses the discard, not the caller. Ordered up by the
    opposition, the dealer is taking a card into a contract they want to fail
    and throws whatever hurts the caller most. Minimaxed here, not assumed away.
  * **Passing is not free.** Its value is whatever the *rest* of the bidding
    produces, which may be worse for you than the call you declined.
  * **Going alone is not free either, and ties resolve against it.** A loner
    worth no more than the same call four-handed is the same nonsense as
    ordering up a hand you know will be euchred. Options are listed pass, call,
    call-alone, and `_best` keeps the first of equals.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import rotation as r
from fast_search import definitive_winner
from game import PLAYERS, Deal

ROUND_ONE = 1
ROUND_TWO = 2

PASS = "pass"
ORDER = "order"
NAME = "name"


def team_of(seat: int) -> int:
    """Seats 0 and 2 are team 0; seats 1 and 3 are team 1."""
    return seat % 2


def net_to_team0(caller_score: int, caller: int) -> int:
    """
    Convert a caller's-perspective trick-play score into net points to team 0.

    `definitive_winner` answers from the calling team's side, which flips
    meaning depending on who called. Bidding has to compare calls by different
    seats, so everything is put on one scale.
    """
    return caller_score if team_of(caller) == 0 else -caller_score


def value_to(seat: int, team0_value: int) -> int:
    """Re-express a team-0 net value from `seat`'s own team's point of view."""
    return team0_value if team_of(seat) == 0 else -team0_value


@dataclass(frozen=True)
class Contract:
    """A settled call: who called what, and the deal it will be played from."""

    trump: int
    caller: int
    bidding_round: int
    deal: Deal
    discard: Optional[r.Card] = None
    alone: bool = False

    @property
    def caller_team(self) -> int:
        return team_of(self.caller)

    @property
    def sitting(self) -> Optional[int]:
        """The caller's partner, if it is sitting out; None otherwise."""
        return (self.caller + 2) % PLAYERS if self.alone else None

    def solve(self) -> int:
        """Trick-play value from the calling team's side."""
        return play_value(self.deal, self.trump, self.caller, self.alone)

    def __str__(self) -> str:
        how = "ordered up" if self.bidding_round == ROUND_ONE else "named"
        text = "seat %d %s %s" % (self.caller, how, r.suit_name(self.trump))
        if self.alone:
            text += " alone (seat %d sits out)" % self.sitting
        if self.discard is not None:
            text += " (dealer pitched %s)" % r.card_name(self.discard)
        return text


@dataclass(frozen=True)
class Outcome:
    """The result of solving the bidding."""

    contract: Optional[Contract]
    value: int                       # net points to team 0
    line: Tuple[str, ...] = ()       # the bids taken, for readability

    @property
    def passed_out(self) -> bool:
        return self.contract is None

    def __str__(self) -> str:
        if self.passed_out:
            return "passed out (0)"
        return "%s -> %+d to team 0" % (self.contract, self.value)


def play_value(deal: Deal, trump: int, caller: int, alone: bool = False) -> int:
    """
    God Mode trick-play value of a settled deal, from the caller's side.

    Play always begins to the dealer's left, whoever called. With `alone` the
    caller's partner sits out, and if that partner is the eldest hand the lead
    passes to the next live seat -- the solver handles it.
    """
    hands = r.deal_to_engine(deal.hands, trump)
    return definitive_winner(hands, deal.first_bidder, caller, alone=alone)


def _prefers(candidate: int, incumbent: int, seat: int) -> bool:
    """Does `seat` prefer `candidate` to `incumbent`, on the team-0 scale?"""
    return candidate > incumbent if team_of(seat) == 0 else candidate < incumbent


def _best(options, seat):
    """
    The option `seat` prefers, on the team-0 scale.

    Ties keep the first option and callers list passing first, so a seat that
    gains nothing by bidding passes. Without that, God Mode cheerfully orders
    up a hand it knows will be euchred whenever declining is equally bad -- the
    value is the same, but the reported line is nonsense.
    """
    best = options[0]
    for option in options[1:]:
        if _prefers(option[0], best[0], seat):
            best = option
    return best


def order_up(deal: Deal, caller: int, alone: bool = False) -> Tuple[int, Contract]:
    """
    `caller` orders up the turned suit; the dealer picks up and discards.

    The discard is the *dealer's* decision, so it is chosen for the dealer's
    team -- which is the caller's opponent whenever the two are on opposite
    sides. Returns (net points to team 0, the resulting contract).

    The dealer chooses among the **five cards it was dealt**. The up-card is not
    a candidate: ordered up, it is in the dealer's hand to stay -- see
    `game.Deal.pick_up`. That is one fewer God Mode solve per order, so round
    one costs 20 solves rather than 24.

    One case collapses: if `caller` goes alone and the dealer is the partner
    sitting out, the dealer's whole hand leaves play, so every discard is worth
    exactly the same and the choice is unobservable. Solving all five would be
    five identical answers, so the first dealt card is pitched by convention and
    one solve is done. `tests/test_loners.py` checks the five really do agree.
    """
    trump = deal.up_card.suit
    dealer = deal.dealer
    sitting = (caller + 2) % PLAYERS if alone else None

    if sitting == dealer:
        after = deal.pick_up(discard=deal.hands[dealer][0])
        value = net_to_team0(play_value(after, trump, caller, alone), caller)
        return value, Contract(trump, caller, ROUND_ONE, after,
                               deal.hands[dealer][0], alone)

    options = []
    for card in deal.hands[dealer]:
        after = deal.pick_up(discard=card)
        value = net_to_team0(play_value(after, trump, caller, alone), caller)
        options.append((value,
                        Contract(trump, caller, ROUND_ONE, after, card, alone)))
    return _best(options, dealer)


def name_suit(deal: Deal, caller: int, trump: int,
              alone: bool = False) -> Tuple[int, Contract]:
    """`caller` names `trump` in round two. The up-card stays turned down."""
    if trump == deal.up_card.suit:
        raise ValueError("the turned suit cannot be named in round two")
    value = net_to_team0(play_value(deal, trump, caller, alone), caller)
    return value, Contract(trump, caller, ROUND_TWO, deal, None, alone)


def _alone_note(alone):
    return " alone" if alone else ""


def _round_two(deal, index, order, stick_the_dealer, allow_loners):
    if index == PLAYERS:
        return Outcome(None, 0, ("all pass",))

    seat = order[index]
    is_dealer = (index == PLAYERS - 1)
    options = []

    # Passing goes first so that ties resolve to passing; under
    # stick-the-dealer the last seat has no such option.
    if not (stick_the_dealer and is_dealer):
        passed = _round_two(deal, index + 1, order, stick_the_dealer,
                            allow_loners)
        options.append((passed.value,
                        Outcome(passed.contract, passed.value,
                                ("seat %d passes" % seat,) + passed.line)))

    for trump in r.SUITS:
        if trump == deal.up_card.suit:
            continue
        # Four-handed first, so a loner that gains nothing is declined.
        for alone in (False, True) if allow_loners else (False,):
            value, contract = name_suit(deal, seat, trump, alone)
            options.append((value, Outcome(contract, value,
                                           ("seat %d names %s%s"
                                            % (seat, r.suit_name(trump),
                                               _alone_note(alone)),))))

    return _best(options, seat)[1]


def _round_one(deal, index, order, stick_the_dealer, allow_loners):
    if index == PLAYERS:
        return _round_two(deal, 0, order, stick_the_dealer, allow_loners)

    seat = order[index]

    passed = _round_one(deal, index + 1, order, stick_the_dealer, allow_loners)
    passed = Outcome(passed.contract, passed.value,
                     ("seat %d passes" % seat,) + passed.line)

    # Passing first, then the four-handed call, then the loner: a seat with
    # nothing to gain declines rather than ordering up a contract it knows will
    # be euchred, and one that gains nothing by sitting its partner down keeps
    # the partner in.
    options = [(passed.value, passed)]
    for alone in (False, True) if allow_loners else (False,):
        value, contract = order_up(deal, seat, alone)
        options.append((value, Outcome(contract, value,
                                       ("seat %d orders up %s%s"
                                        % (seat, r.suit_name(deal.up_card.suit),
                                           _alone_note(alone)),))))

    return _best(options, seat)[1]


def solve_bidding(deal: Deal, stick_the_dealer: bool = False,
                  allow_loners: bool = False) -> Outcome:
    """
    Solve the whole auction in God Mode.

    Every seat sees every hand and bids to maximise its own team's net points,
    knowing how the rest of the auction and the play will go. Returns the
    surviving contract, or a passed-out Outcome worth 0.

    Args:
        deal: a freshly dealt hand, before any pickup.
        stick_the_dealer: if True, the dealer may not pass in round two.
        allow_loners: if True, every call may also be made alone. Off by
            default so existing four-handed measurements stay comparable. It
            changes the auction on ~1% of deals (6 of 480), always by turning a
            made contract into a lone march. It is an extra option for *both*
            teams, so no direction is guaranteed either way.
    """
    if deal.picked_up:
        raise ValueError("bidding starts before the up-card is picked up")
    return _round_one(deal, 0, deal.bidding_order(), stick_the_dealer,
                      allow_loners)


def first_bid_choice(deal: Deal, allow_loners: bool = False) -> Tuple[int, int]:
    """
    What the eldest hand is choosing between: (value of ordering, of passing).

    Both are net points **to the first bidder's own team**, not to team 0, so
    the larger number is simply the better bid. This is the "should I order
    this up?" question in its smallest form. `first_bid_options` is the same
    question with the loner spelled out as its own option.

    `allow_loners` affects only the *passing* branch -- whether the seats after
    this one may go alone. The ordering value is always the four-handed call,
    which is what "should I order this up?" asks.
    """
    seat = deal.first_bidder
    order = deal.bidding_order()
    ordered, _ = order_up(deal, seat)
    passed = _round_one(deal, 1, order, False, allow_loners)
    return value_to(seat, ordered), value_to(seat, passed.value)


def first_bid_options(deal: Deal, allow_loners: bool = True) -> dict:
    """
    Every first-bid option open to the eldest hand, on its own team's scale.

    Returns {"pass": v, "order": v, "order alone": v} -- net points to the
    first bidder's team, so the largest number is simply the best bid. The
    "order alone" entry is dropped when `allow_loners` is False.

    This is the question a calculator front end actually asks. The loner
    usually loses it -- over 32 measured deals the eldest hand never gained by
    going alone, and lost by it on 8 -- but when it wins it wins by two points,
    which is exactly why it is worth showing rather than assuming.
    """
    seat = deal.first_bidder
    order = deal.bidding_order()
    passed = _round_one(deal, 1, order, False, allow_loners)

    options = {"pass": value_to(seat, passed.value),
               "order": value_to(seat, order_up(deal, seat)[0])}
    if allow_loners:
        options["order alone"] = value_to(seat, order_up(deal, seat, True)[0])
    return options


# ------------------------------------------------- entry points for players
#
# `solve_bidding` runs the whole auction itself, which is what the baseline
# wants and exactly what a table of independent players cannot use: there the
# auction is driven one seat at a time from outside, and a seat that is
# considering passing needs the value of *the rest* of it. These expose the two
# pieces `table.py` needs without reaching into the recursion.


def rest_of_auction(deal: Deal, index: int, order=None,
                    stick_the_dealer: bool = False,
                    allow_loners: bool = False,
                    bidding_round: int = ROUND_ONE) -> Outcome:
    """
    The auction from `index` onward, solved in God Mode.

    This is what passing is worth. A seat that declines does not get zero -- it
    gets whatever the remaining seats do, which can be worse than the call it
    turned down. That is why passing is priced rather than assumed free.

    Args:
        deal: the deal, before any pickup.
        index: how far through `order` the auction has already got.
        order: the bidding order; `deal.bidding_order()` by default.
        bidding_round: ROUND_ONE or ROUND_TWO. In round two `index` counts
            from the start of round two, not from the start of the auction.
    """
    order = list(order if order is not None else deal.bidding_order())
    if bidding_round == ROUND_ONE:
        return _round_one(deal, index, order, stick_the_dealer, allow_loners)
    if bidding_round == ROUND_TWO:
        return _round_two(deal, index, order, stick_the_dealer, allow_loners)
    raise ValueError("no such bidding round: %r" % (bidding_round,))


def best_discard(deal: Deal, caller: int, alone: bool = False):
    """
    The card the dealer pitches on being ordered up, in God Mode.

    Chosen for the *dealer's* team, which is the point: ordered up by the
    opposition, it is taking a card into a contract it wants to fail.
    """
    return order_up(deal, caller, alone)[1].discard
