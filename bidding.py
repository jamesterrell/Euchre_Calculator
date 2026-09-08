"""
Bidding, solved under perfect knowledge.

This is the baseline, not the destination. Every seat here sees all four hands
and picks the bid that is genuinely best, which no real player can do. What it
gives you is a *correct* answer to "was this hand worth ordering up", against
which heuristic bidders can later be measured. Replace the decision rule, keep
the machinery.

The bidding tree is small enough to solve exactly. Round one is a chain of four
order-or-pass decisions, and an order ends it; round two is a chain of four
name-a-suit-or-pass decisions. Each leaf is one double-dummy trick-play solve.
That comes to at most 36 solves per deal -- 24 for round one, since ordering up
forces the dealer to choose among six discards, and 12 for round two -- so a
whole bidding solve costs well under a tenth of a second.

Scoring is **net points to team 0** (seats 0 and 2) throughout, so that one
number can be maximised and minimised on a single scale:

    caller on team 0, makes it   ->  +1, or +2 for a march
    caller on team 0, euchred    ->  -2   (team 1 scores 2)
    caller on team 1, makes it   ->  -1, or -2
    caller on team 1, euchred    ->  +2
    passed out                   ->   0

Two details that a looser implementation gets wrong:

  * The **dealer** chooses the discard, not the caller. When the opposing team
    orders it up, the dealer is picking up a card for a contract they want to
    fail, and will throw whatever hurts the caller most. That is real Euchre,
    and it is minimaxed here rather than assumed away.
  * Passing is not free. Its value is whatever the *rest* of the bidding
    produces, which may be the opponents naming a suit that is worse for you
    than the call you declined.

Loners are not modelled yet: a call is always four-handed. That is the next
structural piece, and it belongs in the play engine rather than here.
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

    @property
    def caller_team(self) -> int:
        return team_of(self.caller)

    def solve(self) -> int:
        """Trick-play value from the calling team's side."""
        return play_value(self.deal, self.trump, self.caller)

    def __str__(self) -> str:
        how = "ordered up" if self.bidding_round == ROUND_ONE else "named"
        text = "seat %d %s %s" % (self.caller, how, r.suit_name(self.trump))
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


def play_value(deal: Deal, trump: int, caller: int) -> int:
    """
    Double-dummy trick-play value of a settled deal, from the caller's side.

    Play always begins to the dealer's left, whoever called.
    """
    hands = r.deal_to_engine(deal.hands, trump)
    return definitive_winner(hands, deal.first_bidder, caller)


def _prefers(candidate: int, incumbent: int, seat: int) -> bool:
    """Does `seat` prefer `candidate` to `incumbent`, on the team-0 scale?"""
    return candidate > incumbent if team_of(seat) == 0 else candidate < incumbent


def _best(options, seat):
    """
    The option `seat` prefers, on the team-0 scale.

    Ties keep the first option, and callers list passing first, so a seat that
    gains nothing by bidding will pass. Without that, perfect knowledge happily
    orders up a hand it knows will be euchred whenever declining is equally
    bad -- the value is the same, but the reported line is nonsense.
    """
    best = options[0]
    for option in options[1:]:
        if _prefers(option[0], best[0], seat):
            best = option
    return best


def order_up(deal: Deal, caller: int) -> Tuple[int, Contract]:
    """
    `caller` orders up the turned suit; the dealer picks up and discards.

    The discard is the *dealer's* decision, so it is chosen for the dealer's
    team -- which is the caller's opponent whenever the two are on opposite
    sides. Returns (net points to team 0, the resulting contract).
    """
    trump = deal.up_card.suit
    dealer = deal.dealer
    options = []
    for card in list(deal.hands[dealer]) + [deal.up_card]:
        after = deal.pick_up(discard=card)
        value = net_to_team0(play_value(after, trump, caller), caller)
        options.append((value, Contract(trump, caller, ROUND_ONE, after, card)))
    return _best(options, dealer)


def name_suit(deal: Deal, caller: int, trump: int) -> Tuple[int, Contract]:
    """`caller` names `trump` in round two. The up-card stays turned down."""
    if trump == deal.up_card.suit:
        raise ValueError("the turned suit cannot be named in round two")
    value = net_to_team0(play_value(deal, trump, caller), caller)
    return value, Contract(trump, caller, ROUND_TWO, deal)


def _round_two(deal, index, order, stick_the_dealer):
    if index == PLAYERS:
        return Outcome(None, 0, ("all pass",))

    seat = order[index]
    is_dealer = (index == PLAYERS - 1)
    options = []

    # Passing goes first so that ties resolve to passing; under
    # stick-the-dealer the last seat has no such option.
    if not (stick_the_dealer and is_dealer):
        passed = _round_two(deal, index + 1, order, stick_the_dealer)
        options.append((passed.value,
                        Outcome(passed.contract, passed.value,
                                ("seat %d passes" % seat,) + passed.line)))

    for trump in r.SUITS:
        if trump == deal.up_card.suit:
            continue
        value, contract = name_suit(deal, seat, trump)
        options.append((value, Outcome(contract, value,
                                       ("seat %d names %s"
                                        % (seat, r.suit_name(trump)),))))

    return _best(options, seat)[1]


def _round_one(deal, index, order, stick_the_dealer):
    if index == PLAYERS:
        return _round_two(deal, 0, order, stick_the_dealer)

    seat = order[index]

    value, contract = order_up(deal, seat)
    ordered = Outcome(contract, value,
                      ("seat %d orders up %s"
                       % (seat, r.suit_name(deal.up_card.suit)),))

    passed = _round_one(deal, index + 1, order, stick_the_dealer)
    passed = Outcome(passed.contract, passed.value,
                     ("seat %d passes" % seat,) + passed.line)

    # Passing first, so a seat with nothing to gain declines rather than
    # ordering up a contract it knows will be euchred.
    return _best([(passed.value, passed), (ordered.value, ordered)], seat)[1]


def solve_bidding(deal: Deal, stick_the_dealer: bool = False) -> Outcome:
    """
    Solve the whole auction under perfect knowledge.

    Every seat sees every hand and bids to maximise its own team's net points,
    knowing how the rest of the auction and the play will go. Returns the
    contract that survives, or a passed-out Outcome worth 0.

    Args:
        deal: a freshly dealt hand, before any pickup.
        stick_the_dealer: if True, the dealer may not pass in round two.
    """
    if deal.picked_up:
        raise ValueError("bidding starts before the up-card is picked up")
    return _round_one(deal, 0, deal.bidding_order(), stick_the_dealer)


def first_bid_choice(deal: Deal) -> Tuple[int, int]:
    """
    What the eldest hand is choosing between: (value of ordering, of passing).

    Both are net points **to the first bidder's own team**, not to team 0, so
    the larger number is simply the better bid. This is the "should I order
    this up?" question in its smallest form.
    """
    seat = deal.first_bidder
    order = deal.bidding_order()
    ordered, _ = order_up(deal, seat)
    passed = _round_one(deal, 1, order, False)
    return value_to(seat, ordered), value_to(seat, passed.value)
