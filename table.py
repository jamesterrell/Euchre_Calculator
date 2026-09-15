"""
Play a deal out with four independent players.

Everything above this module answers the deal *at once*: `solve_bidding` runs
the whole auction in one call, and `definitive_winner` returns a score without
naming a card anybody chose. Right shape for a baseline, wrong shape for "what
happens when nobody can see the other hands" -- a player who cannot see them
decides one step at a time, on what it knows then, and gets to be wrong.

So this is a referee. It holds the truth, asks each seat in turn what it wants
to do, checks the answer is legal, and writes down what happened. It holds no
strategy -- every decision comes from a player object in `players.py`. Swap the
players and the same loop gives a God Mode table, a PIMC sim table, or a table
of coin flips.

    >>> import random, game, players
    >>> d = game.deal_random(rng=random.Random(0), dealer=3)
    >>> table = [players.PIMCPlayer(rng=random.Random(i)) for i in range(4)]
    >>> result = play_deal(d, table)            # doctest: +SKIP
    >>> result.value                            # net points to team 0
    1

A player is any object with `bid`, `discard` and `play` methods. Each is handed
a turn object carrying both the true `Deal` and that seat's `Observation`, and
which of the two it reads is the entire difference between a God Mode opponent
and an honest one. **A player that reads `turn.deal` is cheating by
definition** -- `GodModePlayer` does, on purpose.

The trick-winner rule is not reimplemented here. Cards go through
`rotation.card_to_engine`, already tested as a bijection onto the solver's
frame, and are compared there -- so the left bower is trump for ordering
because the encoding says so, not because this module remembered to check.
`tests/test_table.py` pins it against the solver's own `_resolve`.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import bidding as b
import observation as obs
import rotation as r
from fast_search import _final
from game import HAND_SIZE, PLAYERS, Deal

PASS = "pass"
ORDER = "order"
NAME = "name"

# The solver's suit codes, which rotation.card_to_engine lands cards on.
_TRUMP_CODE = 1


@dataclass(frozen=True)
class Bid:
    """One seat's decision in the auction."""

    action: str                      # PASS, ORDER or NAME
    suit: Optional[int] = None       # the named suit, for NAME
    alone: bool = False

    def __str__(self) -> str:
        if self.action == PASS:
            return "pass"
        what = "orders up" if self.action == ORDER else "names"
        return "%s %s%s" % (what, r.suit_name(self.suit),
                            " alone" if self.alone else "")


@dataclass(frozen=True)
class BidTurn:
    """A seat's turn to bid, with the options open to it."""

    deal: Deal
    observation: obs.Observation
    seat: int
    bidding_round: int
    index: int                       # how far through the bidding order
    order: Tuple[int, ...]
    options: Tuple[Bid, ...]
    stick_the_dealer: bool
    allow_loners: bool


@dataclass(frozen=True)
class DiscardTurn:
    """The dealer's turn to pitch a card after picking up the up-card."""

    deal: Deal                       # already picked up; the dealer holds six
    before: Deal                     # the same deal before the pickup
    observation: obs.Observation
    seat: int
    caller: int
    trump: int
    alone: bool
    options: Tuple[r.Card, ...]


@dataclass(frozen=True)
class PlayTurn:
    """A seat's turn to play a card."""

    deal: Deal                       # the settled deal, after any pickup
    observation: obs.Observation
    seat: int
    contract: b.Contract
    legal: Tuple[r.Card, ...]
    plays: Tuple[Tuple[int, r.Card], ...]
    caller_tricks: int
    trick_no: int

    @property
    def trump(self) -> int:
        return self.contract.trump

    @property
    def caller(self) -> int:
        return self.contract.caller

    @property
    def alone(self) -> bool:
        return self.contract.alone


@dataclass(frozen=True)
class Result:
    """What happened when a deal was played out."""

    deal: Deal                                   # as dealt, before the pickup
    contract: Optional[b.Contract]
    value: int                                   # net points to team 0
    caller_score: int                            # from the calling team's side
    caller_tricks: int
    plays: Tuple[Tuple[int, r.Card], ...] = ()
    winners: Tuple[int, ...] = ()
    auction: Tuple[str, ...] = ()

    @property
    def passed_out(self) -> bool:
        return self.contract is None

    @property
    def euchred(self) -> bool:
        return self.contract is not None and self.caller_score < 0

    def __str__(self) -> str:
        if self.passed_out:
            return "passed out (0)"
        return ("%s -> %d trick%s, %+d to the caller, %+d to team 0"
                % (self.contract, self.caller_tricks,
                   "" if self.caller_tricks == 1 else "s",
                   self.caller_score, self.value))


# ------------------------------------------------------------ the card rules


def card_order(card: r.Card, trump: int) -> Tuple[int, int]:
    """
    A card's (suit code, strength) in the solver's frame.

    Routed through `rotation.card_to_engine` rather than read off the natural
    card, so the left bower sorts as second-highest trump for the same reason
    it does inside the search. Mirrors `fast_search.encode_hands` exactly.
    """
    x, y = r.card_to_engine(card, trump)
    if x > 0:
        return 0, x
    if x < 0:
        return 2, -x
    if y > 0:
        return _TRUMP_CODE, y
    return 3, -y


def trick_winner(plays: Sequence[Tuple[int, r.Card]], trump: int) -> int:
    """Highest trump if any was played, else the highest card of the led suit."""
    if not plays:
        raise ValueError("an empty trick has no winner")
    ordered = [(seat, card_order(card, trump)) for seat, card in plays]
    trumps = [(seat, s) for seat, (c, s) in ordered if c == _TRUMP_CODE]
    if trumps:
        return max(trumps, key=lambda p: p[1])[0]
    led = ordered[0][1][0]
    followed = [(seat, s) for seat, (c, s) in ordered if c == led]
    return max(followed, key=lambda p: p[1])[0]


def legal_cards(hand: Sequence[r.Card], led: Optional[r.Card],
                trump: int) -> Tuple[r.Card, ...]:
    """
    The cards a seat may play. Follow suit if you can; otherwise anything.

    Following is the only restriction in Euchre -- no obligation to win, to
    trump, or to play high. Everything else is choice, which is why this
    returns a set rather than a card.
    """
    hand = tuple(hand)
    if led is None:
        return hand
    want = card_order(led, trump)[0]
    following = tuple(c for c in hand if card_order(c, trump)[0] == want)
    return following or hand


def next_seat(seat: int, sitting: Optional[int]) -> int:
    """The seat that acts after `seat`, stepping over a loner's partner."""
    nxt = (seat + 1) % PLAYERS
    if sitting is not None and nxt == sitting:
        nxt = (nxt + 1) % PLAYERS
    return nxt


# ------------------------------------------------------------- the auction


def _bid_options(deal: Deal, bidding_round: int, is_last: bool,
                 stick_the_dealer: bool, allow_loners: bool) -> Tuple[Bid, ...]:
    """
    The bids open to a seat, passing first.

    Not cosmetic: players keep the first of equals, so a seat with nothing to
    gain declines rather than calling a contract it expects to lose.
    `bidding._best` does the same, for the same reason.
    """
    options = []
    if not (stick_the_dealer and is_last and bidding_round == b.ROUND_TWO):
        options.append(Bid(PASS))

    alones = (False, True) if allow_loners else (False,)
    if bidding_round == b.ROUND_ONE:
        for alone in alones:
            options.append(Bid(ORDER, deal.up_card.suit, alone))
    else:
        for suit in r.SUITS:
            if suit == deal.up_card.suit:
                continue
            for alone in alones:
                options.append(Bid(NAME, suit, alone))
    return tuple(options)


def _ask_bid(player, turn: BidTurn) -> Bid:
    bid = player.bid(turn)
    if bid not in turn.options:
        raise ValueError("seat %d bid %s, which is not on offer"
                         % (turn.seat, bid))
    return bid


def _settle_order(deal: Deal, caller: int, alone: bool, players) -> b.Contract:
    """The dealer picks up and pitches a card of its own choosing."""
    trump = deal.up_card.suit
    dealer = deal.dealer
    sitting = (caller + 2) % PLAYERS if alone else None

    # The dealer holds six for exactly as long as this decision takes. Deal is
    # frozen and validated, and a six-card hand fails its check by design, so
    # the intermediate is built directly rather than through pick_up -- which
    # is called below, once there is a card to give it.
    options = tuple(deal.hands[dealer]) + (deal.up_card,)
    taken = Deal(hands=tuple(options if s == dealer else tuple(h)
                             for s, h in enumerate(deal.hands)),
                 up_card=deal.up_card, buried=deal.buried, dealer=dealer,
                 picked_up=True)

    turn = DiscardTurn(
        deal=taken,
        before=deal,
        observation=obs.Observation(
            seat=dealer, hand=options, dealer=dealer, up_card=deal.up_card,
            up_state=obs.PICKED_UP, trump=trump, caller=caller, alone=alone,
            pending_discard=True).check(),
        seat=dealer, caller=caller, trump=trump, alone=alone, options=options)

    if sitting == dealer:
        # The dealer's whole hand is about to leave the game, so every discard
        # is worth the same and the choice is unobservable. bidding.order_up
        # short-circuits this for the same reason; here it also spares a player
        # from being asked a question with no answer.
        pitched = deal.up_card
    else:
        pitched = players[dealer].discard(turn)
        if pitched not in options:
            raise ValueError("the dealer cannot pitch %s, which it does not hold"
                             % r.card_name(pitched))

    return b.Contract(trump, caller, b.ROUND_ONE,
                      deal.pick_up(discard=pitched), pitched, alone)


def run_auction(deal: Deal, players, stick_the_dealer: bool = False,
                allow_loners: bool = False):
    """
    Run the auction one seat at a time. Returns (contract or None, bid log).

    Unlike `bidding.solve_bidding` this does not search the tree -- it walks it
    once, taking whatever each player says. A table of God Mode players
    reproduces `solve_bidding` exactly, which is how `tests/test_table.py`
    checks the loop against the thing it generalises.
    """
    if deal.picked_up:
        raise ValueError("bidding starts before the up-card is picked up")

    order = tuple(deal.bidding_order())
    log = []

    for bidding_round in (b.ROUND_ONE, b.ROUND_TWO):
        up_state = obs.UP if bidding_round == b.ROUND_ONE else obs.TURNED_DOWN
        for index, seat in enumerate(order):
            options = _bid_options(deal, bidding_round, index == PLAYERS - 1,
                                   stick_the_dealer, allow_loners)
            turn = BidTurn(
                deal=deal,
                observation=obs.observe(deal, seat, up_state=up_state),
                seat=seat, bidding_round=bidding_round, index=index,
                order=order, options=options,
                stick_the_dealer=stick_the_dealer, allow_loners=allow_loners)
            bid = _ask_bid(players[seat], turn)
            log.append("seat %d %s" % (seat, bid))

            if bid.action == PASS:
                continue
            if bid.action == ORDER:
                return _settle_order(deal, seat, bid.alone, players), tuple(log)
            return (b.Contract(bid.suit, seat, b.ROUND_TWO, deal, None,
                               bid.alone),
                    tuple(log))

    return None, tuple(log)


# ---------------------------------------------------------------- the play


def play_contract(contract: b.Contract, players) -> Tuple[int, tuple, tuple]:
    """
    Play the five tricks out. Returns (caller tricks, plays, trick winners).

    Play always begins to the dealer's left. If that seat is sitting out a
    loner the lead passes to the next live seat -- the same rule
    `fast_search._setup` applies. The lead does not belong to the caller.
    """
    deal = contract.deal
    trump, caller, alone = contract.trump, contract.caller, contract.alone
    sitting = contract.sitting
    width = 3 if alone else 4

    hands = [list(h) for h in deal.hands]
    if sitting is not None:
        hands[sitting] = []

    leader = deal.first_bidder
    if leader == sitting:
        leader = next_seat(leader, sitting)

    plays: List[Tuple[int, r.Card]] = []
    winners: List[int] = []
    caller_tricks = 0

    for trick_no in range(HAND_SIZE):
        trick: List[Tuple[int, r.Card]] = []
        seat = leader
        for _ in range(width):
            led = trick[0][1] if trick else None
            legal = legal_cards(hands[seat], led, trump)
            turn = PlayTurn(
                deal=deal,
                observation=obs.observe(
                    deal, seat, plays=tuple(plays) + tuple(trick), trump=trump,
                    caller=caller, alone=alone,
                    up_state=(obs.PICKED_UP if deal.picked_up
                              else obs.TURNED_DOWN),
                    discard=(contract.discard if seat == deal.dealer
                             and deal.picked_up else None)),
                seat=seat, contract=contract, legal=tuple(legal),
                plays=tuple(plays) + tuple(trick),
                caller_tricks=caller_tricks, trick_no=trick_no)

            card = players[seat].play(turn)
            if card not in legal:
                raise ValueError(
                    "seat %d played %s, which is not legal here (legal: %s)"
                    % (seat, r.card_name(card), r.hand_name(legal)))

            hands[seat].remove(card)
            trick.append((seat, card))
            seat = next_seat(seat, sitting)

        won_by = trick_winner(trick, trump)
        winners.append(won_by)
        plays.extend(trick)
        if won_by % 2 == caller % 2:
            caller_tricks += 1
        leader = won_by

    return caller_tricks, tuple(plays), tuple(winners)


def play_deal(deal: Deal, players, stick_the_dealer: bool = False,
              allow_loners: bool = False) -> Result:
    """
    Run one whole deal -- auction then play -- and score it.

    `players` is four player objects indexed by seat. They need not be alike:
    three PIMC players and one God Mode player is a perfectly good experiment.
    """
    if len(players) != PLAYERS:
        raise ValueError("a table seats %d players, got %d"
                         % (PLAYERS, len(players)))

    contract, log = run_auction(deal, players, stick_the_dealer, allow_loners)
    if contract is None:
        return Result(deal=deal, contract=None, value=0, caller_score=0,
                      caller_tricks=0, auction=log)

    caller_tricks, plays, winners = play_contract(contract, players)
    caller_score = int(_final(caller_tricks, contract.alone))
    return Result(deal=deal, contract=contract,
                  value=b.net_to_team0(caller_score, contract.caller),
                  caller_score=caller_score, caller_tricks=caller_tricks,
                  plays=plays, winners=winners, auction=log)
