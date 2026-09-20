"""
The two questions a front end asks, and nothing else.

Everything else in this repo is a tool for answering one of them, and each is
answered by a different engine for a different reason:

    evaluate(...)   what is this hand worth if I order it / go alone / pass?
    solve(...)      given all four hands, what happens?

`evaluate` is the calculator. It knows only what one seat can see, so it
samples the layouts that seat could be in and plays each one out with four
players who cannot see each other's cards -- `fastsim` over `bitcore`, through
`hand_ev.run_fast`. A thousand deals is about seven seconds.

`solve` is the microscope. It is handed every card and answers exactly, with no
sampling at all: the auction God Mode would run, and the line of play that
follows it. About sixty milliseconds, on the readable Python path, because at
sixty milliseconds there is nothing to gain by compiling it.

## What `evaluate` reports, and why that one

The mean is taken over **the deals the auction actually reached the seat on**.
A player only ever faces the decision on the deals they get to bid on, so the
rest are not part of what the bid is worth. That is not a detail: pricing the
call on every deal instead gives the *opposite* advice about going alone, since
the deals you never get to bid on are the ones an opponent opened -- and so the
ones where an opponent has the cards. `notes/front_end.md` has the measurement.

Each action gets its own absolute number in points, not a comparison against
the others. "Order this and you score +0.9" is something a player can act on
and check against their own experience; "order beats pass by 0.15" is an
abstraction they would have to take on trust.

## The two sim counts

`deals` is the outer loop and the only thing the error bar is on: roughly
`4 / sqrt(deals)`, and a query costs about 5.7 ms a deal for all three actions.
That is the dial to put in front of a user.

`play_sims` / `bid_sims` / `discard_sims` are how many layouts each player
imagines per decision. They move the mean rather than narrowing it, and
measured at 10,000 deals -- tight enough to see a shift of 0.05 -- raising them
sixteenfold moved the answer by less than the error bar while costing 4.5x the
time. They are exposed because they should be, and defaulted because spending a
user's seconds there is the wrong trade. That was one hand at one `epsilon`; it
is not a flatness claim about PIMC in general, which `CLAUDE.md` is right to
keep warning about.

## Shapes

Every result is a frozen dataclass with an `as_dict()` that returns plain JSON
types -- cards as `"JS"`, suits as `"spades"` -- so a web layer can serialise
one without knowing anything about `rotation.Card`.
"""
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import bidding as b
import game
import hand_ev
import players
import rotation as r
from fast_search import decode_card, solve_line

ORDER = "order"
ORDER_ALONE = "order alone"
PASS = "pass"
ACTIONS = (ORDER, ORDER_ALONE, PASS)

_ASSUME = {ORDER: hand_ev.ASSUME_ORDER,
           ORDER_ALONE: hand_ev.ASSUME_ALONE,
           PASS: hand_ev.ASSUME_PASS}

DEALS = 1000

# Roughly what one deal costs, across all three actions, on ten threads. Only
# ever used to quote a wait time before a query runs -- the cost is linear in
# `deals`, which is what makes quoting one possible at all. Measured on a
# 12-thread machine; a slower one will overrun the estimate and the page says
# "about".
SECONDS_PER_DEAL = 0.0057


# ------------------------------------------------------------------- input


def _cards(hand) -> Tuple[r.Card, ...]:
    """A hand as five cards, from `"TH AS AD KD JD"` or from Cards."""
    if isinstance(hand, str):
        return tuple(r.parse_hand(hand))
    return tuple(hand)


def _card(card) -> r.Card:
    return r.parse_card(card) if isinstance(card, str) else card


def _check_seat(name: str, value: int):
    if not 0 <= value < game.PLAYERS:
        raise ValueError("%s must be 0-%d, got %r"
                         % (name, game.PLAYERS - 1, value))


# ------------------------------------------------------------------ output


@dataclass(frozen=True)
class ActionEV:
    """What one action is worth, over the deals that reached the seat."""

    action: str
    ev: float
    interval: float                  # 95% half-width on `ev`
    deals: int                       # layouts sampled
    reached: int                     # of those, how many reached the seat
    roles: Dict[str, int] = field(default_factory=dict)
    tricks: Dict[str, int] = field(default_factory=dict)

    @property
    def reach_rate(self) -> float:
        return self.reached / self.deals if self.deals else 0.0

    def as_dict(self) -> dict:
        return {"action": self.action, "ev": round(self.ev, 4),
                "interval": round(self.interval, 4), "deals": self.deals,
                "reached": self.reached,
                "reach_rate": round(self.reach_rate, 4),
                "roles": dict(self.roles), "tricks": dict(self.tricks)}


@dataclass(frozen=True)
class Evaluation:
    """`evaluate`'s answer: one `ActionEV` per action, plus what was asked."""

    hand: Tuple[r.Card, ...]
    up_card: r.Card
    seat: int
    dealer: int
    deals: int
    actions: Tuple[ActionEV, ...]

    def __getitem__(self, action: str) -> ActionEV:
        for got in self.actions:
            if got.action == action:
                return got
        raise KeyError(action)

    def best(self) -> ActionEV:
        """
        The action with the highest mean.

        A convenience for a caller that wants one, and **not** advice: two
        actions inside each other's intervals are not separated by this, and
        going alone carries roughly double the interval of a four-handed call
        at the same deal count, because its outcomes run -2..+4 rather than
        -2..+2. Show the intervals.
        """
        return max(self.actions, key=lambda a: a.ev)

    def as_dict(self) -> dict:
        return {"hand": [r.card_name(c) for c in self.hand],
                "up_card": r.card_name(self.up_card),
                "seat": self.seat, "dealer": self.dealer, "deals": self.deals,
                "actions": [a.as_dict() for a in self.actions]}


@dataclass(frozen=True)
class Trick:
    """One trick of an exactly solved hand."""

    number: int                      # 1-5
    plays: Tuple[Tuple[int, r.Card], ...]
    winner: int

    def as_dict(self) -> dict:
        return {"number": self.number, "winner": self.winner,
                "plays": [{"seat": s, "card": r.card_name(c)}
                          for s, c in self.plays]}


@dataclass(frozen=True)
class Solution:
    """`solve`'s answer: the exact auction, and the line of play under it."""

    hands: Tuple[Tuple[r.Card, ...], ...]
    up_card: r.Card
    dealer: int
    auction: Tuple[str, ...]
    caller: Optional[int]
    trump: Optional[int]
    alone: bool
    discard: Optional[r.Card]
    value: int                       # net points to team 0
    caller_score: int
    caller_tricks: int
    tricks: Tuple[Trick, ...]
    options: Optional[Dict[str, int]] = None
    options_seat: Optional[int] = None

    @property
    def passed_out(self) -> bool:
        return self.caller is None

    def as_dict(self) -> dict:
        return {
            "hands": [[r.card_name(c) for c in h] for h in self.hands],
            "up_card": r.card_name(self.up_card),
            "dealer": self.dealer,
            "auction": list(self.auction),
            "passed_out": self.passed_out,
            "caller": self.caller,
            "trump": r.suit_name(self.trump) if self.trump is not None else None,
            "alone": self.alone,
            "discard": r.card_name(self.discard) if self.discard else None,
            "value_to_team0": self.value,
            "caller_score": self.caller_score,
            "caller_tricks": self.caller_tricks,
            "tricks": [t.as_dict() for t in self.tricks],
            "options": dict(self.options) if self.options else None,
            "options_seat": self.options_seat,
        }


# --------------------------------------------------------------- evaluate


def _summarise(action: str, records, seat: int) -> ActionEV:
    """One action's records, reduced to what a front end shows."""
    reached = [rec for rec in records if rec.forced]
    values = [rec.value for rec in reached]
    if values:
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            interval = 1.96 * math.sqrt(var / len(values))
        else:
            interval = 0.0
    else:
        mean = interval = 0.0

    roles: Dict[str, int] = {}
    for rec in reached:
        role = hand_ev.role_of(rec.caller, seat)
        roles[role] = roles.get(role, 0) + 1

    # Tricks only mean something on the deals this seat's team had the
    # contract; on the others the caller's trick count is the opponents'.
    ours = [rec for rec in reached
            if not rec.passed_out and rec.caller % 2 == seat % 2]
    tricks = {"march": sum(1 for x in ours if x.caller_tricks == 5),
              "made": sum(1 for x in ours if 3 <= x.caller_tricks < 5),
              "euchred": sum(1 for x in ours if x.caller_tricks < 3),
              "contracts": len(ours)}

    return ActionEV(action=action, ev=mean, interval=interval,
                    deals=len(records), reached=len(reached), roles=roles,
                    tricks=tricks)


def evaluate(hand, up_card, seat: int = 0, dealer: int = 3,
             deals: int = DEALS, actions: Sequence[str] = ACTIONS,
             play_sims: Optional[int] = None, bid_sims: Optional[int] = None,
             discard_sims: Optional[int] = None,
             pass_model: str = players.PASS_ZERO,
             allow_loners: bool = True, stick: bool = False,
             epsilon: Optional[float] = hand_ev.EPSILON,
             workers: int = 1, seed: int = 0,
             tt_bits: Optional[int] = None, table=None) -> Evaluation:
    """
    What this hand is worth, per action, to a table that cannot see it.

    Args:
        hand: five cards, as `"TH AS AD KD JD"` or as `rotation.Card`s.
        up_card: the turned card, as `"9H"` or a `Card`.
        seat: where you are sitting, 0-3.
        dealer: the dealing seat, 0-3. Together with `seat` this fixes how
            many seats speak before you, which is most of what the up-card is
            worth.
        deals: layouts to play out per action. The error bar is roughly
            `4 / sqrt(deals)`; the whole query costs about `deals * 0.0057`
            seconds for all three actions.
        actions: which of "order", "order alone", "pass" to price.
        play_sims, bid_sims, discard_sims: layouts each player imagines per
            decision. See the module docstring -- these move the mean rather
            than narrowing it, and the measured defaults are hard to beat.
        workers: threads. One query already uses every core it is given, so a
            server should serialise queries rather than run them in parallel.
        table: a transposition table from `hand_ev.new_table`, to reuse across
            queries instead of allocating 134 MB per call. `Engine` does this
            for you.

    Returns an `Evaluation`. Each action's mean is taken over the deals the
    auction reached this seat on, which is the population a player faces.
    """
    hand = _cards(hand)
    up_card = _card(up_card)
    _check_seat("seat", seat)
    _check_seat("dealer", dealer)
    if len(hand) != game.HAND_SIZE:
        raise ValueError("a hand is %d cards, got %d"
                         % (game.HAND_SIZE, len(hand)))
    if len(set(hand)) != len(hand):
        raise ValueError("the same card appears twice in the hand")
    if up_card in hand:
        raise ValueError("%s is both in the hand and the up-card"
                         % r.card_name(up_card))
    if deals < 1:
        raise ValueError("deals must be at least 1, got %r" % (deals,))
    for name in actions:
        if name not in _ASSUME:
            raise ValueError("no such action: %r" % (name,))

    out = []
    for name in actions:
        setup = hand_ev.Setup(
            hand=hand, up_card=up_card, seat=seat, dealer=dealer,
            player_eval_sims=hand_ev._budget(play_sims, None,
                                             hand_ev.PLAYER_EVAL_SIMS),
            bid_eval_sims=hand_ev._budget(bid_sims, play_sims,
                                          hand_ev.BID_EVAL_SIMS),
            discard_eval_sims=hand_ev._budget(discard_sims, play_sims,
                                              hand_ev.DISCARD_EVAL_SIMS),
            pass_model=pass_model, allow_loners=allow_loners, stick=stick,
            seed=seed, epsilon=epsilon, assume=_ASSUME[name],
            let_auction_play=True)
        records, _, _ = hand_ev.run_fast(setup, deals, workers=workers,
                                         tt_bits=tt_bits, table=table)
        out.append(_summarise(name, records, seat))

    return Evaluation(hand=hand, up_card=up_card, seat=seat, dealer=dealer,
                      deals=deals, actions=tuple(out))


# ------------------------------------------------------------------ solve


def _line(contract) -> Tuple[Trick, ...]:
    """The optimal line under a settled contract, as natural cards."""
    engine = r.deal_to_engine(contract.deal.hands, contract.trump)
    _, suits, strengths, seats, winners = solve_line(
        engine, contract.deal.first_bidder, contract.caller,
        alone=contract.alone)
    out = []
    for t in range(suits.shape[0]):
        plays = tuple(
            (int(seats[t, k]),
             r.card_from_engine(decode_card(suits[t, k], strengths[t, k]),
                                contract.trump))
            for k in range(suits.shape[1]))
        out.append(Trick(number=t + 1, plays=plays, winner=int(winners[t])))
    return tuple(out)


def solve(hands, up_card, dealer: int = 3, seat: Optional[int] = None,
          allow_loners: bool = True, stick: bool = False) -> Solution:
    """
    Given every card, what happens -- exactly, with no sampling.

    The auction is solved in God Mode: every seat sees every hand and bids to
    maximise its own team's net points, knowing how the rest of the auction and
    the play will go. The line of play under the surviving contract is the
    minimax one. Both are exact; neither is an estimate.

    Args:
        hands: four hands, each five cards, as strings or `Card`s, indexed by
            seat.
        up_card: the turned card.
        dealer: the dealing seat, which fixes the bidding order and who picks
            up.
        seat: if given, also return every round-one option open to that seat
            on its own team's scale -- `bidding.bid_options`.

    Returns a `Solution`. `tricks` is empty on a deal nobody would take, which
    God Mode essentially never does: 0 of 1600 measured auctions.
    """
    hands = tuple(_cards(h) for h in hands)
    up_card = _card(up_card)
    _check_seat("dealer", dealer)
    if seat is not None:
        _check_seat("seat", seat)
    if len(hands) != game.PLAYERS:
        raise ValueError("a table seats %d, got %d hands"
                         % (game.PLAYERS, len(hands)))
    for i, held in enumerate(hands):
        if len(held) != game.HAND_SIZE:
            raise ValueError("seat %d holds %d cards, not %d"
                             % (i, len(held), game.HAND_SIZE))

    dealt = [c for held in hands for c in held] + [up_card]
    if len(set(dealt)) != len(dealt):
        raise ValueError("the same card was dealt twice")
    buried = tuple(c for c in r.full_deck() if c not in set(dealt))

    deal = game.Deal(hands=hands, up_card=up_card, buried=buried,
                     dealer=dealer).check()
    outcome = b.solve_bidding(deal, stick_the_dealer=stick,
                              allow_loners=allow_loners)
    options = (b.bid_options(deal, seat, allow_loners)
               if seat is not None else None)

    if outcome.passed_out:
        return Solution(hands=hands, up_card=up_card, dealer=dealer,
                        auction=tuple(outcome.line), caller=None, trump=None,
                        alone=False, discard=None, value=0, caller_score=0,
                        caller_tricks=0, tricks=(), options=options,
                        options_seat=seat)

    contract = outcome.contract
    tricks = _line(contract)
    won = sum(1 for t in tricks if t.winner % 2 == contract.caller % 2)
    caller_score = b.value_to(contract.caller, outcome.value)
    return Solution(hands=hands, up_card=up_card, dealer=dealer,
                    auction=tuple(outcome.line), caller=contract.caller,
                    trump=contract.trump, alone=contract.alone,
                    discard=contract.discard, value=outcome.value,
                    caller_score=caller_score, caller_tricks=won,
                    tricks=tricks, options=options, options_seat=seat)


# ----------------------------------------------------------------- engine


class Engine:
    """
    A warm process: one transposition table, one thread count, one query.

    Three things a long-lived process needs and a bare function call does not:

    **A table that survives.** `run_fast` allocates 134 MB per call otherwise,
    and `evaluate` makes three. It is never cleared and its entries are keyed
    by everything their value depends on, so a query inherits whatever the
    last one learned -- the table is not just reused, it gets warmer.

    **A lock.** One query already uses every core it is given. Two at once
    oversubscribe numba's thread pool and both get slower, so queries are
    serialised rather than run in parallel. An HTTP server in front of this
    wants a queue, not a thread pool.

    **A warm-up.** The first call into the compiled engine pays about 1.2 s
    loading the cached machine code, and ~80 s *compiling* it if the cache is
    cold -- which it is after any edit to `bitcore.py` or `fastsim.py`. `warm()`
    gets that over with at boot so no user wears it. `ready` says whether it
    has happened.
    """

    def __init__(self, workers: int = 1, tt_bits: Optional[int] = None,
                 deals: int = DEALS):
        self.workers = max(1, int(workers))
        self._table = hand_ev.new_table(tt_bits, deals)
        self._lock = threading.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def warm(self) -> None:
        """Force the compiled code to load, on a query nobody is waiting for."""
        with self._lock:
            if not self._ready:
                evaluate("JS AS 9H 9D TC", "9S", seat=0, dealer=3, deals=2,
                         play_sims=1, bid_sims=1, discard_sims=1,
                         workers=self.workers, table=self._table)
                # A real deal rather than four hands written out, so the
                # warm-up cannot go stale by dealing a card twice.
                dealt = game.deal_random(rng=random.Random(0), dealer=3)
                solve(dealt.hands, dealt.up_card, dealer=3)
                self._ready = True

    def evaluate(self, hand, up_card, seat: int = 0, dealer: int = 3,
                 deals: int = DEALS, **kw) -> Evaluation:
        kw.setdefault("workers", self.workers)
        with self._lock:
            self._ready = True
            return evaluate(hand, up_card, seat=seat, dealer=dealer,
                            deals=deals, table=self._table, **kw)

    def solve(self, hands, up_card, dealer: int = 3, **kw) -> Solution:
        with self._lock:
            self._ready = True
            return solve(hands, up_card, dealer=dealer, **kw)
