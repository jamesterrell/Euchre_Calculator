"""
Decision rules: the things that sit in the seats.

`table.py` is a referee with no opinions. This is where opinions live. A player
is any object with `bid`, `discard` and `play` methods, each handed a turn
object and each returning one of the options on it. That is the whole protocol
-- no rule language, no registry.

  * `PerfectPlayer` plays in God Mode: it sees all four hands and takes the
    true optimum. Four of them reproduce `bidding.solve_bidding` followed by
    `fast_search.definitive_winner` exactly, which pins the new machinery to
    the old answer.
  * `PIMCPlayer` sees only what its seat has seen, and runs a Perfect
    Information Monte Carlo sim over that: sample layouts, solve each exactly,
    take the best average.
  * `RandomPlayer` picks uniformly among legal options -- a floor to measure
    against, and a cheap way to shake out referee bugs.

## What the PIMC sim is, and is not

For each option, and each of N imagined deals, ask the exact solver what that
option is worth; average; take the best. It is not a search over information
sets and it does not know that it does not know. Two consequences:

  * **Strategy fusion.** It scores each world as though it could play
    differently in each, crediting itself with plans it cannot carry out --
    "I finesse if the king is on my left" is scored as though it always guesses
    right.
  * **Non-locality.** It assumes opponents play the God Mode optimum for a hand
    they cannot see either, so it expects defences no real player could find.

Both make it optimistic; neither makes it weak. It fails in recognisably human
ways -- it cannot signal to its partner or read a signal -- which is exactly
what God Mode bidding was distorting.

## The cost, and the knobs

A decision costs (options x samples) solves. Card play is cheap, since a
mid-hand position solve is far smaller than a whole one. Bidding is where the
time goes, and nearly all of it goes to pricing *passing*: a pass is worth
whatever the rest of the auction does, so valuing it means running the rest of
the auction -- up to 36 solves per sampled world.

`pass_model` chooses how:

    "dd"    price a pass by running the rest of the auction in God Mode inside
            each sampled world. Accurate about the shape of the auction, and
            inconsistent in an obvious way -- inside the sample, the other
            seats can see the hand this player is hiding. Systematically
            pessimistic about passing, because God Mode essentially always
            finds a call, so the pass branch reads "an opponent calls this" far
            more often than a real table would.
    "zero"  a pass is worth nothing. About 4x faster and a markedly more
            selective bidder; euchre rate is roughly half "dd"'s.

**Neither is clearly stronger.** Head to head against God Mode with the teams
swapped on every deal, "dd" scores -1.26 +/- 0.36 points a deal and "zero"
-1.34 +/- 0.36 over the same 50 -- indistinguishable. "zero" looks far better on
mean points *per call* (+0.65 vs -0.17), but that average covers only the deals
a player chose to call and drops whatever passing cost it: the trap `bidding.py`
names when it says passing is not free.

Default is "dd" -- the one that actually answers "what happens if I decline".
"zero" answers a different question, and is the right tool when the sweep needs
to be four times bigger.
"""

import random
from typing import List, Optional, Tuple

import numpy as np

import bidding as b
import observation as obs
import rotation as r
import table as t
from fast_search import position_moves
from game import Deal, PLAYERS

PASS_DD = "dd"
PASS_ZERO = "zero"

# Tie-breaks among options the search rates identically.
LOW = "low"
FIRST = "first"


def to_seat(caller_value: int, caller: int, seat: int) -> int:
    """
    A caller's-perspective score, re-expressed for `seat`'s own team.

    Every option has to land on one scale before it can be compared, and the
    seat's own team is the natural one: bigger is better, no sign convention.
    """
    return caller_value if (seat % 2) == (caller % 2) else -caller_value


# ------------------------------------------------------- engine bridging


def _engine_position(hands, trump, caller, alone, current, to_act):
    """
    Lay a position out the way `fast_search.position_moves` wants it.

    `hands` is each seat's remaining natural cards; `current` is the trick in
    progress as (seat, card). The returned array is indexed the same way
    `hands[to_act]` is, so an index out of the search names a card in hand.
    """
    sitting = (caller + 2) % PLAYERS if alone else None
    counts = np.array([0 if s == sitting else len(hands[s])
                       for s in range(PLAYERS)], dtype=np.int64)
    width = max(1, int(counts.max()))

    arr = np.zeros((PLAYERS, width, 2), dtype=np.int64)
    for seat in range(PLAYERS):
        if seat == sitting:
            continue
        for i, card in enumerate(hands[seat]):
            arr[seat, i] = r.card_to_engine(card, trump)

    trick_cards = np.array(
        [r.card_to_engine(card, trump) for _, card in current],
        dtype=np.int64).reshape(-1, 2)
    trick_players = np.array([seat for seat, _ in current], dtype=np.int64)
    return arr, counts, trick_cards, trick_players


def _remaining(deal: Deal, plays) -> List[Tuple[r.Card, ...]]:
    """Each seat's cards as they stand, given everything played so far."""
    spent = {seat: set() for seat in range(PLAYERS)}
    for seat, card in plays:
        spent[seat].add(card)
    return [tuple(c for c in deal.hands[seat] if c not in spent[seat])
            for seat in range(PLAYERS)]


def _card_values(hands, turn: "t.PlayTurn"):
    """
    God Mode value of each legal card in one fully specified layout.

    Returns {card: value to the calling team}. One solve per candidate card --
    the unit of work every PIMC decision is built out of.
    """
    order = tuple(hands[turn.seat])
    arr, counts, trick_cards, trick_players = _engine_position(
        hands, turn.trump, turn.caller, turn.alone,
        turn.observation.current_trick, turn.seat)

    idx, vals, nodes = position_moves(
        arr, counts, trick_cards, trick_players, turn.seat, turn.caller,
        turn.caller_tricks, turn.trick_no, turn.alone)
    return {order[int(i)]: int(v) for i, v in zip(idx, vals)}, nodes


def _pick(scored, order, tie_break=LOW, trump=None):
    """
    The best-scoring option, with ties broken deliberately rather than by luck.

    Ties are common: a Euchre hand is worth one of four numbers and most cards
    do not change which. LOW then throws the cheapest card -- a heuristic, but
    one that only ever picks between moves of equal expected value, so it
    cannot cost anything the model can see. FIRST keeps hand order instead,
    which is what to use when measuring PIMC rather than winning with it.
    """
    best = max(scored[c] for c in order)
    tied = [c for c in order if scored[c] == best]
    if tie_break == LOW and trump is not None and len(tied) > 1:
        return min(tied, key=lambda c: t.card_order(c, trump))
    return tied[0]


# --------------------------------------------------------------- players


class RandomPlayer:
    """
    Picks uniformly among whatever is legal. No model of anything.

    A floor to measure against, and a referee-bug finder: it will try every
    legal line, including the ones nobody sensible would reach.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def bid(self, turn: "t.BidTurn") -> "t.Bid":
        return self.rng.choice(turn.options)

    def discard(self, turn: "t.DiscardTurn") -> r.Card:
        return self.rng.choice(turn.options)

    def play(self, turn: "t.PlayTurn") -> r.Card:
        return self.rng.choice(turn.legal)


class PerfectPlayer:
    """
    God Mode: sees every hand and plays the true optimum. The baseline.

    Four of these reproduce `bidding.solve_bidding` bid for bid and
    `fast_search.solve_line` card for card, which pins the referee to the old
    answer. It is a cheat by construction -- it reads `turn.deal`, which no
    honest player may touch.
    """

    def __init__(self, tie_break: str = FIRST):
        self.tie_break = tie_break

    def bid(self, turn: "t.BidTurn") -> "t.Bid":
        scored = {}
        for option in turn.options:
            if option.action == t.PASS:
                rest = b.rest_of_auction(
                    turn.deal, turn.index + 1, turn.order,
                    turn.stick_the_dealer, turn.allow_loners,
                    turn.bidding_round)
                value = rest.value
            elif option.action == t.ORDER:
                value = b.order_up(turn.deal, turn.seat, option.alone)[0]
            else:
                value = b.name_suit(turn.deal, turn.seat, option.suit,
                                    option.alone)[0]
            scored[option] = b.value_to(turn.seat, value)
        return _pick(scored, turn.options, tie_break=FIRST)

    def discard(self, turn: "t.DiscardTurn") -> r.Card:
        return b.best_discard(turn.before, turn.caller, turn.alone)

    def play(self, turn: "t.PlayTurn") -> r.Card:
        hands = _remaining(turn.deal, turn.plays)
        values, _ = _card_values(hands, turn)
        scored = {c: to_seat(v, turn.caller, turn.seat)
                  for c, v in values.items()}
        return _pick(scored, turn.legal, tie_break=self.tie_break,
                     trump=turn.trump)


class PIMCPlayer:
    """
    Perfect Information Monte Carlo sim: solve the hands you might be in.

    At every decision it draws `samples` layouts consistent with what its seat
    has seen, scores every option exactly in each, and takes the best average.
    It never reads `turn.deal`.

    Args:
        samples: layouts drawn per card-play decision.
        bid_samples: layouts per bidding decision; defaults to `samples`. These
            cost far more each -- see `pass_model` -- so turn this down first.
        pass_model: PASS_DD or PASS_ZERO; see the module docstring.
        tie_break: LOW or FIRST, for cards the search rates identically.
        rng: seed it for a reproducible player.

    `solves` and `nodes` accumulate what it has spent. `last_scores` holds the
    averaged value of every option from the most recent decision, so a front
    end or a narrated example can show the working rather than just the answer;
    it is empty when there was nothing to decide.
    """

    def __init__(self, samples: int = 20, bid_samples: Optional[int] = None,
                 pass_model: str = PASS_DD, tie_break: str = LOW,
                 rng: Optional[random.Random] = None):
        if pass_model not in (PASS_DD, PASS_ZERO):
            raise ValueError("no such pass model: %r" % (pass_model,))
        self.samples = samples
        self.bid_samples = samples if bid_samples is None else bid_samples
        self.pass_model = pass_model
        self.tie_break = tie_break
        self.rng = rng or random.Random()
        self.solves = 0
        self.nodes = 0
        self.last_scores = {}

    # ------------------------------------------------------------ helpers

    def _worlds(self, observation: obs.Observation, n: int):
        return obs.sample_worlds(observation, n, self.rng)

    def _deal_of(self, world: obs.World, observation: obs.Observation) -> Deal:
        """A sampled world, dressed as a `game.Deal` the auction can run on."""
        buried = tuple(c for c in world.kitty if c != observation.up_card)
        return Deal(hands=world.hands, up_card=observation.up_card,
                    buried=buried, dealer=observation.dealer,
                    picked_up=False).check()

    # --------------------------------------------------------------- bid

    def bid(self, turn: "t.BidTurn") -> "t.Bid":
        observation = turn.observation
        totals = {option: 0 for option in turn.options}

        for world in self._worlds(observation, self.bid_samples):
            deal = self._deal_of(world, observation)
            for option in turn.options:
                totals[option] += self._bid_value(option, deal, turn)

        self.last_scores = {o: v / max(1, self.bid_samples)
                            for o, v in totals.items()}
        return _pick(totals, turn.options)

    def _bid_value(self, option: "t.Bid", deal: Deal,
                   turn: "t.BidTurn") -> int:
        """What one option is worth in one imagined deal, on this seat's scale."""
        if option.action == t.PASS:
            if self.pass_model == PASS_ZERO:
                return 0
            self.solves += 1
            rest = b.rest_of_auction(
                deal, turn.index + 1, turn.order, turn.stick_the_dealer,
                turn.allow_loners, turn.bidding_round)
            value = rest.value
        elif option.action == t.ORDER:
            self.solves += 1
            value = b.order_up(deal, turn.seat, option.alone)[0]
        else:
            self.solves += 1
            value = b.name_suit(deal, turn.seat, option.suit, option.alone)[0]
        return b.value_to(turn.seat, value)

    # ----------------------------------------------------------- discard

    def discard(self, turn: "t.DiscardTurn") -> r.Card:
        """
        Which of the six to bury, judged the same way as everything else.

        Only the dealer ever answers this, and it answers for its *own* team --
        so if the opposition ordered it up, it picks the card that hurts the
        contract most.
        """
        observation = turn.observation
        up_card = observation.up_card       # public; turn.deal is not touched
        totals = {card: 0 for card in turn.options}

        for world in self._worlds(observation, self.bid_samples):
            # The world holds the dealer's six; the deal it came from held five.
            dealt = tuple(c for c in world.hands[turn.seat] if c != up_card)
            hands = tuple(dealt if s == turn.seat else world.hands[s]
                          for s in range(PLAYERS))
            before = Deal(hands=hands, up_card=up_card,
                          buried=world.kitty, dealer=turn.seat,
                          picked_up=False).check()

            for card in turn.options:
                self.solves += 1
                value = b.play_value(before.pick_up(discard=card), turn.trump,
                                     turn.caller, turn.alone)
                totals[card] += to_seat(value, turn.caller, turn.seat)

        self.last_scores = {c: v / max(1, self.bid_samples)
                            for c, v in totals.items()}
        return _pick(totals, turn.options, tie_break=self.tie_break,
                     trump=turn.trump)

    # -------------------------------------------------------------- play

    def play(self, turn: "t.PlayTurn") -> r.Card:
        if len(turn.legal) == 1:
            # Nothing to think about, and thinking costs a few hundred solves.
            self.last_scores = {}
            return turn.legal[0]

        observation = turn.observation
        totals = {card: 0 for card in turn.legal}

        for world in self._worlds(observation, self.samples):
            hands = list(world.hands)
            values, nodes = _card_values(hands, turn)
            self.solves += len(values)
            self.nodes += nodes
            for card, value in values.items():
                totals[card] += to_seat(value, turn.caller, turn.seat)

        self.last_scores = {c: v / max(1, self.samples)
                            for c, v in totals.items()}
        return _pick(totals, turn.legal, tie_break=self.tie_break,
                     trump=turn.trump)


def table_of(factory, n: int = PLAYERS, seed: Optional[int] = None):
    """
    Four players from one factory, called with the seat number.

    Seeding per seat rather than globally keeps a sweep reproducible even
    though seats consume random numbers at rates that depend on their choices.
    """
    return [factory(seat) for seat in range(n)]
