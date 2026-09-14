"""
Decision rules: the things that sit in the seats.

`table.py` is a referee with no opinions. This is where opinions live. A player
is any object with three methods -- `bid`, `discard` and `play` -- each handed a
turn object and each returning one of the options on it. That is the whole
protocol; there is no rule language and no registry, because the roadmap is
explicit that the vocabulary should grow out of rules actually written rather
than be guessed at up front.

Three are here:

  * `PerfectPlayer` sees all four hands and plays the double-dummy optimum. A
    table of four of them reproduces `bidding.solve_bidding` followed by
    `fast_search.definitive_winner` exactly, which is what makes it useful --
    it is the existing baseline expressed as players, so the new machinery can
    be checked against the old answer.
  * `PIMCPlayer` sees only what its seat has seen. At every decision it samples
    layouts consistent with that, solves each one exactly, and takes the option
    with the best average. This is the thing the roadmap called the unclaimed
    middle rung.
  * `RandomPlayer` picks uniformly among legal options. A floor to measure
    against, and a cheap way to shake out legality bugs in the referee.

## What PIMC actually is, and what it is not

For each option, and for each of N imagined deals, ask the exact solver what
that option is worth; average; take the best. It is not a search over
information sets and it does not know that it does not know. Two consequences
are worth having in mind before reading any number this produces:

  * **Strategy fusion.** It evaluates each world as though it could play
    differently in each, so it credits itself with plans it cannot actually
    carry out -- "I finesse if the king is on my left, otherwise I don't" gets
    scored as though it will always guess right.
  * **Non-locality.** It assumes the opponents will play the double-dummy
    optimum for a hand they cannot see either, so it expects them to find
    defences no real player could.

Both make it optimistic. Neither makes it weak: PIMC is famously strong at
trick-taking games in spite of them, and it fails in recognisably human ways --
it cannot signal to its partner and cannot read a signal. What it gives this
project is an opponent that is genuinely not omniscient, which is the thing
perfect-knowledge bidding was distorting.

## The cost, and the knobs

A decision costs (options x samples) double-dummy solves. Card play is cheap,
because a position solve part-way through a hand is far smaller than a whole
one. Bidding is where the time goes, and nearly all of it goes to pricing
*passing*: a pass is worth whatever the rest of the auction does, so valuing it
means running the rest of the auction -- up to 36 solves per sampled world.

`pass_model` chooses how that is done:

    "dd"    price a pass by solving the rest of the auction in each sampled
            world under perfect knowledge. Accurate about the shape of the
            auction, and inconsistent in an obvious way -- inside the sample,
            the other seats can see the hand this player is trying to hide. It
            is also systematically pessimistic about passing, because perfect
            knowledge essentially always finds a call: the pass branch reads
            "an opponent ends up calling this" far more often than a real table
            would, and the player calls too much to head that off.
    "zero"  a pass is worth nothing. About 4x faster, and a markedly more
            selective bidder -- a call has to beat literally nothing instead of
            beating a pessimistically priced pass, so the marginal ones get
            declined. Measured euchre rate is roughly half "dd"'s.

**Neither is clearly stronger, and the obvious measurement says otherwise.**
Head to head against perfect knowledge with the teams swapped on every deal,
"dd" scores -1.26 +/- 0.36 points a deal and "zero" -1.34 +/- 0.36 over the same
50 -- indistinguishable. "zero" looks far better on mean points *per call*
(+0.65 against -0.17), but that average is taken only over the deals a player
chose to call and drops whatever passing cost it, which is exactly the trap
`bidding.py` names when it says passing is not free.

Default is "dd". It is the one that actually tries to answer "what happens if I
decline"; "zero" answers a different question and is the right tool when the
sweep needs to be four times bigger.
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

    Every option a player weighs has to land on one scale before they can be
    compared, and `seat`'s own team is the natural one: bigger is better, with
    no sign convention to remember.
    """
    return caller_value if (seat % 2) == (caller % 2) else -caller_value


# ------------------------------------------------------- engine bridging


def _engine_position(hands, trump, caller, alone, current, to_act):
    """
    Lay a position out the way `fast_search.position_moves` wants it.

    `hands` is the cards each seat still holds, as natural cards; `current` is
    the trick in progress as (seat, card). The returned card array is indexed
    the same way `hands[to_act]` is, so an index that comes back out of the
    search names a card the caller already has in hand.
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
    Double-dummy value of each legal card in one fully specified layout.

    Returns {card: value from the calling team's side}. One solve per candidate
    card, which is the unit of work every PIMC decision is built out of.
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

    Among options the search rates *identically* -- which is common, since a
    Euchre hand is worth one of four numbers and most cards do not change which
    one -- LOW throws the cheapest card. That is a strategy heuristic, but it
    only ever chooses between moves of equal expected value, so it cannot cost
    anything the model can see. FIRST keeps hand order instead, which is what
    to use when measuring PIMC rather than trying to win with it.
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

    Useful for exactly two things: a floor to measure real players against, and
    finding referee bugs, since it will cheerfully try every legal line
    including the ones nobody sensible would reach.
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
    Sees every hand and plays the double-dummy optimum. The existing baseline.

    Four of these at a table reproduce `bidding.solve_bidding` bid for bid and
    `fast_search.solve_line` card for card, which is the point: it pins the new
    referee to the old answer. It is also, of course, a cheat -- it reads
    `turn.deal`, which no honest player may touch.
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
    Perfect-Information Monte Carlo: solve the hands you might be in.

    At every decision it draws `samples` layouts consistent with what its seat
    has seen, scores every option exactly in each, and takes the best average.
    It never reads `turn.deal`.

    Args:
        samples: layouts drawn per card-play decision.
        bid_samples: layouts drawn per bidding decision. Defaults to `samples`.
            Bidding decisions cost far more per sample than card play does --
            see `pass_model` -- so it is usually the one to turn down first.
        pass_model: PASS_DD or PASS_ZERO; see the module docstring.
        tie_break: LOW or FIRST, for cards the search rates identically.
        rng: seed it for a reproducible player.

    `solves` and `nodes` accumulate what it has spent, which is the honest way
    to report the cost of a sweep. `last_scores` holds the averaged value of
    every option from the most recent decision -- the numbers the choice was
    actually made on, kept so that a front end or a narrated example can show
    the working rather than just the answer. It is empty when there was nothing
    to decide.
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

        The dealer is the only seat that ever answers this, and it answers for
        its *own* team -- which, when the opposition ordered it up, means
        choosing the card that hurts the contract most.
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
    Four players from one factory, each with its own seeded rng if it takes one.

    `factory` is called with the seat number. Seeding per seat rather than
    globally keeps a sweep reproducible even though the seats consume random
    numbers at rates that depend on what they decide to do.
    """
    return [factory(seat) for seat in range(n)]
