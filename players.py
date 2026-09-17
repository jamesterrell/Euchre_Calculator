"""
Decision rules: the things that sit in the seats.

`table.py` is a referee with no opinions. This is where opinions live. A player
is any object with `bid`, `discard` and `play` methods, each handed a turn
object and each returning one of the options on it. That is the whole protocol
-- no rule language, no registry.

  * `GodModePlayer` sees all four hands and takes the true optimum. Four of
    them reproduce `bidding.solve_bidding` followed by
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

    "god"   price a pass by running the rest of the auction in God Mode inside
            each sampled world. Accurate about the shape of the auction, and
            inconsistent in an obvious way -- inside the sample, the other
            seats can see the hand this player is hiding. Systematically
            pessimistic about passing, because God Mode essentially always
            finds a call, so the pass branch reads "an opponent calls this" far
            more often than a real table would.
    "zero"  a pass is worth nothing. About 4x faster and a markedly more
            selective bidder; euchre rate is roughly half "god"'s.
    "guard" "god", plus one override: if **every** call comes out negative on
            its own merits, pass. Otherwise price the pass exactly as "god"
            does. Applied to the averaged values once all the worlds are in,
            which is what separates it from "floor" -- see `_guard`. Measured:
            euchre rate 21.7%, and **0.28 points a deal worse than "god"** head
            to head. The per-world-versus-decision-level distinction turned out
            to explain almost none of "floor"'s weakness, and the cost is the
            override itself -- taking the least-bad losing call beats passing.
    "floor" "god", but a pass is never worth less than nothing:
            `max(rest_of_auction, 0)`. Same cost as "god", since it runs the
            same solves. Euchre rate falls from 36.7% to 15.3%, and it is the
            **weakest of the four** head to head. Kept documented rather than
            deleted; see CLAUDE.md before reaching for it.

**Neither is clearly stronger.** Head to head against God Mode with the teams
swapped on every deal, "god" scores -1.26 +/- 0.36 points a deal and "zero"
-1.34 +/- 0.36 over the same 50 -- indistinguishable. "zero" looks far better on
mean points *per call* (+0.65 vs -0.17), but that average covers only the deals
a player chose to call and drops whatever passing cost it: the trap `bidding.py`
names when it says passing is not free.

Default is "god" -- the one that actually answers "what happens if I decline".
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

PASS_GOD_MODE = "god"
PASS_ZERO = "zero"
PASS_FLOOR = "floor"
PASS_GUARD = "guard"
PASS_MODELS = (PASS_GOD_MODE, PASS_ZERO, PASS_FLOOR, PASS_GUARD)

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
            arr[seat, i] = r._vec(card, trump)

    trick_cards = np.array(
        [r._vec(card, trump) for _, card in current],
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


# ------------------------------------------------- researched budgets

# How many sampled worlds each kind of decision actually needs before its
# argmax stops moving, measured over 40,390 decisions -- notes/settle_counts.md.
# At these budgets 99.4-99.8% of decisions with a real margin (>0.15) keep the
# leader they finish with. The ones that drift are near-ties, where either
# answer is worth the same by construction.
#
# They live here rather than in one of the scripts because they are a property
# of the player, not of whichever tool is driving it, and they **are** the
# defaults -- for PIMCPlayer itself and for every script that drives it. The
# eyeballed counts the scripts used to carry (20/10 and 24/16) are gone; pass
# --samples / --bid-samples / --discard-samples to override a run.
RESEARCHED_PLAY = 132            # a card:    mean settle 132.3 +/- 4.2
RESEARCHED_BID = 231             # a bid:     mean settle 231.1 +/- 12.3
RESEARCHED_DISCARD = 266         # a discard: mean settle 265.8 +/- 19.9


# --------------------------------------------- sequential elimination

# A PIMC decision averages every option over N sampled worlds and takes the
# best. But the average is not the answer -- the *argmax* is, and that is
# usually settled long before N. Measured over 12 deals at 200 worlds a
# decision: play decisions settled after a median of 1 world and 69% within 20;
# bidding, the expensive half, took a median of 68. Spending the full N on a
# decision already made is most of what a large sample count buys.
#
# So options are dropped as soon as they cannot matter, on two grounds:
#
#   * **it is losing.** The gap to the leader is larger than the noise on the
#     gap, so more worlds will not close it.
#   * **it is close enough.** The gap is smaller than `epsilon`, so picking
#     either one moves this decision's value by less than epsilon. Without
#     this the near-ties -- two cards averaging 0.81 and 0.79 -- run the full
#     budget to resolve a difference smaller than the sweep's own error bar,
#     and they are the bulk of the cost.
#
# The test is on **paired** differences, not on the two means separately: every
# option is scored in the same sampled world, so v_i - v_j has far less noise
# than either mean alone, and is usually exactly 0. The radius is a normal
# interval on that difference plus a 1/n guard term, which is what lets two
# options that have agreed in every world so far be called tied rather than
# waiting for a bound that assumes they might not.
#
# This is a measured approximation, not a proof: `epsilon=None` turns it off
# and restores the exact averaging, which is what the unit tests and the
# pimc_sweep measurements still run on.

MIN_WORLDS = 24          # never drop an option on fewer worlds than this
Z = 2.576                # ~99% normal quantile, on the paired difference
GUARD = 2.0              # 1/n term, so options that never differ can tie out
GROWTH = 1.5             # check for eliminations on a geometric schedule


def _race(options, draw, budget, epsilon=None, min_worlds=MIN_WORLDS,
          z=Z, guard=GUARD):
    """
    Average `draw` over sampled worlds, dropping options that cannot matter.

    `draw(active)` is handed the options still in contention and returns
    {option: value} covering at least those -- a card-play draw solves every
    legal card in one go and returns them all, while a bidding draw evaluates
    only what it was asked for and so actually saves the solves.

    Returns (sums, counts, survivors): per-option totals and how many worlds
    each was scored in, plus the indices still standing. With `epsilon=None`
    nothing is dropped and exactly `budget` worlds are drawn, which keeps the
    rng consumption -- and so every downstream decision -- identical to the
    unraced player.
    """
    k = len(options)
    n = [0] * k
    s = [0] * k
    alive = list(range(k))
    racing = epsilon is not None

    pn = [0] * (k * k)          # paired counts / sums / sums of squares,
    ps = [0] * (k * k)          # stored at [min*k + max] with the difference
    pq = [0] * (k * k)          # taken low-index minus high-index

    drawn = 0
    check_at = min_worlds
    while drawn < budget:
        if racing and len(alive) < 2:
            break               # nothing left to tell apart
        values = draw([options[i] for i in alive])
        drawn += 1
        for i in alive:
            n[i] += 1
            s[i] += values[options[i]]

        if not racing:
            continue

        for a in range(len(alive)):
            i = alive[a]
            vi = values[options[i]]
            for c in range(a + 1, len(alive)):
                j = alive[c]
                d = vi - values[options[j]]
                q = i * k + j if i < j else j * k + i
                if i > j:
                    d = -d
                pn[q] += 1
                ps[q] += d
                pq[q] += d * d

        if drawn < check_at:
            continue
        check_at = max(drawn + 1, int(drawn * GROWTH))

        leader = max(alive, key=lambda i: s[i] / n[i])
        kept = []
        for j in alive:
            if j == leader:
                kept.append(j)
                continue
            q = leader * k + j if leader < j else j * k + leader
            m = pn[q]
            if m < min_worlds:
                kept.append(j)
                continue
            total = ps[q] if leader < j else -ps[q]
            mean_d = total / m
            var = (pq[q] - ps[q] * ps[q] / m) / (m - 1) if m > 1 else 0.0
            radius = z * ((var / m) ** 0.5 if var > 0 else 0.0) + guard / m
            if mean_d - radius < -epsilon:
                kept.append(j)          # still could be worth as much or more
        alive = kept

    return s, n, alive


def _means(options, sums, counts):
    return {o: (sums[i] / counts[i] if counts[i] else 0.0)
            for i, o in enumerate(options)}


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


class GodModePlayer:
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
        pass_model: PASS_GOD_MODE or PASS_ZERO; see the module docstring.
        tie_break: LOW or FIRST, for cards the search rates identically.
        rng: seed it for a reproducible player.

    `solves` and `nodes` accumulate what it has spent. `last_scores` holds the
    averaged value of every option from the most recent decision, so a front
    end or a narrated example can show the working rather than just the answer;
    it is empty when there was nothing to decide.
    """

    def __init__(self, samples: Optional[int] = None,
                 bid_samples: Optional[int] = None,
                 discard_samples: Optional[int] = None,
                 pass_model: str = PASS_GOD_MODE, tie_break: str = LOW,
                 rng: Optional[random.Random] = None,
                 epsilon: Optional[float] = None,
                 min_worlds: int = MIN_WORLDS,
                 prune_discards: bool = False):
        if pass_model not in PASS_MODELS:
            raise ValueError("no such pass model: %r" % (pass_model,))
        if epsilon is not None and epsilon < 0:
            raise ValueError("epsilon must not be negative: %r" % (epsilon,))
        # Each kind defaults to the number of worlds it was measured to need
        # before its argmax stops moving -- notes/settle_counts.md. They are
        # per-kind rather than one number carried across, because bidding and
        # discarding need roughly twice what a card does.
        self.samples = RESEARCHED_PLAY if samples is None else samples
        self.bid_samples = (RESEARCHED_BID if bid_samples is None
                            else bid_samples)
        self.discard_samples = (RESEARCHED_DISCARD if discard_samples is None
                                else discard_samples)
        self.pass_model = pass_model
        self.tie_break = tie_break
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.min_worlds = min_worlds
        # Axiom 1: some optimal discard is never a top trump, so the right
        # bower, left bower and ace of trump can be struck off. Evidence and
        # caveats in notes/discard_dominance.md -- it is an axiom, not a
        # theorem, which is why it is off unless asked for.
        self.prune_discards = prune_discards
        self.solves = 0
        self.nodes = 0
        self.last_scores = {}
        self.last_samples = {}

    # ------------------------------------------------------------ helpers

    def _worlds(self, observation: obs.Observation, n: int):
        return obs.sample_worlds(observation, n, self.rng)

    def _world_stream(self, observation: obs.Observation):
        """
        Worlds one at a time, so stopping early costs nothing.

        A decision that settles after 20 of its 10,000 worlds should not have
        paid to draw the other 9,980, and `sample_worlds` handing back a
        finished list meant it always had.
        """
        return obs.iter_worlds(observation, self.rng)

    def _deal_of(self, world: obs.World, observation: obs.Observation) -> Deal:
        """A sampled world, dressed as a `game.Deal` the auction can run on."""
        buried = tuple(c for c in world.kitty if c != observation.up_card)
        return Deal(hands=world.hands, up_card=observation.up_card,
                    buried=buried, dealer=observation.dealer,
                    picked_up=False).check()

    # --------------------------------------------------------------- bid

    def bid(self, turn: "t.BidTurn") -> "t.Bid":
        observation = turn.observation
        stream = self._world_stream(observation)

        def draw(active):
            deal = self._deal_of(next(stream), observation)
            return {o: self._bid_value(o, deal, turn) for o in active}

        sums, counts, alive = _race(turn.options, draw, self.bid_samples,
                                    self.epsilon, self.min_worlds)
        self.last_scores = _means(turn.options, sums, counts)
        self.last_samples = {o: counts[i] for i, o in enumerate(turn.options)}
        live = tuple(turn.options[i] for i in alive)

        if self.pass_model == PASS_GUARD:
            forced = self._guard(turn)
            if forced is not None:
                return forced
        return _pick(self.last_scores, live)

    def _guard(self, turn: "t.BidTurn"):
        """
        PASS_GUARD: never take a call that is negative on its own merits.

        The rule is "if every call has negative EV, pass; otherwise evaluate
        passing normally". It is a **decision-level** override, applied to the
        averaged values after all the worlds are in, and that is the whole
        difference between it and PASS_FLOOR.

        PASS_FLOOR clamps inside `_bid_value`, which runs once per sampled
        world, so it computes `sum(max(g_w, 0))` -- every individual world where
        passing went badly is thrown away and replaced by zero. This computes
        at most `max(sum(g_w), 0)`, and by Jensen those are not the same
        number: the per-world clamp is systematically far kinder to passing,
        which is why `"floor"` under-calls and this does not.

        Returns the pass option when the guard fires, else None.
        """
        passes = [o for o in turn.options if o.action == t.PASS]
        calls = [o for o in turn.options if o.action != t.PASS]
        if not passes or not calls:
            return None          # stick-the-dealer leaves nothing to guard
        if all(self.last_scores.get(o, 0.0) < 0 for o in calls):
            return passes[0]
        return None

    def _bid_value(self, option: "t.Bid", deal: Deal,
                   turn: "t.BidTurn") -> int:
        """What one option is worth in one imagined deal, on this seat's scale."""
        if option.action == t.PASS:
            if self.pass_model == PASS_ZERO:
                return 0
            self.solves += 1
            rest = b.rest_of_auction(
                deal, turn.index + 1, turn.order, turn.stick_the_dealer,
                turn.allow_loners, turn.bidding_round, self.prune_discards)
            if self.pass_model == PASS_FLOOR:
                # Declining cannot be worth less than nothing. "god" prices a
                # pass at whatever the rest of the auction does, and inside a
                # sampled world the other seats can see this hand -- so that
                # branch reads "an opponent calls this and makes it" far more
                # often than a real table would, and a seat with a bad hand
                # ends up making a desperate call because passing looked worse.
                # The floor removes only that pessimism and leaves the rest.
                return max(b.value_to(turn.seat, rest.value), 0)
            value = rest.value
        elif option.action == t.ORDER:
            self.solves += 1
            value = b.order_up(deal, turn.seat, option.alone,
                               self.prune_discards)[0]
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
        stream = self._world_stream(observation)
        candidates = turn.options
        if self.prune_discards:
            tops = set(b.top_trumps(turn.trump))
            candidates = tuple(c for c in candidates
                               if c not in tops) or turn.options

        def draw(active):
            # The world holds the dealer's six; the deal it came from held five.
            world = next(stream)
            dealt = tuple(c for c in world.hands[turn.seat] if c != up_card)
            hands = tuple(dealt if s == turn.seat else world.hands[s]
                          for s in range(PLAYERS))
            before = Deal(hands=hands, up_card=up_card,
                          buried=world.kitty, dealer=turn.seat,
                          picked_up=False).check()

            out = {}
            for card in active:
                self.solves += 1
                value = b.play_value(before.pick_up(discard=card), turn.trump,
                                     turn.caller, turn.alone)
                out[card] = to_seat(value, turn.caller, turn.seat)
            return out

        sums, counts, alive = _race(candidates, draw, self.discard_samples,
                                    self.epsilon, self.min_worlds)
        self.last_scores = _means(candidates, sums, counts)
        self.last_samples = {c: counts[i] for i, c in enumerate(candidates)}
        live = tuple(candidates[i] for i in alive)
        return _pick(self.last_scores, live, tie_break=self.tie_break,
                     trump=turn.trump)

    # -------------------------------------------------------------- play

    def play(self, turn: "t.PlayTurn") -> r.Card:
        if len(turn.legal) == 1:
            # Nothing to think about, and thinking costs a few hundred solves.
            self.last_scores = {}
            return turn.legal[0]

        observation = turn.observation
        stream = self._world_stream(observation)

        def draw(active):
            # One solve prices every legal card at once, so `active` cannot
            # narrow the work here -- the saving is in stopping early instead.
            values, nodes = _card_values(list(next(stream).hands), turn)
            self.solves += len(values)
            self.nodes += nodes
            return {c: to_seat(v, turn.caller, turn.seat)
                    for c, v in values.items()}

        sums, counts, alive = _race(turn.legal, draw, self.samples,
                                    self.epsilon, self.min_worlds)
        self.last_scores = _means(turn.legal, sums, counts)
        self.last_samples = {c: counts[i] for i, c in enumerate(turn.legal)}
        live = tuple(turn.legal[i] for i in alive)
        return _pick(self.last_scores, live, tie_break=self.tie_break,
                     trump=turn.trump)


class ForcedOpeningBid:
    """
    A player whose **first** bid is pinned; everything after it is its own.

    This is how "what is this hand worth if I order it up" gets asked, as
    opposed to "what happens to this hand at a table". The two are different
    questions and `hand_ev.py` answers the second by default: it walks the
    auction, so the reported mean mixes the deals the seat called with the ones
    it passed and somebody else called. Pinning the opening bid conditions on
    the call instead.

    Only the opening bid is forced. The seat still discards and plays for
    itself, every other seat bids normally, and -- crucially -- the dealer
    still chooses its own discard through `table._settle_order`, so an
    opposing dealer still pitches to hurt the contract. Nothing about the sim
    is bypassed except the one decision being conditioned on.

    `forced` records whether the pin actually fired. It will not when an
    earlier seat has already ended the auction, which cannot happen from the
    eldest seat but can from any other, and a sweep that silently averaged
    those in would not be answering the question it was asked.
    """

    def __init__(self, inner, action: str, alone: bool = False,
                 suit: Optional[int] = None):
        self.inner = inner
        self.action = action
        self.alone = alone
        self.suit = suit
        self.spoken = False
        self.forced = False

    def _match(self, options):
        for option in options:
            if option.action != self.action or bool(option.alone) != self.alone:
                continue
            if self.suit is not None and option.suit != self.suit:
                continue
            return option
        return None

    def bid(self, turn):
        if not self.spoken:
            self.spoken = True
            pinned = self._match(turn.options)
            if pinned is not None:
                self.forced = True
                return pinned
        return self.inner.bid(turn)

    def discard(self, turn):
        return self.inner.discard(turn)

    def play(self, turn):
        return self.inner.play(turn)

    @property
    def last_scores(self):
        return getattr(self.inner, "last_scores", {})

    @property
    def solves(self):
        return getattr(self.inner, "solves", 0)


def table_of(factory, n: int = PLAYERS, seed: Optional[int] = None):
    """
    Four players from one factory, called with the seat number.

    Seeding per seat rather than globally keeps a sweep reproducible even
    though seats consume random numbers at rates that depend on their choices.
    """
    return [factory(seat) for seat in range(n)]
