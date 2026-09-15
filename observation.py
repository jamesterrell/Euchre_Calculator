"""
What one seat knows, and the layouts that are consistent with it.

`game.Deal` is the deal in God Mode. This module is the deal as a *player* sees
it: their own five cards, the up-card and what became of it, every card played
face up, and nothing else. `bidding.py` and `fast_search.py` both take the God
Mode view -- the right baseline and the wrong opponent. An `Observation` is the
other view, and `sample_worlds` turns it into concrete deals a solver can take.

That pairing is the whole idea behind the Perfect Information Monte Carlo sim:
a player cannot solve the hand it is in, because it does not know the hand it
is in. So it invents a few dozen hands it *could* be in, solves each exactly,
and plays the card that does best on average. The solving already existed; the
inventing was what was missing, and inventing badly is the easy way to get a
plausible-looking player that is quietly cheating or quietly stupid.

Three kinds of inference go into a sampled world, all real Euchre rather than
bookkeeping:

  * **Counts.** Every seat has played the same number of cards, so how many
    each still holds is public.
  * **Voids.** A seat that failed to follow a led suit holds none of it, for
    the rest of the hand. Read in *effective* suits, so the left bower counts
    as trump: following a club lead cannot be done with it, and ruffing a club
    lead with it shows no club void.
  * **The up-card.** Turned down, it is buried and nobody holds it. Ordered up,
    the dealer took it, and unless the dealer has since shown out of trump it
    is still in their hand -- much sharper than letting it float free.

Inference from the *bidding* is deliberately not modelled: worlds are drawn as
though the auction said nothing about anybody's cards. That makes PIMC players
weaker than they could be, in a specific direction -- they under-rate the
caller. Fixing it needs a bidding model, which is the thing this project is
trying to produce, so the circularity is left open rather than guessed at.

Imports `rotation` and `game` and nothing heavier -- no numpy, no numba, no
solver. A front end can ask what a seat knows without compiling anything.
"""
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import rotation as r
from game import DECK_SIZE, HAND_SIZE, PLAYERS

# What became of the up-card. UP and TURNED_DOWN both mean "on the kitty, and
# everyone knows it"; they differ only in whether the auction is still in round
# one, which matters to bidding and not to sampling.
UP = "up"
PICKED_UP = "picked_up"
TURNED_DOWN = "turned_down"

KITTY = PLAYERS          # the index sample_worlds uses for the buried pile


def effective_suit(card: r.Card, trump: Optional[int]) -> int:
    """
    The suit a card follows and is followed by, once trump is called.

    One card disagrees with its printed suit, and it is why this exists rather
    than `card.suit`: the left bower is trump. A hand holding the jack of clubs
    with spades called is void in clubs as far as the rules care, and a sampler
    using the printed suit would deal it clubs it cannot hold.
    """
    if trump is None:
        return card.suit
    if card.rank == r.JACK and card.suit == r.same_colour(trump):
        return trump
    return card.suit


@dataclass(frozen=True)
class World:
    """
    One layout consistent with an Observation: a guess at the whole table.

    `hands` holds the cards each seat *still has*, so it is ragged in the
    middle of a trick -- the seats that have already played to it hold one
    fewer. The observer's own entry is its real hand, not a guess.
    """

    hands: Tuple[Tuple[r.Card, ...], ...]
    kitty: Tuple[r.Card, ...]

    def counts(self) -> List[int]:
        return [len(h) for h in self.hands]


@dataclass(frozen=True)
class Observation:
    """
    Everything seat `seat` has seen, and nothing it has not.

    Attributes:
        seat: whose view this is.
        hand: the cards that seat still holds.
        dealer: the dealing seat.
        up_card: the card that was turned up. Always known to everyone.
        up_state: UP, PICKED_UP or TURNED_DOWN.
        plays: every card played so far, in play order, as (seat, card).
            Trick boundaries are implied -- every trick is the same width.
        trump: the called suit, or None while round one is still running.
        caller: the seat that called it, or None.
        alone: whether the caller is playing alone.
        discard: the card this seat buried, which only the dealer knows and
            only after picking up.
        pending_discard: True in the one moment the dealer holds six cards --
            picked up, not yet thrown. It is a real decision point with a real
            information set, so it gets to be an Observation rather than a
            special case bolted onto the player.
    """

    seat: int
    hand: Tuple[r.Card, ...]
    dealer: int
    up_card: r.Card
    up_state: str = UP
    plays: Tuple[Tuple[int, r.Card], ...] = ()
    trump: Optional[int] = None
    caller: Optional[int] = None
    alone: bool = False
    discard: Optional[r.Card] = None
    pending_discard: bool = False

    # ------------------------------------------------------------ structure

    @property
    def sitting(self) -> Optional[int]:
        """The seat sitting out a loner, if there is one."""
        if not self.alone or self.caller is None:
            return None
        return (self.caller + 2) % PLAYERS

    @property
    def width(self) -> int:
        """Cards in a complete trick."""
        return 3 if self.sitting is not None else 4

    @property
    def trick_no(self) -> int:
        """Which trick is on the table, 0-based."""
        return len(self.plays) // self.width

    @property
    def current_trick(self) -> Tuple[Tuple[int, r.Card], ...]:
        """The cards played to the trick now in progress, in order."""
        return self.plays[self.trick_no * self.width:]

    def completed_tricks(self) -> List[Tuple[Tuple[int, r.Card], ...]]:
        """The finished tricks, each a tuple of (seat, card) in play order."""
        w = self.width
        return [tuple(self.plays[i * w:(i + 1) * w])
                for i in range(self.trick_no)]

    def counts(self) -> List[int]:
        """How many cards each seat still holds. Public information."""
        held = [HAND_SIZE] * PLAYERS
        for seat, _ in self.plays:
            held[seat] -= 1
        if self.pending_discard:
            held[self.seat] += 1
        return held

    # ------------------------------------------------------------ inference

    def voids(self) -> Dict[int, FrozenSet[int]]:
        """
        The effective suits each seat has shown it cannot hold.

        A seat that discards or ruffs on a led suit is void in it for the rest
        of the hand -- there is no getting more of a suit back in Euchre. The
        observer's own entry is included and is always derivable from its hand,
        which makes it a free self-check rather than a special case.
        """
        out = {seat: set() for seat in range(PLAYERS)}
        for trick in self.completed_tricks() + [self.current_trick]:
            if not trick:
                continue
            led = effective_suit(trick[0][1], self.trump)
            for seat, card in trick[1:]:
                if effective_suit(card, self.trump) != led:
                    out[seat].add(led)
        return {seat: frozenset(s) for seat, s in out.items()}

    def played_cards(self) -> List[r.Card]:
        return [card for _, card in self.plays]

    def known_kitty(self) -> List[r.Card]:
        """
        The buried cards this seat can name.

        Never more than one: the up-card if it was turned down, or the
        dealer's own discard if this seat is the dealer. The other three are
        face down and were face down when they were dealt.
        """
        known = []
        if self.up_state in (UP, TURNED_DOWN):
            known.append(self.up_card)
        if self.discard is not None:
            known.append(self.discard)
        return known

    def seen(self) -> List[r.Card]:
        """Every card whose location this seat knows."""
        return list(self.hand) + self.played_cards() + self.known_kitty()

    def unseen(self) -> List[r.Card]:
        """
        The cards this seat cannot place: everything it has not seen.

        The up-card can be in here. If it was ordered up it is in the dealer's
        hand or in the dealer's discard, and which of those is exactly the
        thing the observer does not know.
        """
        placed = set(self.seen())
        return [c for c in r.full_deck() if c not in placed]

    def check(self):
        """Raise unless this observation is internally consistent."""
        if not 0 <= self.seat < PLAYERS:
            raise ValueError("seat must be 0-%d, got %r"
                             % (PLAYERS - 1, self.seat))
        if not 0 <= self.dealer < PLAYERS:
            raise ValueError("dealer must be a seat 0-%d, got %r"
                             % (PLAYERS - 1, self.dealer))
        if self.up_state not in (UP, PICKED_UP, TURNED_DOWN):
            raise ValueError("no such up-card state: %r" % (self.up_state,))
        if self.discard is not None and self.seat != self.dealer:
            raise ValueError("seat %d is not the dealer and buried nothing"
                             % self.seat)
        if self.discard is not None and self.up_state != PICKED_UP:
            raise ValueError("nothing is buried until the up-card is taken")
        if self.pending_discard:
            if self.seat != self.dealer:
                raise ValueError("only the dealer picks the up-card up")
            if self.up_state != PICKED_UP:
                raise ValueError("nothing is pending until the up-card is taken")
            if self.discard is not None:
                raise ValueError("the discard is either pending or made")
            if self.plays:
                raise ValueError("the discard is made before a card is played")

        if self.counts()[self.seat] != len(self.hand):
            raise ValueError(
                "seat %d has played %d cards but holds %d"
                % (self.seat, HAND_SIZE - self.counts()[self.seat],
                   len(self.hand)))

        seen = self.seen()
        if len(set(seen)) != len(seen):
            raise ValueError("the same card was seen in two places")
        for card in seen:
            if card.suit not in r.SUITS or card.rank not in r.RANKS:
                raise ValueError("not a Euchre card: %r" % (card,))

        sitting = self.sitting
        if sitting is not None:
            for seat, _ in self.plays:
                if seat == sitting:
                    raise ValueError("seat %d is sitting out and cannot play"
                                     % sitting)
        return self


# --------------------------------------------------------------- sampling


def _dealer_holds_up_card(obs: Observation) -> bool:
    """
    Should a sampled world put the up-card back in the dealer's hand?

    Only when it was ordered up, has not since been played, and the dealer
    still could be holding it. A dealer who has failed to follow trump has
    shown it cannot be -- it must have been the card they buried. That case is
    rare and the inference is free, so it is made rather than assumed away.
    """
    if obs.up_state != PICKED_UP:
        return False
    if obs.seat == obs.dealer:
        return False                       # the dealer knows; no guessing
    if obs.up_card in set(obs.played_cards()):
        return False                       # public now
    if obs.counts()[obs.dealer] == 0:
        # The dealer has played its last card and this was not one of them, so
        # there is nothing left for it to be in. It was the discard.
        return False
    if obs.trump is not None:
        shown = obs.voids().get(obs.dealer, frozenset())
        if effective_suit(obs.up_card, obs.trump) in shown:
            return False                   # it went under, not into the hand
    return True


def _capacities(obs: Observation) -> List[int]:
    """
    How many unknown cards each slot absorbs: one per seat, then the kitty.

    The observer's own slot is zero -- it is holding its real cards. A loner's
    sitting partner still takes five, because those five are out of circulation
    for everybody else even though the seat never plays one. Forgetting that is
    a subtle way to make every other hand slightly too good.
    """
    counts = obs.counts()
    caps = [0] * (PLAYERS + 1)
    for seat in range(PLAYERS):
        caps[seat] = 0 if seat == obs.seat else counts[seat]
    # Whatever the unseen cards do not owe the other seats, they owe the kitty.
    # Deriving it rather than counting it keeps the one case where the pile is
    # temporarily three cards deep -- the dealer picked up and has not yet
    # thrown -- from needing a rule of its own.
    caps[KITTY] = len(obs.unseen()) - sum(caps[:PLAYERS])
    if caps[KITTY] < 0:
        raise RuntimeError("the seats between them hold more cards than are "
                           "left unseen; the observation is inconsistent")
    return caps


def _allowed(card: r.Card, slot: int, obs: Observation,
             voids: Dict[int, FrozenSet[int]]) -> bool:
    """Can `slot` hold `card` without contradicting something already seen?"""
    if slot == KITTY:
        return True
    if slot == obs.sitting:
        return True                        # never plays, so never shows out
    return effective_suit(card, obs.trump) not in voids.get(slot, frozenset())


def _deal_out(pool: List[r.Card], caps: List[int], obs: Observation,
              voids, rng: random.Random) -> Optional[List[List[r.Card]]]:
    """
    One attempt at dealing `pool` into the slots without breaking a void.

    Cards are placed most-constrained first -- the ones fewest slots will take
    go while there is still room to take them -- and among the slots that will
    take a card, one is drawn with probability proportional to the room it has
    left. That weighting is what an ordinary shuffle-and-deal does, so with no
    voids in play this is uniform over layouts. With voids it is close but not
    exact, and `sample_worlds` says so rather than claiming otherwise.

    Returns None if it paints itself into a corner, which the caller retries.
    """
    room = list(caps)
    out = [[] for _ in range(PLAYERS + 1)]

    order = sorted(
        pool,
        key=lambda c: (sum(1 for s in range(PLAYERS + 1)
                           if caps[s] and _allowed(c, s, obs, voids)),
                       rng.random()))

    for card in order:
        slots = [s for s in range(PLAYERS + 1)
                 if room[s] and _allowed(card, s, obs, voids)]
        if not slots:
            return None
        weights = [room[s] for s in slots]
        slot = rng.choices(slots, weights=weights, k=1)[0]
        out[slot].append(card)
        room[slot] -= 1

    return out


def sample_world(obs: Observation, rng: Optional[random.Random] = None,
                 attempts: int = 200) -> World:
    """
    Draw one layout of the unseen cards that is consistent with `obs`.

    Consistent means: the counts are right, nobody is dealt a suit they have
    already shown out of, the buried cards stay buried, and the up-card is
    where the table watched it go. It does **not** mean the layout is likely
    given the bidding -- see the module docstring.

    The draw is close to uniform over consistent layouts but not exactly so.
    Placement is sequential with the most-constrained card first, which is
    exactly uniform when no void is in play and slightly biased when one is.
    Raises RuntimeError if `attempts` deals all fail, which means the
    observation is contradictory rather than merely hard.
    """
    rng = rng or random.Random()
    obs.check()

    pool = obs.unseen()
    caps = _capacities(obs)
    voids = obs.voids()

    forced = []
    if _dealer_holds_up_card(obs):
        forced.append((obs.up_card, obs.dealer))
        pool = [c for c in pool if c != obs.up_card]
        caps[obs.dealer] -= 1
        if caps[obs.dealer] < 0:
            raise RuntimeError("the dealer cannot be holding the up-card")

    if sum(caps) != len(pool):
        raise RuntimeError(
            "%d unseen cards do not fill %d slots -- the observation is "
            "inconsistent" % (len(pool), sum(caps)))

    for _ in range(attempts):
        dealt = _deal_out(pool, caps, obs, voids, rng)
        if dealt is not None:
            break
    else:
        raise RuntimeError(
            "could not place %d unseen cards in %d attempts without breaking a "
            "void; seat %d's observation may be contradictory"
            % (len(pool), attempts, obs.seat))

    for card, seat in forced:
        dealt[seat].append(card)

    hands = []
    for seat in range(PLAYERS):
        hands.append(tuple(obs.hand) if seat == obs.seat
                     else tuple(dealt[seat]))
    kitty = tuple(dealt[KITTY]) + tuple(obs.known_kitty())
    return World(hands=tuple(hands), kitty=kitty)


def sample_worlds(obs: Observation, n: int,
                  rng: Optional[random.Random] = None,
                  attempts: int = 200) -> List[World]:
    """`n` independent draws from `sample_world`. Duplicates are not filtered."""
    rng = rng or random.Random()
    return [sample_world(obs, rng, attempts) for _ in range(n)]


def check_world(obs: Observation, world: World):
    """
    Raise unless `world` could be the real table, given what `obs` has seen.

    This is the test oracle for the sampler, and it is written from the
    observation rather than from the sampler's own bookkeeping so that the two
    can disagree. Card conservation is checked the same way `game.Deal.check`
    does it, for the same reason: a layout that quietly loses or duplicates a
    card solves to a number that means nothing.
    """
    counts = obs.counts()
    for seat in range(PLAYERS):
        if len(world.hands[seat]) != counts[seat]:
            raise ValueError("seat %d holds %d cards, should hold %d"
                             % (seat, len(world.hands[seat]), counts[seat]))

    if tuple(world.hands[obs.seat]) != tuple(obs.hand):
        raise ValueError("the observer's own hand was replaced by a guess")

    every = ([c for h in world.hands for c in h]
             + list(world.kitty) + obs.played_cards())
    if len(every) != DECK_SIZE:
        raise ValueError("the world holds %d cards, expected %d"
                         % (len(every), DECK_SIZE))
    if set(every) != set(r.full_deck()):
        raise ValueError("the world is not made from the Euchre deck")

    voids = obs.voids()
    for seat in range(PLAYERS):
        if seat == obs.sitting:
            continue
        for card in world.hands[seat]:
            suit = effective_suit(card, obs.trump)
            if suit in voids.get(seat, frozenset()):
                raise ValueError(
                    "seat %d was dealt %s but has shown out of %s"
                    % (seat, r.card_name(card), r.suit_name(suit)))

    for card in obs.known_kitty():
        if card not in world.kitty:
            raise ValueError("%s is buried but the world puts it elsewhere"
                             % r.card_name(card))

    if _dealer_holds_up_card(obs) and obs.up_card not in world.hands[obs.dealer]:
        raise ValueError("the dealer took %s but the world puts it elsewhere"
                         % r.card_name(obs.up_card))
    return world


# ------------------------------------------------------- from the God's eye


def observe(deal, seat: int, plays=(), trump=None, caller=None, alone=False,
            up_state=UP, discard=None) -> Observation:
    """
    Take one seat's view of a `game.Deal` -- the narrowing that makes a player.

    `deal.hands[seat]` is the hand as dealt; the cards this seat has already
    played are removed from it, so the result is what the seat is holding now.
    Everything else about the deal is dropped on the floor, which is the point.
    """
    spent = {card for who, card in plays if who == seat}
    hand = tuple(c for c in deal.hands[seat] if c not in spent)
    return Observation(
        seat=seat,
        hand=hand,
        dealer=deal.dealer,
        up_card=deal.up_card,
        up_state=up_state,
        plays=tuple(plays),
        trump=trump,
        caller=caller,
        alone=alone,
        discard=discard,
    ).check()
