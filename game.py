"""
The dealt state of a Euchre hand: four hands, an up-card, a kitty, a dealer.

`dealer.py` deals 20 canonical vectors and drops the other four on the floor.
That is enough to solve trick-play with trump already chosen, but it cannot
support bidding: with no named up-card there is nothing to order up, and with no
dealer seat there is no bidding order and nobody to pick up and discard.

This module deals *natural* cards -- `rotation.Card`, a real (suit, rank) pair --
because at deal time nobody has called anything yet. Trump only enters when a
hand is rotated for the solver, which is `rotation.deal_to_engine`'s job.

    >>> d = deal_random(rng=random.Random(0), dealer=3)
    >>> d.up_card                       # everyone can see this
    Card(suit=1, rank=13)
    >>> d.bidding_order()               # starts left of the dealer
    [0, 1, 2, 3]
    >>> d = d.pick_up(discard=d.hands[3][0])
    >>> len(d.hands[3])                 # still five
    5

All 24 cards are always accounted for -- see `all_cards` and the invariant
`check` enforces. A deal that quietly loses a card produces a game that cannot
be reasoned about, and the old dealer had exactly that bug twice.
"""
import random
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from rotation import Card, card_name, full_deck

HAND_SIZE = 5
PLAYERS = 4
KITTY_SIZE = 4          # the up-card plus the three buried under it
DECK_SIZE = 24


@dataclass(frozen=True)
class Deal:
    """
    One dealt Euchre hand, before or after the dealer picks up.

    Attributes:
        hands: four hands of five natural cards, indexed by seat.
        up_card: the card turned up off the kitty. Everyone has seen it, so it
            stays recorded even after it is picked up or turned down.
        buried: the rest of the kitty, face down. Three cards before the pickup,
            four after (the dealer's discard joins them).
        dealer: the seat that dealt, 0-3. Bidding starts to its left.
        picked_up: True once the dealer has taken the up-card into hand.
    """

    hands: Tuple[Tuple[Card, ...], ...]
    up_card: Card
    buried: Tuple[Card, ...]
    dealer: int
    picked_up: bool = False

    # -------------------------------------------------------------- seats

    @property
    def first_bidder(self) -> int:
        """Eldest hand -- the seat to the dealer's left, who speaks first."""
        return (self.dealer + 1) % PLAYERS

    def bidding_order(self):
        """The four seats in bidding order, starting left of the dealer."""
        return [(self.dealer + 1 + i) % PLAYERS for i in range(PLAYERS)]

    def partner(self, seat: int) -> int:
        return (seat + 2) % PLAYERS

    # -------------------------------------------------------------- cards

    def all_cards(self):
        """Every card in the deal. Always the full 24."""
        cards = [c for hand in self.hands for c in hand]
        cards.extend(self.buried)
        if not self.picked_up:
            # still on the kitty; once picked up it lives in the dealer's hand
            cards.append(self.up_card)
        return cards

    def check(self):
        """Raise unless this is a structurally sound deal."""
        if len(self.hands) != PLAYERS:
            raise ValueError("expected %d hands, got %d" % (PLAYERS, len(self.hands)))
        for seat, hand in enumerate(self.hands):
            if len(hand) != HAND_SIZE:
                raise ValueError(
                    "seat %d holds %d cards, expected %d"
                    % (seat, len(hand), HAND_SIZE))
        if not 0 <= self.dealer < PLAYERS:
            raise ValueError("dealer must be a seat 0-%d, got %r"
                             % (PLAYERS - 1, self.dealer))

        want_buried = KITTY_SIZE - 1 + (1 if self.picked_up else 0)
        if len(self.buried) != want_buried:
            raise ValueError("expected %d buried cards, got %d"
                             % (want_buried, len(self.buried)))

        cards = self.all_cards()
        if len(cards) != DECK_SIZE:
            raise ValueError("deal holds %d cards, expected %d"
                             % (len(cards), DECK_SIZE))
        if len(set(cards)) != DECK_SIZE:
            raise ValueError("deal contains duplicate cards")
        if set(cards) != set(full_deck()):
            raise ValueError("deal is not made from the Euchre deck")
        return self

    # -------------------------------------------------------------- pickup

    def pick_up(self, discard: Card) -> "Deal":
        """
        The dealer takes the up-card and discards, returning a new Deal.

        This is the mechanical half of ordering up -- who decides to order, and
        which card the dealer throws, are decisions that belong to a player.
        """
        if self.picked_up:
            raise ValueError("the up-card has already been picked up")

        hand = list(self.hands[self.dealer]) + [self.up_card]
        if discard not in hand:
            raise ValueError(
                "the dealer cannot discard %s, which it does not hold"
                % card_name(discard))
        hand.remove(discard)

        hands = list(self.hands)
        hands[self.dealer] = tuple(hand)
        return replace(self,
                       hands=tuple(hands),
                       buried=self.buried + (discard,),
                       picked_up=True).check()

    def turn_down(self) -> "Deal":
        """
        Nobody ordered it up. The up-card is out of play but stays on record --
        every seat saw it, which is information the bidding depends on.
        """
        if self.picked_up:
            raise ValueError("the up-card has already been picked up")
        return self

    # -------------------------------------------------------------- display

    def describe(self) -> str:
        lines = []
        for seat, hand in enumerate(self.hands):
            mark = " (dealer)" if seat == self.dealer else ""
            lines.append("  seat %d%s: %s"
                         % (seat, mark, " ".join(card_name(c) for c in hand)))
        lines.append("  up-card: %s%s"
                     % (card_name(self.up_card),
                        " (picked up)" if self.picked_up else ""))
        lines.append("  buried:  %s"
                     % " ".join(card_name(c) for c in self.buried))
        return "\n".join(lines)


def deal_from_order(cards, dealer: int = 0) -> Deal:
    """
    Deal from an explicit 24-card ordering: five to each seat in turn, then the
    kitty, with the next card turned up. Deterministic, so tests can pin a deal.
    """
    cards = list(cards)
    if len(cards) != DECK_SIZE:
        raise ValueError("need all %d cards, got %d" % (DECK_SIZE, len(cards)))

    hands = tuple(tuple(cards[i * HAND_SIZE:(i + 1) * HAND_SIZE])
                  for i in range(PLAYERS))
    kitty = cards[PLAYERS * HAND_SIZE:]
    return Deal(hands=hands,
                up_card=kitty[0],
                buried=tuple(kitty[1:]),
                dealer=dealer).check()


def deal_random(rng=None, dealer: int = 0) -> Deal:
    """Shuffle and deal. Pass a seeded `random.Random` for a reproducible deal."""
    rng = rng or random.Random()
    cards = full_deck()
    rng.shuffle(cards)
    return deal_from_order(cards, dealer=dealer)


def deal_around(known_hand=None, seat: int = 0, up_card: Optional[Card] = None,
                rng=None, dealer: int = 0) -> Deal:
    """
    Deal with part of the layout fixed and the rest random.

    This is the shape of the question the calculator exists to answer: "I hold
    these five and this is turned up -- what can I expect?" Everything not
    pinned is dealt uniformly from what remains.

    Args:
        known_hand: cards to give `seat`, up to five of them.
        seat: which seat holds `known_hand`.
        up_card: the card to turn up.
        rng: a seeded `random.Random` for reproducibility.
        dealer: the dealing seat.
    """
    rng = rng or random.Random()
    known_hand = list(known_hand or [])

    if not 0 <= seat < PLAYERS:
        raise ValueError("seat must be 0-%d, got %r" % (PLAYERS - 1, seat))
    if len(known_hand) > HAND_SIZE:
        raise ValueError("a hand holds at most %d cards, got %d"
                         % (HAND_SIZE, len(known_hand)))

    pinned = known_hand + ([up_card] if up_card is not None else [])
    if len(set(pinned)) != len(pinned):
        raise ValueError("the same card was pinned twice")

    remaining = [c for c in full_deck() if c not in set(pinned)]
    if len(remaining) != DECK_SIZE - len(pinned):
        raise ValueError("a pinned card is not in the Euchre deck")
    rng.shuffle(remaining)

    hands = []
    for s in range(PLAYERS):
        if s == seat:
            fill = HAND_SIZE - len(known_hand)
            hands.append(tuple(known_hand + [remaining.pop() for _ in range(fill)]))
        else:
            hands.append(tuple(remaining.pop() for _ in range(HAND_SIZE)))

    turned = up_card if up_card is not None else remaining.pop()
    return Deal(hands=tuple(hands),
                up_card=turned,
                buried=tuple(remaining),
                dealer=dealer).check()
