"""
Rotate a real Euchre hand into the solver's canonical frame, and back.

`fast_search` is written for a single fixed trump suit: spades. Everything
downstream of it -- the encoding, `norm > 80` being a trump test, the left bower
living at `[0, 135]` -- assumes that. Real games call any of the four suits, so
a hand has to be rotated into the canonical frame before it can be solved. That
obligation used to be unwritten, with no code behind it; this module is the code.

Rotation is not a plain suit relabel. The left bower is the jack of the suit
*the same colour as trump*, so which jack leaves its own suit changes with the
call:

    trump      right bower   left bower   suit that loses its jack
    spades     JS            JC           clubs
    clubs      JC            JS           spades
    hearts     JH            JD           diamonds
    diamonds   JD            JH           hearts

So the same-colour suit always maps onto the canonical clubs axis (the one
`deck.py` gives only five cards), the trump suit maps onto the trump axis, and
the two off-colour suits map onto the hearts and diamonds axes, which keep all
six ranks. The mapping is a bijection onto `full_euchre_deck` for every trump
suit, and it is the identity when trump is spades.

Cards here are natural (suit, rank) pairs -- what a player actually holds --
which is a different thing from the canonical vectors in `deck.py`. Those
vectors already encode a trump call; these do not.

    >>> hand = parse_hand("JH JD AH 9S TC")
    >>> to_engine(hand, HEARTS)          # hearts called
    array([[  0, 140], [  0, 135], [  0, 130], [ -9,   0], [ 10,   0]])

Imports numpy and nothing from this repo.
"""
from collections import namedtuple

import numpy as np

# Natural suits. Ordered so that spades-as-trump comes out as the identity
# mapping against deck.py; see _plain_axes.
CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3
SUITS = (CLUBS, DIAMONDS, HEARTS, SPADES)
SUIT_LETTERS = "CDHS"

NINE, TEN, JACK, QUEEN, KING, ACE = 9, 10, 11, 12, 13, 14
RANKS = (NINE, TEN, JACK, QUEEN, KING, ACE)
RANK_LETTERS = {NINE: "9", TEN: "T", JACK: "J", QUEEN: "Q", KING: "K", ACE: "A"}
LETTER_RANKS = {v: k for k, v in RANK_LETTERS.items()}

# Strength on the trump axis. The jack is absent because both bowers are
# special-cased: the right bower is 140 and the left is 135, above the ace.
TRUMP_STRENGTH = {NINE: 90, TEN: 100, QUEEN: 110, KING: 120, ACE: 130}
STRENGTH_RANK = {v: k for k, v in TRUMP_STRENGTH.items()}
RIGHT_BOWER = 140
LEFT_BOWER = 135

Card = namedtuple("Card", "suit rank")


def same_colour(suit):
    """The other suit of the same colour -- whose jack becomes the left bower."""
    return {SPADES: CLUBS, CLUBS: SPADES, HEARTS: DIAMONDS, DIAMONDS: HEARTS}[suit]


# American spelling, since the rest of the repo uses it in places.
same_color = same_colour


def _plain_axes(trump):
    """
    The two off-colour suits, as (diamonds_axis_suit, hearts_axis_suit).

    They are interchangeable as far as the solver is concerned -- both are plain
    suits that never interact -- so the only requirement is that the choice be
    deterministic, to keep the rotation invertible. Taking them in suit order
    also makes spades-as-trump the identity: with trump spades the off-colour
    pair is {diamonds, hearts}, and diamonds lands on the +x axis exactly where
    deck.py puts it.
    """
    off = sorted(set(SUITS) - {trump, same_colour(trump)})
    return off[0], off[1]


def card_to_engine(card, trump):
    """One natural card -> its canonical [x, y] vector under `trump`."""
    suit, rank = card
    if suit not in SUITS:
        raise ValueError("no such suit: %r" % (suit,))
    if rank not in RANKS:
        raise ValueError("no such rank: %r" % (rank,))
    if trump not in SUITS:
        raise ValueError("no such trump suit: %r" % (trump,))

    left_suit = same_colour(trump)

    if suit == trump:
        if rank == JACK:
            return [0, RIGHT_BOWER]
        return [0, TRUMP_STRENGTH[rank]]

    if suit == left_suit:
        if rank == JACK:
            return [0, LEFT_BOWER]      # trump, not a card of its own suit
        return [0, -rank]               # canonical clubs axis

    plus_x, _ = _plain_axes(trump)
    if suit == plus_x:
        return [rank, 0]
    return [-rank, 0]


def card_from_engine(vector, trump):
    """Inverse of card_to_engine: canonical vector -> natural Card."""
    x, y = int(vector[0]), int(vector[1])
    left_suit = same_colour(trump)
    plus_x, minus_x = _plain_axes(trump)

    if y > 0:
        if y == RIGHT_BOWER:
            return Card(trump, JACK)
        if y == LEFT_BOWER:
            return Card(left_suit, JACK)
        if y not in STRENGTH_RANK:
            raise ValueError("not a trump strength: %d" % y)
        return Card(trump, STRENGTH_RANK[y])
    if y < 0:
        return Card(left_suit, _check_rank(-y))
    if x > 0:
        return Card(plus_x, _check_rank(x))
    if x < 0:
        return Card(minus_x, _check_rank(-x))
    raise ValueError("[0, 0] is not a card")


def _check_rank(rank):
    if rank not in RANKS:
        raise ValueError("not a rank: %d" % rank)
    return rank


def to_engine(cards, trump):
    """A sequence of natural cards -> an (n, 2) int64 array of canonical vectors."""
    return np.array([card_to_engine(c, trump) for c in cards], dtype=np.int64)


def from_engine(vectors, trump):
    """An (n, 2) array of canonical vectors -> a list of natural Cards."""
    return [card_from_engine(v, trump) for v in vectors]


def deal_to_engine(hands, trump):
    """
    Four hands of natural cards -> a (4, 5, 2) int64 deal ready for the solver.

    Raises if the deal is not four hands of five distinct real cards, since a
    duplicate would otherwise be solved without complaint.
    """
    hands = [list(h) for h in hands]
    if len(hands) != 4:
        raise ValueError("a deal needs exactly 4 hands, got %d" % len(hands))
    for i, h in enumerate(hands):
        if len(h) != 5:
            raise ValueError("hand %d holds %d cards, expected 5" % (i, len(h)))

    flat = [tuple(c) for h in hands for c in h]
    if len(set(flat)) != 20:
        raise ValueError("the deal contains duplicate cards")

    return np.array([[card_to_engine(c, trump) for c in h] for h in hands],
                    dtype=np.int64)


# ------------------------------------------------------------ notation

def parse_card(text):
    """'JH' -> Card(HEARTS, JACK). Accepts '10H' as well as 'TH'."""
    s = str(text).strip().upper()
    if len(s) < 2:
        raise ValueError("cannot read a card from %r" % (text,))
    rank_part, suit_part = s[:-1], s[-1]
    if suit_part not in SUIT_LETTERS:
        raise ValueError("no such suit in %r" % (text,))
    if rank_part == "10":
        rank_part = "T"
    if rank_part not in LETTER_RANKS:
        raise ValueError("no such rank in %r" % (text,))
    return Card(SUIT_LETTERS.index(suit_part), LETTER_RANKS[rank_part])


def parse_hand(text):
    """'JH JD AH 9S TC' or ['JH', 'JD', ...] -> a list of Cards."""
    if isinstance(text, str):
        text = text.split()
    return [parse_card(t) for t in text]


def card_name(card):
    """Card(HEARTS, JACK) -> 'JH'."""
    suit, rank = card
    return RANK_LETTERS[rank] + SUIT_LETTERS[suit]


def hand_name(cards):
    return " ".join(card_name(c) for c in cards)


def parse_suit(text):
    """'H' or 'hearts' -> HEARTS."""
    s = str(text).strip().upper()
    if len(s) == 1 and s in SUIT_LETTERS:
        return SUIT_LETTERS.index(s)
    for suit, word in enumerate(("CLUBS", "DIAMONDS", "HEARTS", "SPADES")):
        if s == word:
            return suit
    raise ValueError("cannot read a suit from %r" % (text,))


def suit_name(suit):
    return ("clubs", "diamonds", "hearts", "spades")[suit]


def full_deck():
    """All 24 real Euchre cards as natural Cards."""
    return [Card(s, r) for s in SUITS for r in RANKS]
