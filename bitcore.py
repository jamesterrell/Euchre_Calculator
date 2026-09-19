"""
The solver again, as bitboards over a trump-canonical card space.

`fast_search.py` is the reference: a depth-first alpha-beta over (suit,
strength) planes, one card per recursive call, hands carried as arrays that get
mutated and restored. It is correct, it is the thing every test is written
against, and it is not fast enough to run ten thousand deals of a Perfect
Information Monte Carlo sweep in a minute. This is the same search with three
things changed, each of which is a theorem rather than a heuristic:

  * **a 24-bit board.** A hand is one integer. Following suit is a mask, the
    trick winner is two comparisons, and playing a card is `h ^ bit`. No
    arrays, no swap-and-restore, no per-node bounds to get wrong.
  * **equivalent-card move reduction.** Two cards in the same hand with no
    other live card between them are interchangeable, so only one of them is
    ever searched. Proved in `notes/equivalence.md`, Theorem 1.
  * **a transposition table over rank-compressed positions.** Only the *order*
    of the live cards matters, not their names, so positions that differ only
    by which cards are dead collapse onto one another. Theorem 2, same note.
    Seats are rotated so the leader is seat 0 (Theorem 3) and the three plain
    suits are sorted into a canonical order (Theorem 4).

Every one of those is exact: the value returned is the same minimax value
`fast_search` returns, and `tests/test_bitcore.py` asserts that on randomised
sweeps against both it and `reference_solver`.

## The card space

Cards are natural ids `suit * 6 + rank`, suits `C D H S = 0 1 2 3` and ranks
`9 T J Q K A = 0..5` -- `rotation.Card` without the namedtuple. Inside a solve
they are *canonical* bit positions, laid out so that a single fixed layout
serves every trump suit:

    bits  0..6   trump:        9 T Q K A  left-bower  right-bower
    bits  7..11  same colour:  9 T Q K A            (its jack is trump)
    bits 12..17  plain A:      9 T J Q K A
    bits 18..23  plain B:      9 T J Q K A

Strength is the bit position within the field, so "higher card" is "higher
bit". Which of the two off-colour suits is A and which is B does not matter --
they never interact -- so they are taken in suit order, which makes the whole
map a permutation of 0..23 and `CANON[trump]` its table.

This is `rotation.py`'s job done as a lookup instead of a vector: the left
bower lands inside the trump field, above the ace and below the right bower,
for the same reason it lands on the trump axis there.
"""
import numpy as np
from numba import njit

# ------------------------------------------------------------------ layout

CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3
SAME_COLOUR = (SPADES, HEARTS, DIAMONDS, CLUBS)

R9, RT, RJ, RQ, RK, RA = 0, 1, 2, 3, 4, 5

HAND_SIZE = 5
TRICKS = 5
NEEDED = 3
MARCH = 2
LONE_MARCH = 4

FIELD_OFF = np.array([0, 7, 12, 18], dtype=np.int64)
FIELD_WIDTH = np.array([7, 5, 6, 6], dtype=np.int64)
FIELD_MASK = np.array([(1 << 7) - 1,
                       ((1 << 5) - 1) << 7,
                       ((1 << 6) - 1) << 12,
                       ((1 << 6) - 1) << 18], dtype=np.int64)
TRUMP_FIELD = 0


def _canon_table():
    """`CANON[trump][natural id]` -> canonical bit, and its inverse."""
    canon = np.zeros((4, 24), dtype=np.int64)
    back = np.zeros((4, 24), dtype=np.int64)
    for trump in range(4):
        same = SAME_COLOUR[trump]
        plains = sorted({0, 1, 2, 3} - {trump, same})
        rows = [
            [(trump, R9), (trump, RT), (trump, RQ), (trump, RK), (trump, RA),
             (same, RJ), (trump, RJ)],
            [(same, R9), (same, RT), (same, RQ), (same, RK), (same, RA)],
            [(plains[0], r) for r in range(6)],
            [(plains[1], r) for r in range(6)],
        ]
        for field, row in enumerate(rows):
            for i, (suit, rank) in enumerate(row):
                bit = int(FIELD_OFF[field]) + i
                canon[trump, suit * 6 + rank] = bit
                back[trump, bit] = suit * 6 + rank
    return canon, back


CANON, DECANON = _canon_table()


def _field_of():
    out = np.zeros(24, dtype=np.int64)
    for field in range(4):
        for i in range(int(FIELD_WIDTH[field])):
            out[int(FIELD_OFF[field]) + i] = field
    return out


FIELD_OF = _field_of()


def _mask_tables():
    """12-bit halves of a natural-id mask -> the canonical mask, per trump."""
    lo = np.zeros((4, 1 << 12), dtype=np.int64)
    hi = np.zeros((4, 1 << 12), dtype=np.int64)
    for trump in range(4):
        for value in range(1 << 12):
            a = 0
            b = 0
            for bit in range(12):
                if value >> bit & 1:
                    a |= 1 << int(CANON[trump, bit])
                    b |= 1 << int(CANON[trump, bit + 12])
            lo[trump, value] = a
            hi[trump, value] = b
    return lo, hi


CONV_LO, CONV_HI = _mask_tables()


def _pack_tables():
    """
    `PEXT[live][value]` and `PDEP[live][value]`, over 7-bit fields.

    `PEXT` squeezes the bits of `value` down onto the positions `live` marks,
    in order; `PDEP` puts them back. They are the whole of the rank
    compression: a suit's live cards become 0, 1, 2, ... and everything the
    search asks about a card -- which is only ever "is it higher than that
    one" -- survives, because the squeeze is monotone.
    """
    pext = np.zeros((1 << 7, 1 << 7), dtype=np.int64)
    pdep = np.zeros((1 << 7, 1 << 7), dtype=np.int64)
    for live in range(1 << 7):
        slots = [b for b in range(7) if live >> b & 1]
        for value in range(1 << 7):
            packed = 0
            for i, b in enumerate(slots):
                if value >> b & 1:
                    packed |= 1 << i
            pext[live, value] = packed
        for value in range(1 << 7):
            spread = 0
            for i, b in enumerate(slots):
                if value >> i & 1:
                    spread |= 1 << b
            pdep[live, value] = spread
    return pext, pdep


PEXT, PDEP = _pack_tables()

def _win_masks():
    """
    `WIN_MASK[winner]` -- every card that would take the trick off `winner`.

    Trump beats anything that is not trump, and within a suit the higher card
    wins, so this is two ranges and no comparisons. It is what lets the move
    loop try the cards that fight for the trick before the ones that give it
    up, which is the ordering alpha-beta wants and costs nothing to compute.
    """
    out = np.zeros(24, dtype=np.int64)
    for card in range(24):
        field = int(FIELD_OF[card])
        top = int(FIELD_OFF[field]) + int(FIELD_WIDTH[field])
        above = 0
        for b in range(card + 1, top):
            above |= 1 << b
        if field == TRUMP_FIELD:
            out[card] = above
        else:
            out[card] = above | int(FIELD_MASK[TRUMP_FIELD])
    return out


WIN_MASK = _win_masks()


POPCOUNT = np.array([bin(v).count("1") for v in range(1 << 12)], dtype=np.int64)

# Transposition table entry flags, in the usual three kinds.
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TT_BITS = 22

# One entry is one int64, and every field of it -- the whole position, the
# value, the bound flag and the depth -- lives inside that word:
#
#   bits  0..31  the owner of each live card, two bits apiece
#   bits 32..40  how many live cards the first three suits hold
#   bits 41..43  the trick on the table, 1..4 (so a real entry is never 0)
#   bit  44      the caller is playing alone
#   bits 45..47  tricks the calling team has already taken
#   bit  48      which side of the rotated table the caller is on
#   bits 49..50  bound flag
#   bits 51..52  the value, as an index into TT_VALUES
#   bits 53..55  how big the subtree under it was
#
# That it fits in one word is what lets a single table be shared by every
# thread with no lock and no risk: an aligned 64-bit load or store is atomic,
# so a reader sees an entry exactly as some writer left it, never half of one
# and half of another. Two threads racing for a slot is fine -- whoever lands
# last wins, and both entries were true.
TT_KEY_BITS = 49
TT_KEY_MASK = (1 << TT_KEY_BITS) - 1

# Slots to a bucket. Eight int64s are one 64-byte cache line, so a probe that
# looks at all eight costs the same one miss a probe that looks at one does,
# and the table stops throwing away a deep entry every time a shallow one
# happens to hash next to it.
TT_WAY_BITS = 3
TT_WAYS = 1 << TT_WAY_BITS
TT_VALUES = np.array([-2, 1, 2, 4], dtype=np.int64)
TT_CODES = np.zeros(9, dtype=np.int64)
TT_CODES[-2 + 4] = 0
TT_CODES[1 + 4] = 1
TT_CODES[2 + 4] = 2
TT_CODES[4 + 4] = 3


def _spread_table():
    """`SPREAD[b]` -- byte `b` with its bits moved to every other position."""
    out = np.zeros(256, dtype=np.int64)
    for b in range(256):
        v = 0
        for i in range(8):
            if b >> i & 1:
                v |= 1 << (2 * i)
        out[b] = v
    return out


SPREAD = _spread_table()


def new_tt(bits: int = TT_BITS):
    """A fresh transposition table of `2 ** bits` slots, zero meaning empty."""
    return np.zeros(1 << bits, dtype=np.int64)


def tt_mask(tt):
    """The bucket mask for a table: `_tt_index` returns a bucket, not a slot."""
    return np.int64((len(tt) >> TT_WAY_BITS) - 1)


# The search's explicit call stack: one row per ply, and a hand is twenty of
# them. Handed in rather than allocated per solve, since a sweep runs a couple
# of thousand solves a deal and each would otherwise pay for the array.
STACK_PLIES = TRICKS * 4 + 2
STACK_FIELDS = 18


def new_stack():
    return np.zeros((STACK_PLIES, STACK_FIELDS), dtype=np.int64)


@njit(inline="always", cache=True)
def to_canon(nat_mask, trump):
    """A 24-bit mask of natural card ids -> the same cards as canonical bits."""
    return (CONV_LO[trump, nat_mask & 4095]
            | CONV_HI[trump, (nat_mask >> 12) & 4095])


# --------------------------------------------------------------- the rules


@njit(inline="always", cache=True)
def _beats(card, winner):
    """
    Does `card` take the trick from `winner`?

    Trump beats anything that is not trump; otherwise the higher bit of the
    same field wins, and a card of neither the led suit nor trump cannot win at
    all. Fields never overlap, so no cross-suit comparison is possible -- the
    same reason `fast_search._resolve` can scale trump strengths by ten.
    """
    fc = FIELD_OF[card]
    fw = FIELD_OF[winner]
    if fc == TRUMP_FIELD:
        return fw != TRUMP_FIELD or card > winner
    return fc == fw and card > winner


@njit(inline="always", cache=True)
def _moves(hand, live, n_in_trick, led):
    """
    The cards worth trying: legal, with equivalent duplicates struck out.

    Legality is the whole of the rule -- follow the led suit if you hold it,
    otherwise anything. The reduction on top of it is Theorem 1: within one
    hand, two cards of the same suit with no live card between them lead to
    positions of equal value, so only the lowest of each such run is returned.
    Cards already played to the trick and beaten are not live: they can never
    be compared against again, which is what makes the runs longer than they
    look.
    """
    if n_in_trick > 0:
        followed = hand & FIELD_MASK[led]
        if followed != 0:
            hand = followed
    reps = 0
    m = hand & 127
    if m != 0:
        lf = live & 127
        p = PEXT[lf, m]
        reps |= PDEP[lf, p & ~(p << 1)]
    m = (hand >> 7) & 31
    if m != 0:
        lf = (live >> 7) & 31
        p = PEXT[lf, m]
        reps |= PDEP[lf, p & ~(p << 1)] << 7
    m = (hand >> 12) & 63
    if m != 0:
        lf = (live >> 12) & 63
        p = PEXT[lf, m]
        reps |= PDEP[lf, p & ~(p << 1)] << 12
    m = (hand >> 18) & 63
    if m != 0:
        lf = (live >> 18) & 63
        p = PEXT[lf, m]
        reps |= PDEP[lf, p & ~(p << 1)] << 18
    return reps


@njit(inline="always", cache=True)
def _lowest_bit(x):
    """Index of the lowest set bit. `x` must be non-zero."""
    n = 0
    if (x & 4095) == 0:
        n += 12
        x >>= 12
    if (x & 63) == 0:
        n += 6
        x >>= 6
    if (x & 7) == 0:
        n += 3
        x >>= 3
    while (x & 1) == 0:
        n += 1
        x >>= 1
    return n


@njit(inline="always", cache=True)
def _final(caller_tricks, alone):
    if caller_tricks == TRICKS:
        return LONE_MARCH if alone else MARCH
    if caller_tricks >= NEEDED:
        return 1
    return -2


@njit(inline="always", cache=True)
def _next(seat, sitting):
    s = (seat + 1) & 3
    if s == sitting:
        s = (s + 1) & 3
    return s


# ------------------------------------------------ the canonical position key


@njit(inline="always", cache=True)
def _tt_key(h0, h1, h2, h3, leader, caller_tricks, caller_team, alone,
            trick_no):
    """
    The position at a trick boundary, reduced to the two words that name it.

    Three exact reductions, all proved in `notes/equivalence.md`:

      * seats are rotated so the leader is seat 0 (Theorem 3), which folds four
        positions into one and takes the leader out of the key;
      * each suit's live cards are compressed to 0, 1, 2, ... (Theorem 2), so
        only their order survives and every position that differs by which
        cards are already dead collapses onto one representative;
      * the three plain suits are sorted into a canonical order (Theorem 4),
        since they never interact and so are interchangeable.

    The fourth hand is not stored: it is whatever the other three do not hold,
    and the per-suit live counts -- which are in the key -- say how wide that
    is. Nothing else is needed, because a trick boundary has no trick.
    """
    a0 = h0
    a1 = h1
    a2 = h2
    a3 = h3
    if leader == 1:
        a0 = h1
        a1 = h2
        a2 = h3
        a3 = h0
    elif leader == 2:
        a0 = h2
        a1 = h3
        a2 = h0
        a3 = h1
    elif leader == 3:
        a0 = h3
        a1 = h0
        a2 = h1
        a3 = h2
    team = (caller_team - leader) & 1

    live = a0 | a1 | a2 | a3

    # Per-field compressed contents. Held in scalars rather than a small
    # array: this runs at every trick boundary of every node of every solve,
    # and four heap allocations there cost more than the table saves.
    l = live & 127
    n0 = POPCOUNT[l]
    x0 = PEXT[l, a0 & 127]
    y0 = PEXT[l, a1 & 127]
    z0 = PEXT[l, a2 & 127]

    l = (live >> 7) & 31
    n1 = POPCOUNT[l]
    x1 = PEXT[l, (a0 >> 7) & 31]
    y1 = PEXT[l, (a1 >> 7) & 31]
    z1 = PEXT[l, (a2 >> 7) & 31]

    l = (live >> 12) & 63
    n2 = POPCOUNT[l]
    x2 = PEXT[l, (a0 >> 12) & 63]
    y2 = PEXT[l, (a1 >> 12) & 63]
    z2 = PEXT[l, (a2 >> 12) & 63]

    l = (live >> 18) & 63
    n3 = POPCOUNT[l]
    x3 = PEXT[l, (a0 >> 18) & 63]
    y3 = PEXT[l, (a1 >> 18) & 63]
    z3 = PEXT[l, (a2 >> 18) & 63]

    # Sort the three plain suits into a canonical order -- Theorem 4. Three
    # compare-and-swaps, on a single packed key so the comparison is one test.
    k1 = (n1 << 21) | (x1 << 14) | (y1 << 7) | z1
    k2 = (n2 << 21) | (x2 << 14) | (y2 << 7) | z2
    k3 = (n3 << 21) | (x3 << 14) | (y3 << 7) | z3
    if k2 < k1:
        k1, k2 = k2, k1
        n1, n2 = n2, n1
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        z1, z2 = z2, z1
    if k3 < k1:
        k1, k3 = k3, k1
        n1, n3 = n3, n1
        x1, x3 = x3, x1
        y1, y3 = y3, y1
        z1, z3 = z3, z1
    if k3 < k2:
        k2, k3 = k3, k2
        n2, n3 = n3, n2
        x2, x3 = x3, x2
        y2, y3 = y3, y2
        z2, z3 = z3, z2

    g0 = x0
    g1 = y0
    g2 = z0
    shift = n0
    g0 |= x1 << shift
    g1 |= y1 << shift
    g2 |= z1 << shift
    shift += n1
    g0 |= x2 << shift
    g1 |= y2 << shift
    g2 |= z2 << shift
    shift += n2
    g0 |= x3 << shift
    g1 |= y3 << shift
    g2 |= z3 << shift
    shift += n3

    # Who holds each live card, two bits apiece. Three disjoint masks over at
    # most sixteen cards would need forty-eight bits; the same information as
    # one digit per card needs thirty-two, and thirty-two is what leaves room
    # for the value in the same word. The fourth hand is whatever is left.
    g3 = ((np.int64(1) << shift) - 1) & ~(g0 | g1 | g2)
    lo = g1 | g3
    hi = g2 | g3
    owners = (SPREAD[lo & 255] | (SPREAD[(lo >> 8) & 255] << 16))         | ((SPREAD[hi & 255] | (SPREAD[(hi >> 8) & 255] << 16)) << 1)

    return (owners | (n0 << 32) | (n1 << 35) | (n2 << 38)
            | (trick_no << 41) | (alone << 44) | (caller_tricks << 45)
            | (team << 48))


@njit(inline="always", cache=True)
def _tt_index(key, mask):
    h = key * np.int64(0x9E3779B1)
    h ^= (h >> 29) & np.int64(0x7FFFFFFF)
    h *= np.int64(0x2545F491)
    h ^= (h >> 32) & np.int64(0xFFFFFFFF)
    return h & mask


@njit(inline="always", cache=True)
def _highest_bit(x):
    """Index of the highest set bit. `x` must be non-zero."""
    n = 0
    if x >> 12 != 0:
        n += 12
        x >>= 12
    if x >> 6 != 0:
        n += 6
        x >>= 6
    if x >> 3 != 0:
        n += 3
        x >>= 3
    while x > 1:
        n += 1
        x >>= 1
    return n


# ------------------------------------------------------------- the search


@njit(cache=True)
def _search(h, to_act, n_in_trick, led, win_card, win_seat,
            caller_team, caller_tricks, trick_no, alpha, beta,
            sitting, width, alone, tt, ttmask, nodes, stk):
    """
    Alpha-beta from any position. Returns the calling team's score.

    `h` holds the four hands as canonical bitmasks and is restored exactly by
    the time this returns -- a card is played with one xor and put back with
    the same one, so there is no swap-and-restore to get out of step.

    Everything specific to a loner is in `sitting` and `width`: the sitting
    partner's mask is empty, `_next` steps over the seat, and a trick completes
    one card sooner.

    **The recursion is an explicit stack** -- `stk`, one row per ply and never
    more than twenty of them. That is not a micro-optimisation: numba 0.60
    segfaults, reliably, loading a *recursive* njit function from its on-disk
    cache, so a recursive search means paying a minute of compilation in every
    process that imports it. Written flat it caches, and a sweep starts in a
    second rather than in a minute. The shape below is the ordinary one for
    turning a call stack into a loop: each ply is a little state machine with
    four states -- arrive, try the next move, fold a child's answer back in,
    hand the answer up.
    """
    # Frame layout. One row of `stk` per ply; the ply being worked on lives in
    # locals and is written out only when a child is pushed underneath it.
    F_ACT = 0
    F_NIT = 1
    F_LED = 2
    F_WCARD = 3
    F_WSEAT = 4
    F_CT = 5
    F_TNO = 6
    F_ALPHA = 7
    F_BETA = 8
    F_A0 = 9
    F_B0 = 10
    F_BEST = 11
    F_REST = 12
    F_LATER = 13
    F_CARD = 14
    F_KEY = 15
    F_STORE = 16
    F_MAX = 17

    ENTER = 0
    STEP = 1
    FOLD = 2
    UNWIND = 3

    depth = 0
    phase = ENTER
    ret = 0

    a0 = alpha
    b0 = beta
    best = 0
    rest = 0
    later = 0
    card = 0
    key = 0
    store = False
    maximizing = False

    while True:
        if phase == ENTER:
            nodes[0] += 1
            a0 = alpha
            b0 = beta
            store = False
            key = 0
            settled = False

            if n_in_trick == 0:
                # The result is already decided; playing it out cannot change
                # it.
                left = TRICKS - trick_no
                if caller_tricks + left < NEEDED:
                    ret = -2
                    settled = True
                elif caller_tricks >= NEEDED and (trick_no - caller_tricks) >= 1:
                    ret = 1
                    settled = True
                elif 0 < trick_no < TRICKS - 1:
                    # The last trick is not looked up: what is left under it is
                    # one trick of at most four cards, which the search
                    # finishes in fewer cycles than a miss into a table this
                    # size costs.
                    key = _tt_key(h[0], h[1], h[2], h[3], to_act,
                                  caller_tricks, caller_team, alone, trick_no)
                    base = _tt_index(key, ttmask) << TT_WAY_BITS
                    for j in range(TT_WAYS):
                        entry = tt[base + j]
                        if (entry & TT_KEY_MASK) != key:
                            continue
                        v = TT_VALUES[(entry >> 51) & 3]
                        f = (entry >> 49) & 3
                        if f == TT_EXACT:
                            ret = v
                            settled = True
                        elif f == TT_LOWER:
                            if v >= beta:
                                ret = v
                                settled = True
                            elif v > alpha:
                                alpha = v
                        else:
                            if v <= alpha:
                                ret = v
                                settled = True
                            elif v < beta:
                                beta = v
                        break
                    if not settled:
                        store = True

            if settled:
                phase = UNWIND
                continue

            live = h[0] | h[1] | h[2] | h[3]
            if n_in_trick > 0:
                live |= np.int64(1) << win_card
            reps = _moves(h[to_act], live, n_in_trick, led)

            # Cards that fight for the trick first, cheapest of them first,
            # then the ones that give it up, cheapest first. Alpha-beta wants
            # the best move first from *both* sides -- the minimiser is just as
            # keen to take the trick away as the maximiser is to keep it.
            if n_in_trick > 0:
                rest = reps & WIN_MASK[win_card]
                later = reps & ~WIN_MASK[win_card]
                if rest == 0:
                    rest = later
                    later = 0
            else:
                rest = reps
                later = 0

            maximizing = (to_act & 1) == caller_team
            best = -1000 if maximizing else 1000
            card = -1
            phase = STEP

        if phase == STEP:
            if rest == 0:
                ret = best
                if store:
                    # Eight slots to a bucket and a bucket to a cache line, so
                    # looking at all of them costs what looking at one costs.
                    # The one that goes is the one with the least under it:
                    # a trick-one entry stands for a subtree a hundred times
                    # the size of a trick-three entry, and a direct-mapped
                    # table lets the cheap one evict it.
                    base = _tt_index(key, ttmask) << TT_WAY_BITS
                    depth_left = TRICKS - trick_no
                    victim = base
                    worst = 99
                    for j in range(TT_WAYS):
                        entry = tt[base + j]
                        if entry == 0 or (entry & TT_KEY_MASK) == key:
                            victim = base + j
                            worst = -1
                            break
                        d = (entry >> 53) & 7
                        if d < worst:
                            worst = d
                            victim = base + j
                    flag = TT_EXACT
                    if best <= a0:
                        flag = TT_UPPER
                    elif best >= b0:
                        flag = TT_LOWER
                    tt[victim] = (key | (flag << 49)
                                  | (TT_CODES[best + 4] << 51)
                                  | (depth_left << 53))
                phase = UNWIND
                continue

            bit = _lowest_bit(rest)
            rest ^= np.int64(1) << bit
            if rest == 0:
                rest = later
                later = 0
            card = bit
            h[to_act] ^= np.int64(1) << bit

            if n_in_trick == 0:
                n_led = FIELD_OF[bit]
                n_win = bit
                n_seat = to_act
            else:
                n_led = led
                if _beats(bit, win_card):
                    n_win = bit
                    n_seat = to_act
                else:
                    n_win = win_card
                    n_seat = win_seat

            if n_in_trick == width - 1:
                taken = caller_tricks
                if (n_seat & 1) == caller_team:
                    taken += 1
                if trick_no + 1 == TRICKS:
                    ret = _final(taken, alone == 1)
                    phase = FOLD
                    continue
                c_act = n_seat
                c_nit = 0
                c_led = 0
                c_wcard = 0
                c_wseat = 0
                c_ct = taken
                c_tno = trick_no + 1
            else:
                c_act = _next(to_act, sitting)
                c_nit = n_in_trick + 1
                c_led = n_led
                c_wcard = n_win
                c_wseat = n_seat
                c_ct = caller_tricks
                c_tno = trick_no

            stk[depth, F_ACT] = to_act
            stk[depth, F_NIT] = n_in_trick
            stk[depth, F_LED] = led
            stk[depth, F_WCARD] = win_card
            stk[depth, F_WSEAT] = win_seat
            stk[depth, F_CT] = caller_tricks
            stk[depth, F_TNO] = trick_no
            stk[depth, F_ALPHA] = alpha
            stk[depth, F_BETA] = beta
            stk[depth, F_A0] = a0
            stk[depth, F_B0] = b0
            stk[depth, F_BEST] = best
            stk[depth, F_REST] = rest
            stk[depth, F_LATER] = later
            stk[depth, F_CARD] = card
            stk[depth, F_KEY] = key
            stk[depth, F_STORE] = 1 if store else 0
            stk[depth, F_MAX] = 1 if maximizing else 0
            depth += 1

            to_act = c_act
            n_in_trick = c_nit
            led = c_led
            win_card = c_wcard
            win_seat = c_wseat
            caller_tricks = c_ct
            trick_no = c_tno
            phase = ENTER
            continue

        if phase == FOLD:
            h[to_act] ^= np.int64(1) << card
            if maximizing:
                if ret > best:
                    best = ret
                if best > alpha:
                    alpha = best
            else:
                if ret < best:
                    best = ret
                if best < beta:
                    beta = best
            if beta <= alpha:
                rest = 0
                later = 0
            phase = STEP
            continue

        # UNWIND: hand `ret` to the ply above, or out of the search.
        depth -= 1
        if depth < 0:
            return ret
        to_act = stk[depth, F_ACT]
        n_in_trick = stk[depth, F_NIT]
        led = stk[depth, F_LED]
        win_card = stk[depth, F_WCARD]
        win_seat = stk[depth, F_WSEAT]
        caller_tricks = stk[depth, F_CT]
        trick_no = stk[depth, F_TNO]
        alpha = stk[depth, F_ALPHA]
        beta = stk[depth, F_BETA]
        a0 = stk[depth, F_A0]
        b0 = stk[depth, F_B0]
        best = stk[depth, F_BEST]
        rest = stk[depth, F_REST]
        later = stk[depth, F_LATER]
        card = stk[depth, F_CARD]
        key = stk[depth, F_KEY]
        store = stk[depth, F_STORE] == 1
        maximizing = stk[depth, F_MAX] == 1
        phase = FOLD


@njit(cache=True)
def _moves_at(h, to_act, n_in_trick, led, win_card, win_seat,
              caller_team, caller_tricks, trick_no, sitting, width, alone,
              tt, ttmask, nodes, stk, out_bits, out_vals):
    """
    The value of every legal card for the seat to act, one solve each.

    The equivalence reduction runs here too, but the values it saves are filled
    back in rather than dropped: a caller choosing a card wants a number for
    each of them, and Theorem 1 says the duplicates take the value of their
    representative. Returns how many cards were written.
    """
    live = h[0] | h[1] | h[2] | h[3]
    if n_in_trick > 0:
        live |= np.int64(1) << win_card

    legal = h[to_act]
    if n_in_trick > 0:
        followed = legal & FIELD_MASK[led]
        if followed != 0:
            legal = followed

    reps = _moves(h[to_act], live, n_in_trick, led)
    m = 0
    rest = reps
    while rest != 0:
        bit = _highest_bit(rest)
        card = np.int64(1) << bit
        rest ^= card
        h[to_act] ^= card

        if n_in_trick == 0:
            n_led = FIELD_OF[bit]
            n_win = bit
            n_seat = to_act
        else:
            n_led = led
            if _beats(bit, win_card):
                n_win = bit
                n_seat = to_act
            else:
                n_win = win_card
                n_seat = win_seat

        if n_in_trick == width - 1:
            taken = caller_tricks
            if (n_seat & 1) == caller_team:
                taken += 1
            if trick_no + 1 == TRICKS:
                val = _final(taken, alone == 1)
            else:
                val = _search(h, n_seat, 0, 0, 0, 0, caller_team, taken,
                              trick_no + 1, -1000, 1000, sitting, width,
                              alone, tt, ttmask, nodes, stk)
        else:
            val = _search(h, _next(to_act, sitting), n_in_trick + 1, n_led,
                          n_win, n_seat, caller_team, caller_tricks, trick_no,
                          -1000, 1000, sitting, width, alone,
                          tt, ttmask, nodes, stk)

        h[to_act] ^= card
        out_bits[m] = bit
        out_vals[m] = val
        m += 1

    # Fill the cards the reduction struck out with their representative's
    # value: same suit, no live card between, so Theorem 1 gives them exactly
    # this number and re-solving would only confirm it.
    extra = legal & ~reps
    while extra != 0:
        bit = _highest_bit(extra)
        extra ^= np.int64(1) << bit
        field = FIELD_OF[bit]
        off = FIELD_OFF[field]
        lf = (live >> off) & 127
        rank = PEXT[lf, np.int64(1) << (bit - off)]
        best_i = -1
        best_r = -1
        for i in range(m):
            if FIELD_OF[out_bits[i]] != field:
                continue
            other = PEXT[lf, np.int64(1) << (out_bits[i] - off)]
            # its representative is the nearest one below it
            if other < rank and other > best_r:
                best_r = other
                best_i = i
        out_bits[m] = bit
        out_vals[m] = out_vals[best_i]
        m += 1
    return m


# ---------------------------------------------------------- entry points


@njit(cache=True)
def solve_canon(h0, h1, h2, h3, leader, caller, alone,
                tt, ttmask, nodes, stk, alpha=-1000, beta=1000):
    """
    God Mode score for a fresh deal, from the calling team's side.

    Hands are canonical bitmasks -- `to_canon(natural_mask, trump)`. A loner's
    partner is emptied here rather than in the recursion, and the lead moves on
    if it would have fallen on that seat, which is the same rule
    `fast_search._setup` applies.
    """
    h = np.empty(4, dtype=np.int64)
    h[0] = h0
    h[1] = h1
    h[2] = h2
    h[3] = h3
    sitting = -1
    width = 4
    if alone == 1:
        sitting = (caller + 2) & 3
        h[sitting] = 0
        width = 3
        if leader == sitting:
            leader = (leader + 1) & 3
    return _search(h, leader, 0, 0, 0, 0, caller & 1, 0, 0, alpha, beta,
                   sitting, width, alone, tt, ttmask, nodes, stk)


@njit(cache=True)
def position_canon(h, to_act, n_in_trick, led, win_card, win_seat,
                   caller, caller_tricks, trick_no, alone,
                   tt, ttmask, nodes, stk, out_bits, out_vals):
    """
    Value of each legal card from a part-played position. Returns how many.

    The trick in progress is given as the seat that is winning it and the card
    that is doing so, not as the cards played: the beaten ones are dead, they
    can never be compared against again, and leaving them out is what makes the
    equivalence classes as wide as they are. `led` is the field the trick was
    led in, which is all that follow-suit needs.
    """
    sitting = -1
    width = 4
    if alone == 1:
        sitting = (caller + 2) & 3
        width = 3
    return _moves_at(h, to_act, n_in_trick, led, win_card, win_seat,
                     caller & 1, caller_tricks, trick_no, sitting, width,
                     alone, tt, ttmask, nodes, stk,
                     out_bits, out_vals)
