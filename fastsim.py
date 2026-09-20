"""
The whole sweep, compiled: deal, auction, sampling, search and all.

`hand_ev.py` asks one question -- what is this hand worth to a table that
cannot see it -- and answers it by playing the deal out tens of thousands of
times. The machinery that does the playing is spread over `table.py`,
`players.py`, `observation.py` and `bidding.py`, which is the right shape for
reading and the wrong shape for a sweep: every sampled world is built out of
Python tuples and dataclasses, and a profile of the old path put roughly half
the time in that plumbing rather than in the search it exists to feed.

This module is those four files again with nothing in them but integers, so the
entire per-deal simulation compiles. It is not a different model -- the
decisions, the sampling, the pass models, the stopping rule and the tie-breaks
are the ones `players.py` documents, and `tests/test_fastsim.py` holds both
implementations to the same answers.

## What is represented as what

A card is its natural id `suit * 6 + (rank - 9)`, suits `C D H S = 0 1 2 3`.
A set of cards -- a hand, the kitty, the cards a seat has been shown not to
hold -- is a 24-bit mask. Nothing is a tuple, nothing is a dataclass, and the
only arrays are the ones the recursion writes into.

`bitcore.py` holds the search itself and its own, trump-canonical, bit layout;
this module converts into that layout at the call and back at the answer.

## Randomness

`random.Random` is not available inside njit, so this carries its own
splitmix64 and every stream is an `int64[1]`. That is a deliberate second
implementation rather than a mimicry of Python's Mersenne Twister: the two
engines draw different worlds, so they do not agree deal by deal, and
`tests/test_fastsim.py` compares them the way the sweep itself is read --
distributions, with an error bar.
"""
import hashlib
import os

import numpy as np
from numba import njit, prange

import bitcore


def _drop_stale_cache():
    """
    Throw this module's compiled cache away when `bitcore.py` has moved.

    numba caches each function against **its own** source and nothing else.
    Everything here has `bitcore`'s search compiled into it, so editing the
    solver alone leaves this module's cache in place and quietly keeps running
    the old one. That is not hypothetical: it hid a 2x improvement to the
    search for an afternoon, and the only symptom was a change that measured
    as doing nothing.

    So this module stamps what it was built against and drops its own cache
    when the stamp moves. Deleting too much costs seventy seconds of
    recompilation; deleting too little costs a wrong measurement, which is
    worse.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, "__pycache__")
    if not os.path.isdir(cache):
        return
    try:
        with open(os.path.join(here, "bitcore.py"), "rb") as handle:
            stamp = hashlib.sha1(handle.read()).hexdigest()
    except OSError:
        return
    mark = os.path.join(cache, "fastsim.dependency")
    try:
        with open(mark) as handle:
            if handle.read().strip() == stamp:
                return
    except OSError:
        pass
    for name in os.listdir(cache):
        if name.startswith("fastsim.") and name.endswith((".nbi", ".nbc")):
            try:
                os.remove(os.path.join(cache, name))
            except OSError:
                pass
    try:
        with open(mark, "w") as handle:
            handle.write(stamp)
    except OSError:
        pass


_drop_stale_cache()
from bitcore import (CANON, DECANON, FIELD_MASK, FIELD_OF, PEXT,
                     TRICKS, NEEDED, new_tt)

DECK = 24
PLAYERS = 4
HAND = 5
KITTY = 4
FULL_DECK = (1 << DECK) - 1

# Natural suit -> the suit of the same colour, whose jack becomes the left
# bower. Mirrors rotation.same_colour.
SAME_COLOUR = np.array([3, 2, 1, 0], dtype=np.int64)
RANK_JACK = 2                       # rank index of the jack

# Up-card states, as observation.py names them.
UP = 0
PICKED_UP = 1
TURNED_DOWN = 2

# Bid actions, as table.py names them.
PASS = 0
ORDER = 1
NAME = 2

# Pass models, as players.py names them.
PASS_GOD = 0
PASS_ZERO = 1
PASS_FLOOR = 2
PASS_GUARD = 3

# What the asking seat does with its opening bid, as hand_ev.py names them.
ASSUME_AUCTION = 0
ASSUME_ORDER = 1
ASSUME_ALONE = 2
ASSUME_PASS = 3

MIN_WORLDS = 24
Z = 2.576
GUARD = 2.0
GROWTH = 1.5

MAX_DRAW_ATTEMPTS = 200


def _suit_masks():
    """`SUIT_MASK[trump][suit]` -- membership in *effective* suits."""
    out = np.zeros((4, 4), dtype=np.int64)
    for trump in range(4):
        left = int(SAME_COLOUR[trump])
        for card in range(DECK):
            suit = card // 6
            rank = card % 6
            eff = trump if (rank == RANK_JACK and suit == left) else suit
            out[trump, eff] |= 1 << card
    return out


SUIT_MASK = _suit_masks()


def _eff_suits():
    """`EFF_SUIT[trump][card]`, with trump -1 folded in as row 4."""
    out = np.zeros((5, DECK), dtype=np.int64)
    for trump in range(4):
        left = int(SAME_COLOUR[trump])
        for card in range(DECK):
            suit = card // 6
            rank = card % 6
            out[trump, card] = trump if (rank == RANK_JACK and suit == left) \
                else suit
    for card in range(DECK):
        out[4, card] = card // 6          # no trump named yet
    return out


EFF_SUIT = _eff_suits()

POP12 = np.array([bin(v).count("1") for v in range(1 << 12)], dtype=np.int64)


# ------------------------------------------------------------------- bits


@njit(inline="always", cache=True)
def popcount(x):
    return POP12[x & 4095] + POP12[(x >> 12) & 4095]


@njit(inline="always", cache=True)
def lowest(x):
    """Index of the lowest set bit of a non-zero mask."""
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


# ------------------------------------------------------------------- rng


@njit(inline="always", cache=True)
def _shr(x, k):
    """Logical right shift. `>>` on int64 is arithmetic, which splitmix is not."""
    return (x >> k) & ((np.int64(1) << (64 - k)) - np.int64(1))


@njit(inline="always", cache=True)
def next_u64(state):
    """splitmix64: one 64-bit draw, advancing `state` in place."""
    state[0] = state[0] + np.int64(-7046029254386353131)
    z = state[0]
    z = (z ^ _shr(z, 30)) * np.int64(-4658895280553007687)
    z = (z ^ _shr(z, 27)) * np.int64(-7723592293110705685)
    return z ^ _shr(z, 31)


@njit(inline="always", cache=True)
def rand_below(state, n):
    """Uniform on 0..n-1, by rejection so the modulo cannot skew it."""
    if n <= 1:
        return 0
    limit = (np.int64(1) << 62) - ((np.int64(1) << 62) % n)
    while True:
        v = _shr(next_u64(state), 2)
        if v < limit:
            return v % n


@njit(inline="always", cache=True)
def rand_unit(state):
    """Uniform on [0, 1), with 53 bits of it."""
    return _shr(next_u64(state), 11) * (1.0 / 9007199254740992.0)


@njit(inline="always", cache=True)
def seed_stream(state, seed):
    state[0] = np.int64(seed) * np.int64(6364136223846793005) \
        + np.int64(1442695040888963407)
    for _ in range(4):
        next_u64(state)


# ------------------------------------------------------------- dealing out


@njit(cache=True)
def deal_around(known_hand, seat, up_card, dealer, rng, hands):
    """
    `game.deal_around`: pin what the asker can see, deal the rest uniformly.

    Writes the four hands into `hands` and returns (up-card, kitty mask). The
    pinned cards stay where they were put; everything else is a shuffle of what
    is left, which is what makes each deal of the sweep an independent draw
    from the layouts consistent with the question.
    """
    pool = FULL_DECK & ~known_hand
    if up_card >= 0:
        pool &= ~(np.int64(1) << up_card)

    n = popcount(pool)
    cards = np.empty(n, dtype=np.int64)
    i = 0
    rest = pool
    while rest != 0:
        b = lowest(rest)
        rest &= rest - 1
        cards[i] = b
        i += 1
    for j in range(n - 1, 0, -1):
        k = rand_below(rng, j + 1)
        tmp = cards[j]
        cards[j] = cards[k]
        cards[k] = tmp

    take = n
    for s in range(PLAYERS):
        if s == seat:
            h = known_hand
            need = HAND - popcount(known_hand)
        else:
            h = 0
            need = HAND
        for _ in range(need):
            take -= 1
            h |= np.int64(1) << cards[take]
        hands[s] = h

    if up_card >= 0:
        turned = up_card
    else:
        take -= 1
        turned = cards[take]

    buried = 0
    for j in range(take):
        buried |= np.int64(1) << cards[j]
    return turned, buried


# ------------------------------------------------------- what a seat knows


@njit(cache=True)
def voids_of(plays_seat, plays_card, n_plays, width, trump, out):
    """
    The effective suits each seat has shown it cannot hold.

    A seat that fails to follow the led suit is void in it for the rest of the
    hand. Read in effective suits, so ruffing a club lead with the left bower
    shows no club void and following one cannot have been done with it --
    `observation.Observation.voids` for why that distinction is load-bearing.
    """
    for s in range(PLAYERS):
        out[s] = 0
    i = 0
    while i < n_plays:
        led = EFF_SUIT[trump, plays_card[i]]
        j = i + 1
        while j < n_plays and j < i + width:
            if EFF_SUIT[trump, plays_card[j]] != led:
                out[plays_seat[j]] |= np.int64(1) << led
            j += 1
        i += width


@njit(cache=True)
def obs_setup(observer, hands, dealer, up_card, up_state, trump, sitting,
              discard, pending, plays_seat, plays_card, n_plays, width,
              caps, voids):
    """
    The pool a sampled world is drawn from, and the room each slot has for it.

    Returns (pool, forced_up, known_kitty). `forced_up` is the up-card if it
    must be put back in the dealer's hand and -1 otherwise; `known_kitty` is
    the buried cards this seat can already name -- the up-card once it is
    turned down, and the dealer's own discard -- which are kept out of the pool
    so they cannot be dealt to anybody, and so have to be put back on the pile
    afterwards. `caps` and `voids` are filled in place: one entry per seat plus
    the kitty, whose capacity is *derived* from card conservation rather than
    counted, which is what lets the dealer's six-card moment be an ordinary
    observation.
    """
    voids_of(plays_seat, plays_card, n_plays, width, trump, voids)

    played = 0
    for i in range(n_plays):
        played |= np.int64(1) << plays_card[i]

    counts = np.empty(PLAYERS, dtype=np.int64)
    for s in range(PLAYERS):
        counts[s] = HAND
    for i in range(n_plays):
        counts[plays_seat[i]] -= 1
    if pending:
        counts[observer] += 1

    known_kitty = 0
    if up_state != PICKED_UP:
        known_kitty |= np.int64(1) << up_card
    if observer == dealer and discard >= 0:
        known_kitty |= np.int64(1) << discard

    seen = hands[observer] | played | known_kitty
    pool = FULL_DECK & ~seen

    total = 0
    for s in range(PLAYERS):
        caps[s] = 0 if s == observer else counts[s]
        total += caps[s]
    caps[KITTY] = popcount(pool) - total

    forced = -1
    if (up_state == PICKED_UP and observer != dealer
            and (played >> up_card) & 1 == 0 and counts[dealer] != 0):
        # Ordered up and not seen since, so it is still in the dealer's hand --
        # unless the dealer has shown out of trump, in which case it can only
        # have been what they buried.
        if trump == 4 or ((voids[dealer] >> EFF_SUIT[trump, up_card]) & 1) == 0:
            forced = up_card
            pool &= ~(np.int64(1) << up_card)
            caps[dealer] -= 1
            held = 0
            for s in range(PLAYERS):
                held += caps[s]
            caps[KITTY] = popcount(pool) - held
    return pool, forced, known_kitty


@njit(cache=True)
def sample_world(pool, caps, voids, sitting, trump, rng, out):
    """
    One layout consistent with an observation, or failure to find one.

    Cards go out most-constrained first -- the ones fewest slots will take go
    while there is still room to take them -- and among the slots that will
    take a card one is drawn with probability proportional to the room it has
    left, which is what an ordinary shuffle-and-deal does. With no voids in
    play that is uniform over layouts; with voids it is close but not exact,
    the same approximation `observation.sample_worlds` documents.

    Returns False if it paints itself into a corner, which the caller retries.
    """
    n = popcount(pool)
    cards = np.empty(24, dtype=np.int64)
    tight = np.empty(24, dtype=np.int64)
    key = np.empty(24, dtype=np.float64)

    i = 0
    rest = pool
    while rest != 0:
        card = lowest(rest)
        rest &= rest - 1
        cards[i] = card
        eff = EFF_SUIT[trump, card]
        fits = 0
        for s in range(PLAYERS + 1):
            if caps[s] == 0:
                continue
            if s == KITTY or s == sitting or ((voids[s] >> eff) & 1) == 0:
                fits += 1
        tight[i] = fits
        key[i] = rand_unit(rng)
        i += 1

    # insertion sort on (slots that will take it, then a random tiebreak)
    for a in range(1, n):
        c = cards[a]
        t = tight[a]
        k = key[a]
        b = a - 1
        while b >= 0 and (tight[b] > t or (tight[b] == t and key[b] > k)):
            cards[b + 1] = cards[b]
            tight[b + 1] = tight[b]
            key[b + 1] = key[b]
            b -= 1
        cards[b + 1] = c
        tight[b + 1] = t
        key[b + 1] = k

    room = np.empty(PLAYERS + 1, dtype=np.int64)
    for s in range(PLAYERS + 1):
        room[s] = caps[s]
        out[s] = 0

    for a in range(n):
        card = cards[a]
        eff = EFF_SUIT[trump, card]
        total = 0
        for s in range(PLAYERS + 1):
            if room[s] == 0:
                continue
            if s == KITTY or s == sitting or ((voids[s] >> eff) & 1) == 0:
                total += room[s]
        if total == 0:
            return False
        pick = rand_unit(rng) * total
        run = 0.0
        chosen = -1
        for s in range(PLAYERS + 1):
            if room[s] == 0:
                continue
            if s == KITTY or s == sitting or ((voids[s] >> eff) & 1) == 0:
                run += room[s]
                if chosen < 0 and pick < run:
                    chosen = s
        if chosen < 0:
            chosen = PLAYERS          # rounding at the very top of the range
            while chosen >= 0 and room[chosen] == 0:
                chosen -= 1
            if chosen < 0:
                return False
        out[chosen] |= np.int64(1) << card
        room[chosen] -= 1
    return True


@njit(cache=True)
def draw_world(pool, caps, voids, sitting, trump, own, observer, dealer,
               forced, known_kitty, rng, out):
    """
    `sample_world`, retried, with what is already known written back over it.

    Three things are not guesses and so are not dealt: the observer's own hand,
    the up-card when it was ordered up and nothing since says it went under,
    and the buried cards the seat can already name. All three are put back
    after the deal-out, which never saw them -- so that every one of the
    twenty-four cards is in exactly one of the five slots, which is the
    invariant the rest of the module is entitled to assume.
    """
    ok = False
    for _ in range(MAX_DRAW_ATTEMPTS):
        ok = sample_world(pool, caps, voids, sitting, trump, rng, out)
        if ok:
            break
    if not ok:
        return False
    if forced >= 0:
        out[dealer] |= np.int64(1) << forced
    out[KITTY] |= known_kitty
    out[observer] = own
    return True


# ------------------------------------------------------------- card order


def _strength_table():
    """`STRENGTH[trump][card]` -- comparable only within an effective suit."""
    out = np.zeros((4, DECK), dtype=np.int64)
    for trump in range(4):
        left = int(SAME_COLOUR[trump])
        for card in range(DECK):
            suit = card // 6
            rank = card % 6
            if suit == trump and rank == RANK_JACK:
                out[trump, card] = 16
            elif suit == left and rank == RANK_JACK:
                out[trump, card] = 15
            else:
                out[trump, card] = rank
    return out


STRENGTH = _strength_table()


def _tie_table():
    """
    `TIE_RANK[trump][card]` -- the order `players._pick` throws cards in.

    LOW breaks ties with `table.card_order`, which is the solver's own
    (suit code, strength). The codes are the engine's, not the natural suits':
    the +x plain axis is 0, trump is 1, the -x plain axis is 2 and the
    same-colour suit is 3. Reproduced rather than reinvented, because two runs
    that disagree here disagree on the card played.
    """
    out = np.zeros((4, DECK), dtype=np.int64)
    for trump in range(4):
        left = int(SAME_COLOUR[trump])
        plains = sorted({0, 1, 2, 3} - {trump, left})
        code = {plains[0]: 0, trump: 1, plains[1]: 2, left: 3}
        for card in range(DECK):
            suit = card // 6
            rank = card % 6
            if rank == RANK_JACK and suit == left:
                where = code[trump]
                strength = 15
            elif suit == trump:
                where = code[trump]
                strength = 16 if rank == RANK_JACK else rank
            else:
                where = code[suit]
                strength = rank
            out[trump, card] = where * 32 + strength
    return out


TIE_RANK = _tie_table()


@njit(inline="always", cache=True)
def beats(card, winner, trump):
    """Does `card` take the trick from `winner`? Effective suits throughout."""
    ec = EFF_SUIT[trump, card]
    ew = EFF_SUIT[trump, winner]
    if ec == trump:
        return ew != trump or STRENGTH[trump, card] > STRENGTH[trump, winner]
    if ec != ew:
        return False
    return STRENGTH[trump, card] > STRENGTH[trump, winner]


@njit(inline="always", cache=True)
def legal_mask(hand, led_card, trump):
    """Follow the led suit if you hold it; otherwise anything. The whole rule."""
    if led_card < 0:
        return hand
    follow = hand & SUIT_MASK[trump, EFF_SUIT[trump, led_card]]
    return follow if follow != 0 else hand


@njit(inline="always", cache=True)
def next_seat(seat, sitting):
    s = (seat + 1) & 3
    if s == sitting:
        s = (s + 1) & 3
    return s


@njit(inline="always", cache=True)
def final_score(caller_tricks, alone):
    if caller_tricks == TRICKS:
        return 4 if alone == 1 else 2
    if caller_tricks >= NEEDED:
        return 1
    return -2


@njit(inline="always", cache=True)
def net_to_team0(caller_score, caller):
    return caller_score if (caller & 1) == 0 else -caller_score


@njit(inline="always", cache=True)
def value_to(seat, team0):
    return team0 if (seat & 1) == 0 else -team0


@njit(inline="always", cache=True)
def to_seat(caller_value, caller, seat):
    return caller_value if (seat & 1) == (caller & 1) else -caller_value


@njit(inline="always", cache=True)
def prefers(candidate, incumbent, seat):
    """Does `seat` prefer `candidate` to `incumbent`, on the team-0 scale?"""
    if (seat & 1) == 0:
        return candidate > incumbent
    return candidate < incumbent


# -------------------------------------------------------- God Mode values


@njit(cache=True)
def play_value(hands, trump, dealer, caller, alone, tt,
               ttm, nodes, stk):
    """
    God Mode trick-play value of a settled deal, from the caller's side.

    Play begins to the dealer's left whoever called, and the solver moves the
    lead on if that seat is the one sitting out a loner.

    Asked as two yes-or-no questions rather than as one open one. A Euchre hand
    is worth -2, 1 or 2 -- or -2, 1 or 4 alone -- so "is it worth at least 1?"
    and then "at least 2?" pin it exactly, and each of those runs on a window
    one point wide, where alpha-beta cuts off almost everywhere. The second
    question also inherits the first one's bounds out of the transposition
    table. Same number as one wide search, and measurably fewer nodes.
    """
    h0 = bitcore.to_canon(hands[0], trump)
    h1 = bitcore.to_canon(hands[1], trump)
    h2 = bitcore.to_canon(hands[2], trump)
    h3 = bitcore.to_canon(hands[3], trump)
    leader = (dealer + 1) & 3
    if bitcore.solve_canon(h0, h1, h2, h3, leader, caller, alone,
                           tt, ttm, nodes, stk, 0, 1) <= 0:
        return -2
    if bitcore.solve_canon(h0, h1, h2, h3, leader, caller, alone,
                           tt, ttm, nodes, stk, 1, 2) <= 1:
        return 1
    return 4 if alone == 1 else 2



@njit(cache=True)
def discard_reps(candidates, live, trump):
    """
    Which of the dealer's cards are worth solving, and which are copies.

    Burying one of two cards with no live card between them leaves the same
    game as burying the other: only one of them is ever played, and every
    comparison the played one takes part in comes out the same way either way.
    That is Theorem 1 of `notes/equivalence.md` again, with "the card that
    stayed in hand" in place of "the card that was not led".

    Returns the representatives as a mask of natural card ids -- the lowest of
    each run -- so a caller can solve those and copy the rest.
    """
    reps = bitcore._moves(bitcore.to_canon(candidates, trump),
                          bitcore.to_canon(live, trump), 0, 0)
    out = 0
    while reps != 0:
        bit = reps & -reps
        reps ^= bit
        out |= np.int64(1) << DECANON[trump, lowest(bit)]
    return out


@njit(cache=True)
def rep_of(cards, k, reps, trump):
    """
    For each candidate, the index of the representative that stands for it.

    A card's representative is the nearest one at or below it in the same
    effective suit, which is how `discard_reps` picked them.
    """
    out = np.zeros(MAX_OPTIONS, dtype=np.int64)
    for i in range(k):
        bit = CANON[trump, cards[i]]
        field = FIELD_OF[bit]
        best = -1
        best_at = i
        for j in range(k):
            if (reps >> cards[j]) & 1 == 0:
                continue
            other = CANON[trump, cards[j]]
            if FIELD_OF[other] != field or other > bit:
                continue
            if other > best:
                best = other
                best_at = j
        out[i] = best_at
    return out

@njit(cache=True)
def order_up_value(hands, up_card, dealer, caller, alone,
                   tt, ttm, nodes, stk):
    """
    What ordering the turned suit up is worth, in net points to team 0.

    The **dealer** chooses the discard, for the dealer's own team -- ordered up
    by the opposition it is taking a card into a contract it wants to fail. The
    up-card is not a candidate: ordered up, it is in the dealer's hand to stay.

    One case collapses. If the caller goes alone and the dealer is the partner
    sitting out, the dealer's whole hand leaves play, so all five discards are
    worth exactly the same and one solve answers for them.
    """
    trump = up_card // 6
    sitting = (caller + 2) & 3 if alone == 1 else -1
    work = np.empty(PLAYERS, dtype=np.int64)
    for s in range(PLAYERS):
        work[s] = hands[s]
    held = hands[dealer]
    up_bit = np.int64(1) << up_card

    if sitting == dealer:
        work[dealer] = (held & ~(np.int64(1) << lowest(held))) | up_bit
        return net_to_team0(play_value(work, trump, dealer, caller, alone,
                                       tt, ttm, nodes, stk), caller)

    # The dealer keeps only the best of the five, so the four it rejects need
    # not be valued -- only shown to be no better. Each solve after the first
    # runs with the window open on one side only, which is exact: a fail-soft
    # search that comes back above its alpha has come back with the true value,
    # and one that does not has come back with a bound, which is all a
    # rejection needs.
    same_team = (dealer & 1) == (caller & 1)
    best = 0
    have = False
    rest = discard_reps(held, hands[0] | hands[1] | hands[2] | hands[3] | up_bit,
                        trump)
    while rest != 0:
        card = lowest(rest)
        rest &= rest - 1
        work[dealer] = (held & ~(np.int64(1) << card)) | up_bit
        alpha = -1000
        beta = 1000
        if have:
            # `best` is on the team-0 scale; the window is on the caller's.
            bound = best if (caller & 1) == 0 else -best
            if same_team:
                alpha = bound
            else:
                beta = bound
        raw = bitcore.solve_canon(
            bitcore.to_canon(work[0], trump), bitcore.to_canon(work[1], trump),
            bitcore.to_canon(work[2], trump), bitcore.to_canon(work[3], trump),
            (dealer + 1) & 3, caller, alone, tt, ttm, nodes, stk, alpha, beta)
        value = net_to_team0(raw, caller)
        if not have or prefers(value, best, dealer):
            best = value
            have = True
    return best


@njit(inline="always", cache=True)
def name_suit_value(hands, trump, dealer, caller, alone,
                    tt, ttm, nodes, stk):
    """What naming `trump` in round two is worth, in net points to team 0."""
    return net_to_team0(play_value(hands, trump, dealer, caller, alone,
                                   tt, ttm, nodes, stk), caller)


@njit(cache=True)
def rest_of_auction(hands, up_card, dealer, index, bidding_round, stick,
                    loners, tt, ttm, nodes, stk):
    """
    What the auction from `index` onward is worth, in net points to team 0.

    This is the price of a pass: a seat that declines does not get zero, it
    gets whatever the remaining seats do, which can be worse than the call it
    turned down. `bidding._round_one` and `_round_two` say the same thing as a
    recursion; this is the same chain solved backwards, which is the same tree
    and the same solves without the call stack.

    Options are listed pass, call, call-alone and ties keep the first, so a
    seat with nothing to gain declines and a loner worth no more than the same
    call four-handed is not taken.
    """
    up_suit = up_card // 6
    alones = 2 if loners == 1 else 1

    # ---- round two, backwards from the dealer
    r2 = np.zeros(PLAYERS + 1, dtype=np.int64)
    stop2 = index if bidding_round == 2 else 0
    for idx in range(PLAYERS - 1, stop2 - 1, -1):
        seat = (dealer + 1 + idx) & 3
        have = False
        best = 0
        if not (stick == 1 and idx == PLAYERS - 1):
            best = r2[idx + 1]
            have = True
        for suit in range(4):
            if suit == up_suit:
                continue
            for alone in range(alones):
                value = name_suit_value(hands, suit, dealer, seat, alone,
                                        tt, ttm, nodes, stk)
                if not have or prefers(value, best, seat):
                    best = value
                    have = True
        r2[idx] = best
    if bidding_round == 2:
        return r2[index]

    # ---- round one, backwards, with "everybody passes" landing in round two
    r1 = np.zeros(PLAYERS + 1, dtype=np.int64)
    r1[PLAYERS] = r2[0]
    for idx in range(PLAYERS - 1, index - 1, -1):
        seat = (dealer + 1 + idx) & 3
        best = r1[idx + 1]
        for alone in range(alones):
            value = order_up_value(hands, up_card, dealer, seat, alone,
                                   tt, ttm, nodes, stk)
            if prefers(value, best, seat):
                best = value
        r1[idx] = best
    return r1[index]


# --------------------------------------------- sequential elimination

MAX_OPTIONS = 8


@njit(cache=True)
def _prune(k, alive, n_alive, s, n, pn, ps, pq, epsilon, min_worlds):
    """
    Drop the options that cannot matter. Returns how many are left.

    Two grounds, and they are `players._race`'s: an option is **losing** when
    the gap to the leader is bigger than the noise on the gap, and **close
    enough** when the gap is smaller than `epsilon`, so picking either moves
    this decision by less than the band.

    The test is on **paired** differences rather than on the two means
    separately -- every option is scored in the same sampled world, so
    `v_i - v_j` is far quieter than either mean and is usually exactly zero.
    The `guard / m` term is what lets two options that have agreed in every
    world so far be called tied rather than waiting for an interval that
    assumes they might not.
    """
    leader = alive[0]
    best = s[leader] / n[leader]
    for a in range(1, n_alive):
        i = alive[a]
        mean = s[i] / n[i]
        if mean > best:
            best = mean
            leader = i

    kept = 0
    for a in range(n_alive):
        j = alive[a]
        if j == leader:
            alive[kept] = j
            kept += 1
            continue
        q = leader * k + j if leader < j else j * k + leader
        m = pn[q]
        if m < min_worlds:
            alive[kept] = j
            kept += 1
            continue
        total = ps[q] if leader < j else -ps[q]
        mean_d = total / m
        if m > 1:
            var = (pq[q] - ps[q] * ps[q] / m) / (m - 1)
        else:
            var = 0.0
        radius = GUARD / m
        if var > 0:
            radius += Z * np.sqrt(var / m)
        if mean_d - radius < -epsilon:
            alive[kept] = j
            kept += 1
    return kept


@njit(inline="always", cache=True)
def _pair_update(k, alive, n_alive, vals, pn, ps, pq):
    for a in range(n_alive):
        i = alive[a]
        vi = vals[i]
        for c in range(a + 1, n_alive):
            j = alive[c]
            d = vi - vals[j]
            if i < j:
                q = i * k + j
            else:
                q = j * k + i
                d = -d
            pn[q] += 1
            ps[q] += d
            pq[q] += d * d


@njit(inline="always", cache=True)
def _argmax_first(k, alive, n_alive, s, n):
    """The leading option, ties keeping the earliest -- `players._pick`."""
    leader = alive[0]
    best = s[leader] / n[leader]
    for a in range(1, n_alive):
        i = alive[a]
        mean = s[i] / n[i]
        if mean > best:
            best = mean
            leader = i
    return leader, best


@njit(inline="always", cache=True)
def _argmax_low(k, alive, n_alive, s, n, cards, trump):
    """The leader, with ties thrown low -- `players._pick` under LOW."""
    leader, best = _argmax_first(k, alive, n_alive, s, n)
    chosen = leader
    rank = TIE_RANK[trump, cards[leader]]
    for a in range(n_alive):
        i = alive[a]
        if s[i] / n[i] != best:
            continue
        r = TIE_RANK[trump, cards[i]]
        if r < rank:
            rank = r
            chosen = i
    return chosen


# ----------------------------------------------------------- the decisions


@njit(cache=True)
def pimc_bid(observer, hands, dealer, up_card, up_state,
             bidding_round, index, stick, loners, pass_model,
             budget, epsilon, racing, min_worlds,
             rng, tt, ttm, nodes, stk,
             opt_act, opt_suit, opt_alone):
    """
    One seat's bid: sample layouts, price every option in each, take the best.

    The options are built here in the order `table._bid_options` builds them --
    passing first, then the call, then the call alone -- because `_pick` keeps
    the first of equals, so a seat with nothing to gain declines rather than
    ordering up a contract it expects to lose.

    Returns the index of the chosen option. The caller reads the action out of
    `opt_act` / `opt_suit` / `opt_alone`.
    """
    up_suit = up_card // 6
    is_last = index == PLAYERS - 1
    k = 0
    if not (stick == 1 and is_last and bidding_round == 2):
        opt_act[k] = PASS
        opt_suit[k] = -1
        opt_alone[k] = 0
        k += 1
    alones = 2 if loners == 1 else 1
    if bidding_round == 1:
        for a in range(alones):
            opt_act[k] = ORDER
            opt_suit[k] = up_suit
            opt_alone[k] = a
            k += 1
    else:
        for suit in range(4):
            if suit == up_suit:
                continue
            for a in range(alones):
                opt_act[k] = NAME
                opt_suit[k] = suit
                opt_alone[k] = a
                k += 1

    caps = np.zeros(PLAYERS + 1, dtype=np.int64)
    voids = np.zeros(PLAYERS, dtype=np.int64)
    plays_seat = np.zeros(1, dtype=np.int64)
    plays_card = np.zeros(1, dtype=np.int64)
    pool, forced, buried = obs_setup(observer, hands, dealer, up_card, up_state,
                                     4, -1, -1, False, plays_seat, plays_card,
                                     0, 4, caps, voids)

    world = np.zeros(PLAYERS + 1, dtype=np.int64)
    vals = np.zeros(MAX_OPTIONS, dtype=np.float64)
    s = np.zeros(MAX_OPTIONS, dtype=np.float64)
    n = np.zeros(MAX_OPTIONS, dtype=np.int64)
    alive = np.arange(MAX_OPTIONS).astype(np.int64)
    pn = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.int64)
    ps = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    pq = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    n_alive = k

    drawn = 0
    check_at = min_worlds
    while drawn < budget:
        if racing and n_alive < 2:
            break
        if not draw_world(pool, caps, voids, -1, 4, hands[observer], observer,
                          dealer, forced, buried, rng, world):
            break
        drawn += 1
        nodes[1] += 1
        for a in range(n_alive):
            i = alive[a]
            nodes[2] += 1
            if opt_act[i] == PASS:
                if pass_model == PASS_ZERO:
                    v = 0.0
                else:
                    team0 = rest_of_auction(world, up_card, dealer, index + 1,
                                            bidding_round, stick, loners,
                                            tt, ttm, nodes, stk)
                    v = float(value_to(observer, team0))
                    if pass_model == PASS_FLOOR and v < 0.0:
                        v = 0.0
            elif opt_act[i] == ORDER:
                v = float(value_to(observer,
                                   order_up_value(world, up_card, dealer,
                                                  observer, opt_alone[i],
                                                  tt, ttm, nodes, stk)))
            else:
                v = float(value_to(observer,
                                   name_suit_value(world, opt_suit[i], dealer,
                                                   observer, opt_alone[i],
                                                   tt, ttm, nodes,
                                                   stk)))
            vals[i] = v
            n[i] += 1
            s[i] += v
        if not racing:
            continue
        _pair_update(k, alive, n_alive, vals, pn, ps, pq)
        if drawn < check_at:
            continue
        check_at = max(drawn + 1, int(drawn * GROWTH))
        n_alive = _prune(k, alive, n_alive, s, n, pn, ps, pq, epsilon,
                         min_worlds)

    if drawn == 0:
        return 0

    if pass_model == PASS_GUARD:
        # Never take a call that is negative on its own merits -- applied to
        # the averaged values, which is the whole difference from PASS_FLOOR.
        has_pass = False
        has_call = False
        all_bad = True
        for i in range(k):
            if opt_act[i] == PASS:
                has_pass = True
            else:
                has_call = True
                # An option that was never scored reads as 0.0, which is not
                # negative -- `players._means` says the same, and the guard is
                # a rule about calls that are known to lose.
                mean = s[i] / n[i] if n[i] > 0 else 0.0
                if mean >= 0.0:
                    all_bad = False
        if has_pass and has_call and all_bad:
            for i in range(k):
                if opt_act[i] == PASS:
                    return i

    leader, _ = _argmax_first(k, alive, n_alive, s, n)
    return leader


@njit(cache=True)
def pimc_discard(observer, hands, up_card, trump, caller, alone,
                 budget, epsilon, racing, min_worlds,
                 rng, tt, ttm, nodes, stk):
    """
    Which of the six the dealer buries, judged the same way as everything else.

    The dealer answers for its **own** team, so ordered up by the opposition it
    picks the card that hurts the contract most. The up-card is not a candidate
    -- ordered up, it stays in the hand.
    """
    sitting = (caller + 2) & 3 if alone == 1 else -1
    candidates = hands[observer] & ~(np.int64(1) << up_card)
    k = popcount(candidates)
    cards = np.zeros(MAX_OPTIONS, dtype=np.int64)
    i = 0
    rest = candidates
    while rest != 0:
        cards[i] = lowest(rest)
        rest &= rest - 1
        i += 1
    if k == 1:
        return cards[0]

    caps = np.zeros(PLAYERS + 1, dtype=np.int64)
    voids = np.zeros(PLAYERS, dtype=np.int64)
    plays_seat = np.zeros(1, dtype=np.int64)
    plays_card = np.zeros(1, dtype=np.int64)
    pool, forced, buried = obs_setup(observer, hands, observer, up_card,
                                     PICKED_UP, trump, sitting, -1, True,
                                     plays_seat, plays_card, 0, 4, caps, voids)

    world = np.zeros(PLAYERS + 1, dtype=np.int64)
    work = np.zeros(PLAYERS, dtype=np.int64)
    done = np.zeros(MAX_OPTIONS, dtype=np.int64)
    held = np.zeros(MAX_OPTIONS, dtype=np.float64)
    vals = np.zeros(MAX_OPTIONS, dtype=np.float64)
    s = np.zeros(MAX_OPTIONS, dtype=np.float64)
    n = np.zeros(MAX_OPTIONS, dtype=np.int64)
    alive = np.arange(MAX_OPTIONS).astype(np.int64)
    pn = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.int64)
    ps = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    pq = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    n_alive = k

    drawn = 0
    check_at = min_worlds
    while drawn < budget:
        if racing and n_alive < 2:
            break
        if not draw_world(pool, caps, voids, sitting, trump, hands[observer],
                          observer, observer, forced, buried, rng, world):
            break
        drawn += 1
        nodes[3] += 1
        # Which cards this world makes interchangeable -- the three it buried
        # can close a gap between two of the six, and then burying either of
        # them leaves the same game.
        reps = discard_reps(candidates, FULL_DECK & ~world[KITTY], trump)
        owner = rep_of(cards, k, reps, trump)
        for i in range(k):
            done[i] = 0
        for a in range(n_alive):
            i = alive[a]
            r = owner[i]
            if done[r] == 0:
                nodes[4] += 1
                for s2 in range(PLAYERS):
                    work[s2] = world[s2]
                work[observer] = world[observer] & ~(np.int64(1) << cards[r])
                held[r] = float(to_seat(
                    play_value(work, trump, observer, caller, alone,
                               tt, ttm, nodes, stk), caller, observer))
                done[r] = 1
            v = held[r]
            vals[i] = v
            n[i] += 1
            s[i] += v
        if not racing:
            continue
        _pair_update(k, alive, n_alive, vals, pn, ps, pq)
        if drawn < check_at:
            continue
        check_at = max(drawn + 1, int(drawn * GROWTH))
        n_alive = _prune(k, alive, n_alive, s, n, pn, ps, pq, epsilon,
                         min_worlds)

    if drawn == 0:
        return cards[0]
    return cards[_argmax_low(k, alive, n_alive, s, n, cards, trump)]


@njit(cache=True)
def pimc_play(observer, hands, dealer, up_card, up_state, trump, caller,
              alone, discard, plays_seat, plays_card, n_plays, width,
              n_in_trick, led_card, win_card, win_seat, caller_tricks,
              trick_no, budget, epsilon, racing, min_worlds,
              rng, tt, ttm, nodes, stk):
    """
    One card, chosen by solving the hands this seat might be in.

    A seat with one legal card skips the search entirely rather than spending
    a few hundred solves confirming it has no choice. Otherwise one
    `position_canon` call prices every legal card in a sampled world at once,
    which is why dropping an option here saves nothing inside a world -- the
    saving from the stopping rule is in ending the world loop early.
    """
    sitting = (caller + 2) & 3 if alone == 1 else -1
    legal = legal_mask(hands[observer], led_card, trump)
    k = popcount(legal)
    cards = np.zeros(MAX_OPTIONS, dtype=np.int64)
    i = 0
    rest = legal
    while rest != 0:
        cards[i] = lowest(rest)
        rest &= rest - 1
        i += 1
    if k == 1:
        return cards[0]

    slot = np.full(DECK, -1, dtype=np.int64)
    for j in range(k):
        slot[cards[j]] = j

    caps = np.zeros(PLAYERS + 1, dtype=np.int64)
    voids = np.zeros(PLAYERS, dtype=np.int64)
    pool, forced, buried = obs_setup(observer, hands, dealer, up_card, up_state,
                                     trump, sitting, discard, False,
                                     plays_seat, plays_card, n_plays, width,
                                     caps, voids)

    world = np.zeros(PLAYERS + 1, dtype=np.int64)
    ch = np.zeros(PLAYERS, dtype=np.int64)
    out_bits = np.zeros(MAX_OPTIONS, dtype=np.int64)
    out_vals = np.zeros(MAX_OPTIONS, dtype=np.int64)
    vals = np.zeros(MAX_OPTIONS, dtype=np.float64)
    s = np.zeros(MAX_OPTIONS, dtype=np.float64)
    n = np.zeros(MAX_OPTIONS, dtype=np.int64)
    alive = np.arange(MAX_OPTIONS).astype(np.int64)
    pn = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.int64)
    ps = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    pq = np.zeros(MAX_OPTIONS * MAX_OPTIONS, dtype=np.float64)
    n_alive = k

    if n_in_trick > 0:
        led_field = FIELD_OF[CANON[trump, led_card]]
        win_bit = CANON[trump, win_card]
    else:
        led_field = 0
        win_bit = 0
        win_seat = 0

    drawn = 0
    check_at = min_worlds
    while drawn < budget:
        if racing and n_alive < 2:
            break
        if not draw_world(pool, caps, voids, sitting, trump, hands[observer],
                          observer, dealer, forced, buried, rng, world):
            break
        drawn += 1
        nodes[5] += 1
        for s2 in range(PLAYERS):
            ch[s2] = 0 if s2 == sitting else bitcore.to_canon(world[s2], trump)
        m = bitcore.position_canon(ch, observer, n_in_trick, led_field,
                                   win_bit, win_seat, caller, caller_tricks,
                                   trick_no, alone, tt, ttm, nodes, stk,
                                   out_bits, out_vals)
        for j in range(m):
            card = DECANON[trump, out_bits[j]]
            vals[slot[card]] = float(to_seat(out_vals[j], caller, observer))
        for a in range(n_alive):
            i = alive[a]
            n[i] += 1
            s[i] += vals[i]
        if not racing:
            continue
        _pair_update(k, alive, n_alive, vals, pn, ps, pq)
        if drawn < check_at:
            continue
        check_at = max(drawn + 1, int(drawn * GROWTH))
        n_alive = _prune(k, alive, n_alive, s, n, pn, ps, pq, epsilon,
                         min_worlds)

    if drawn == 0:
        return cards[0]
    return cards[_argmax_low(k, alive, n_alive, s, n, cards, trump)]


# ------------------------------------------------------------- the referee

# One row of `out` per deal, in this order.
R_VALUE = 0          # points to the asking seat's team
R_CALLER = 1         # -1 if the deal was passed out
R_TRUMP = 2
R_ALONE = 3
R_TRICKS = 4         # tricks to the calling team
R_SCORE = 5          # from the calling team's side
R_FORCED = 6         # did the --assume pin actually fire
RECORD = 7

# Per-thread tallies the sweep hands back: nodes searched, then the
# worlds and option-solves each kind of decision spent.
C_NODES = 0
C_BID_WORLDS = 1
C_BID_SOLVES = 2
C_DISCARD_WORLDS = 3
C_DISCARD_SOLVES = 4
C_PLAY_WORLDS = 5
COUNTERS = 8


@njit(cache=True)
def play_one_deal(pin_hand, pin_seat, pin_up, dealer,
                  budget_play, budget_bid, budget_discard,
                  epsilon, racing, min_worlds,
                  pass_model, assume, let_auction_play, stick, loners,
                  rngs, tt, ttm, nodes, stk, out, row):
    """
    One sampled layout, played out by four PIMC sim players.

    `table.play_deal` and `table.play_pinned_order` in one function. The referee
    holds no opinions: it offers the legal options, takes whatever the decision
    functions say, and writes down what happened. Every decision comes out of
    `pimc_bid`, `pimc_discard` or `pimc_play`.
    """
    hands = np.zeros(PLAYERS, dtype=np.int64)
    # The kitty is not tracked from here on. Nobody sees it, nobody plays it,
    # and every observation derives how deep it is from card conservation
    # rather than from being told -- `obs_setup`.
    up_card, _ = deal_around(pin_hand, pin_seat, pin_up, dealer,
                             rngs[PLAYERS], hands)

    opt_act = np.zeros(MAX_OPTIONS, dtype=np.int64)
    opt_suit = np.zeros(MAX_OPTIONS, dtype=np.int64)
    opt_alone = np.zeros(MAX_OPTIONS, dtype=np.int64)
    plays_seat = np.zeros(TRICKS * PLAYERS, dtype=np.int64)
    plays_card = np.zeros(TRICKS * PLAYERS, dtype=np.int64)

    caller = -1
    trump = -1
    alone = 0
    ordered = False
    forced = 0

    if assume != ASSUME_AUCTION and let_auction_play == 0:
        # No auction at all: price the call from this seat every deal.
        caller = pin_seat
        alone = 1 if assume == ASSUME_ALONE else 0
        trump = up_card // 6
        ordered = True
        forced = 1
    else:
        spoken = False
        done = False
        for bidding_round in range(1, 3):
            up_state = UP if bidding_round == 1 else TURNED_DOWN
            for index in range(PLAYERS):
                seat = (dealer + 1 + index) & 3
                act = PASS
                suit = -1
                al = 0
                pinned = False
                if (assume != ASSUME_AUCTION and let_auction_play == 1
                        and seat == pin_seat and not spoken):
                    # Only the opening bid is pinned, and only if it is on
                    # offer -- `players.ForcedOpeningBid`.
                    spoken = True
                    if assume == ASSUME_PASS:
                        if not (stick == 1 and index == PLAYERS - 1
                                and bidding_round == 2):
                            pinned = True
                            act = PASS
                    elif bidding_round == 1:
                        pinned = True
                        act = ORDER
                        suit = up_card // 6
                        al = 1 if assume == ASSUME_ALONE else 0
                    if pinned:
                        forced = 1
                if not pinned:
                    j = pimc_bid(seat, hands, dealer, up_card, up_state,
                                 bidding_round, index, stick, loners,
                                 pass_model, budget_bid, epsilon, racing,
                                 min_worlds, rngs[seat], tt, ttm, nodes,
                                 stk, opt_act, opt_suit, opt_alone)
                    act = opt_act[j]
                    suit = opt_suit[j]
                    al = opt_alone[j]
                if act == PASS:
                    continue
                caller = seat
                alone = al
                if act == ORDER:
                    trump = up_card // 6
                    ordered = True
                else:
                    trump = suit
                done = True
                break
            if done:
                break
        if not done:
            out[row, R_VALUE] = 0
            out[row, R_CALLER] = -1
            out[row, R_TRUMP] = -1
            out[row, R_ALONE] = 0
            out[row, R_TRICKS] = 0
            out[row, R_SCORE] = 0
            out[row, R_FORCED] = forced
            return

    sitting = (caller + 2) & 3 if alone == 1 else -1
    discard = -1
    if ordered:
        # The dealer picks up and pitches a card of its own choosing, which is
        # a card it chooses for *its* team -- ordered up by the opposition it
        # is taking a card into a contract it wants to fail.
        held = hands[dealer] | (np.int64(1) << up_card)
        if sitting == dealer:
            # That hand is about to leave the game, so every discard is worth
            # the same and the choice is unobservable; one is pitched by
            # convention. `bidding.order_up` short-circuits the same case.
            discard = lowest(hands[dealer])
        else:
            hands[dealer] = held
            discard = pimc_discard(dealer, hands, up_card, trump, caller,
                                   alone, budget_discard, epsilon, racing,
                                   min_worlds, rngs[dealer], tt, ttm,
                                   nodes, stk)
        hands[dealer] = held & ~(np.int64(1) << discard)

    up_state = PICKED_UP if ordered else TURNED_DOWN
    width = 3 if alone == 1 else 4
    if sitting >= 0:
        hands[sitting] = 0

    leader = (dealer + 1) & 3
    if leader == sitting:
        leader = next_seat(leader, sitting)

    n_plays = 0
    caller_tricks = 0
    for trick_no in range(TRICKS):
        seat = leader
        n_in = 0
        led_card = -1
        win_card = -1
        win_seat = -1
        for _ in range(width):
            card = pimc_play(seat, hands, dealer, up_card, up_state, trump,
                             caller, alone, discard, plays_seat, plays_card,
                             n_plays, width, n_in, led_card, win_card,
                             win_seat, caller_tricks, trick_no, budget_play,
                             epsilon, racing, min_worlds, rngs[seat],
                             tt, ttm, nodes, stk)
            hands[seat] &= ~(np.int64(1) << card)
            plays_seat[n_plays] = seat
            plays_card[n_plays] = card
            n_plays += 1
            if n_in == 0:
                led_card = card
                win_card = card
                win_seat = seat
            elif beats(card, win_card, trump):
                win_card = card
                win_seat = seat
            n_in += 1
            seat = next_seat(seat, sitting)
        if (win_seat & 1) == (caller & 1):
            caller_tricks += 1
        leader = win_seat

    caller_score = final_score(caller_tricks, alone)
    out[row, R_VALUE] = value_to(pin_seat, net_to_team0(caller_score, caller))
    out[row, R_CALLER] = caller
    out[row, R_TRUMP] = trump
    out[row, R_ALONE] = alone
    out[row, R_TRICKS] = caller_tricks
    out[row, R_SCORE] = caller_score
    out[row, R_FORCED] = forced


@njit(cache=True)
def solve_bidding(hands, up_card, dealer, stick, loners,
                  tt, ttm, nodes, stk, out, row):
    """
    The whole auction in God Mode, every seat seeing every hand.

    `bidding.solve_bidding`, solved backwards along the chain rather than
    forwards down a recursion -- the same tree and the same solves. Writes
    (net points to team 0, caller, trump, alone) into `out[row]`, with a caller
    of -1 for a deal nobody would take.

    Options are listed pass, call, call-alone and ties keep the first, so a
    seat that gains nothing declines rather than calling a contract it knows
    will be euchred, and a loner worth no more than the same call four-handed
    is not taken.
    """
    up_suit = up_card // 6
    alones = 2 if loners == 1 else 1

    r2v = np.zeros(PLAYERS + 1, dtype=np.int64)
    r2c = np.full(PLAYERS + 1, -1, dtype=np.int64)
    r2t = np.full(PLAYERS + 1, -1, dtype=np.int64)
    r2a = np.zeros(PLAYERS + 1, dtype=np.int64)

    for idx in range(PLAYERS - 1, -1, -1):
        seat = (dealer + 1 + idx) & 3
        have = False
        best = 0
        b_caller = -1
        b_trump = -1
        b_alone = 0
        if not (stick == 1 and idx == PLAYERS - 1):
            best = r2v[idx + 1]
            b_caller = r2c[idx + 1]
            b_trump = r2t[idx + 1]
            b_alone = r2a[idx + 1]
            have = True
        for suit in range(4):
            if suit == up_suit:
                continue
            for alone in range(alones):
                value = name_suit_value(hands, suit, dealer, seat, alone,
                                        tt, ttm, nodes, stk)
                if not have or prefers(value, best, seat):
                    best = value
                    b_caller = seat
                    b_trump = suit
                    b_alone = alone
                    have = True
        r2v[idx] = best
        r2c[idx] = b_caller
        r2t[idx] = b_trump
        r2a[idx] = b_alone

    value = r2v[0]
    caller = r2c[0]
    trump = r2t[0]
    alone = r2a[0]
    for idx in range(PLAYERS - 1, -1, -1):
        seat = (dealer + 1 + idx) & 3
        for a in range(alones):
            got = order_up_value(hands, up_card, dealer, seat, a,
                                 tt, ttm, nodes, stk)
            if prefers(got, value, seat):
                value = got
                caller = seat
                trump = up_suit
                alone = a
        # `value` is now what the auction is worth from this seat onward, which
        # is exactly what the seat before it is choosing against.

    out[row, R_VALUE] = value
    out[row, R_CALLER] = caller
    out[row, R_TRUMP] = trump
    out[row, R_ALONE] = alone


@njit(parallel=True, cache=True)
def run_deals(lo, hi, seed, pin_hand, pin_seat, pin_up, dealer,
              budget_play, budget_bid, budget_discard,
              epsilon, racing, min_worlds,
              pass_model, assume, let_auction_play, stick, loners,
              chunks, tt, out, god_out, want_god, counters):
    """
    Deals `lo` up to `hi` of the sweep, spread over `chunks` threads.

    Deals are handed out round-robin rather than in blocks, so a thread that
    draws a run of cheap deals does not finish early and sit idle. Each deal
    seeds its own streams from its own index, so **the answer does not depend
    on the thread count** -- the same property the process pool had, and the
    cheapest available check that the parallel path is sound.

    **One transposition table, shared, and never cleared.** A stored entry is
    keyed by everything its value depends on, so it is as true for the next
    deal as for the one that wrote it, and as true on another thread as on the
    one that wrote it; and since the key is the *compressed* position, deals
    that differ in every card still land on it. Warming across deals -- and
    across threads -- is most of what makes the sweep cost less than the sum of
    its solves.

    Sharing it needs no lock because an entry is a single int64 and an aligned
    64-bit store is atomic: a reader sees some writer's whole entry or some
    other writer's whole entry, never half of each. Two threads racing for a
    slot is not a correctness question, only a question of whose true answer
    stays.
    """
    mask = np.int64(len(tt) - 1)
    for c in prange(chunks):
        nodes = np.zeros(8, dtype=np.int64)
        stk = np.zeros((bitcore.STACK_PLIES, bitcore.STACK_FIELDS),
                       dtype=np.int64)
        rngs = np.zeros((PLAYERS + 1, 1), dtype=np.int64)
        hands = np.zeros(PLAYERS, dtype=np.int64)
        for i in range(lo + c, hi, chunks):
            seed_stream(rngs[PLAYERS], seed + i)
            for s in range(PLAYERS):
                seed_stream(rngs[s], seed * 7919 + i * 4 + s)
            play_one_deal(pin_hand, pin_seat, pin_up, dealer,
                          budget_play, budget_bid, budget_discard,
                          epsilon, racing, min_worlds,
                          pass_model, assume, let_auction_play, stick, loners,
                          rngs, tt, mask, nodes, stk, out, i)
            if want_god == 1:
                # The same layout in God Mode. Dealt again from the same
                # stream, since the deal is a function of its index alone.
                seed_stream(rngs[PLAYERS], seed + i)
                up, _ = deal_around(pin_hand, pin_seat, pin_up, dealer,
                                    rngs[PLAYERS], hands)
                solve_bidding(hands, up, dealer, stick, loners,
                              tt, mask, nodes, stk, god_out, i)
        for z in range(COUNTERS):
            # Added to, not assigned: a progress line splits the sweep into
            # blocks, and each block is another call into here.
            counters[c, z] += nodes[z]
