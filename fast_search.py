"""
Depth-first alpha-beta solver for a single Euchre hand (spades = trump).

Replaces the breadth-first "materialise every branch, then filter" pipeline in
tree_search.py. That design expands the game tree once per candidate opening
card and again once per responder prune, for every trick -- roughly 40 tree
expansions per hand. This walks the tree once, depth-first, and prunes with
alpha-beta instead of with play heuristics.

Cards keep the existing 2D-vector encoding on the way in; internally they are
decomposed into (suit, strength) integers so the hot loop never touches
floating point or np.linalg.norm.

Loners are supported by `alone=True`: the caller's partner sits out, tricks are
three cards instead of four, and a march is worth 4 instead of 2. That is a
second recursion, `_search_alone`, rather than a trick width threaded through
the existing one -- see the comment above it for the measurement that decided
it.

NOTE: do not add cache=True to these functions. Numba 0.60 segfaults on cache
load for recursive njit functions (verified: reliable SIGSEGV on _search).
"""
import numpy as np
from numba import njit

# suit codes: 0 = diamonds (x>0), 1 = spades/trump (x==0,y>0),
#             2 = hearts (x<0),   3 = clubs (x==0,y<0)
TRUMP = 1

# tricks the calling team needs, and the number of tricks in a hand
_NEEDED = 3
_ALL = 5

# points for taking every trick: four-handed, and alone
MARCH = 2
LONE_MARCH = 4


@njit
def encode_hands(hands):
    """(P, C, 2) vector cards -> integer (suit, strength) planes."""
    n_p, n_c = hands.shape[0], hands.shape[1]
    suits = np.zeros((n_p, n_c), dtype=np.int64)
    strength = np.zeros((n_p, n_c), dtype=np.int64)
    for p in range(n_p):
        for c in range(n_c):
            x = hands[p, c, 0]
            y = hands[p, c, 1]
            if x > 0:
                suits[p, c] = 0
                strength[p, c] = x
            elif x < 0:
                suits[p, c] = 2
                strength[p, c] = -x
            elif y > 0:
                suits[p, c] = 1
                strength[p, c] = y
            else:
                suits[p, c] = 3
                strength[p, c] = -y
    return suits, strength


@njit
def _trick_size(sitting):
    """Cards in a complete trick: four normally, three with a seat sitting out."""
    if sitting < 0:
        return 4
    return 3


@njit
def _next(seat, sitting):
    """The seat that acts after `seat`, stepping over a sitting partner."""
    s = (seat + 1) % 4
    if s == sitting:
        s = (s + 1) % 4
    return s


@njit
def _resolve(t_suit, t_str, t_player, r, size):
    """Winner of the complete size-card trick held in row r."""
    have_trump = False
    for k in range(size):
        if t_suit[r, k] == TRUMP:
            have_trump = True
            break
    best_s = -1
    win_idx = 0
    if have_trump:
        for k in range(size):
            if t_suit[r, k] == TRUMP and t_str[r, k] > best_s:
                best_s = t_str[r, k]
                win_idx = k
    else:
        led = t_suit[r, 0]
        for k in range(size):
            if t_suit[r, k] == led and t_str[r, k] > best_s:
                best_s = t_str[r, k]
                win_idx = k
    return t_player[r, win_idx]


@njit
def _final(caller_tricks, alone):
    """Calling team's score. A lone march pays 4; nothing else changes."""
    if caller_tricks == _ALL:
        if alone:
            return LONE_MARCH
        return MARCH
    elif caller_tricks >= _NEEDED:
        return 1
    return -2


@njit
def _search(suits, strs, n, t_suit, t_str, t_player,
            n_in_trick, to_act, caller_team, caller_tricks, trick_no,
            alpha, beta, nodes):
    nodes[0] += 1

    if n_in_trick == 0:
        # The result is already decided; playing it out cannot change it.
        tricks_left = _ALL - trick_no
        if caller_tricks + tricks_left < _NEEDED:
            return -2
        if caller_tricks >= _NEEDED and (trick_no - caller_tricks) >= 1:
            return 1

    maximizing = (to_act % 2) == caller_team
    cnt = n[to_act]

    led = -1
    has_led = False
    if n_in_trick > 0:
        led = t_suit[trick_no, 0]
        for i in range(cnt):
            if suits[to_act, i] == led:
                has_led = True
                break

    best = -1000 if maximizing else 1000
    last = cnt - 1

    for i in range(cnt):
        if has_led and suits[to_act, i] != led:
            continue

        t_suit[trick_no, n_in_trick] = suits[to_act, i]
        t_str[trick_no, n_in_trick] = strs[to_act, i]
        t_player[trick_no, n_in_trick] = to_act

        # play card i: swap it to the end of the hand, then shrink the hand
        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp
        n[to_act] = last

        if n_in_trick == 3:
            w = _resolve(t_suit, t_str, t_player, trick_no, 4)
            nct = caller_tricks
            if (w % 2) == caller_team:
                nct += 1
            if trick_no + 1 == _ALL:
                val = _final(nct, False)
            else:
                val = _search(suits, strs, n, t_suit, t_str, t_player,
                              0, w, caller_team, nct, trick_no + 1,
                              alpha, beta, nodes)
        else:
            val = _search(suits, strs, n, t_suit, t_str, t_player,
                          n_in_trick + 1, (to_act + 1) % 4, caller_team,
                          caller_tricks, trick_no, alpha, beta, nodes)

        # undo: the same swap puts the card back where it came from
        n[to_act] = cnt
        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp

        if maximizing:
            if val > best:
                best = val
            if best > alpha:
                alpha = best
        else:
            if val < best:
                best = val
            if best < beta:
                beta = best
        if beta <= alpha:
            break

    return best


# _search_alone is _search with three seats instead of four. It is a copy on
# purpose, and the duplication was measured rather than guessed: threading the
# trick width and the sitting seat through the single recursion instead cost the
# four-handed path 1.7x-2.5x per node (identical node counts, so purely the
# per-node width test and the skip-the-sitting-seat step). Four-handed play is
# the hot path -- an EV sweep is tens of thousands of those solves -- so it keeps
# its constants and loners get their own function.
#
# What differs, and nothing else does: a trick completes at three cards rather
# than four, the next seat steps over `sitting`, and a march scores 4. Both
# functions are checked against reference_solver and against the no-pruning
# minimax in tests/euchre_testkit.py, which is what guards the copy from drift.
@njit
def _search_alone(suits, strs, n, t_suit, t_str, t_player,
                  n_in_trick, to_act, caller_team, caller_tricks, trick_no,
                  alpha, beta, nodes, sitting):
    nodes[0] += 1

    if n_in_trick == 0:
        # The result is already decided; playing it out cannot change it.
        tricks_left = _ALL - trick_no
        if caller_tricks + tricks_left < _NEEDED:
            return -2
        if caller_tricks >= _NEEDED and (trick_no - caller_tricks) >= 1:
            return 1

    maximizing = (to_act % 2) == caller_team
    cnt = n[to_act]

    led = -1
    has_led = False
    if n_in_trick > 0:
        led = t_suit[trick_no, 0]
        for i in range(cnt):
            if suits[to_act, i] == led:
                has_led = True
                break

    best = -1000 if maximizing else 1000
    last = cnt - 1

    for i in range(cnt):
        if has_led and suits[to_act, i] != led:
            continue

        t_suit[trick_no, n_in_trick] = suits[to_act, i]
        t_str[trick_no, n_in_trick] = strs[to_act, i]
        t_player[trick_no, n_in_trick] = to_act

        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp
        n[to_act] = last

        if n_in_trick == 2:
            w = _resolve(t_suit, t_str, t_player, trick_no, 3)
            nct = caller_tricks
            if (w % 2) == caller_team:
                nct += 1
            if trick_no + 1 == _ALL:
                val = _final(nct, True)
            else:
                val = _search_alone(suits, strs, n, t_suit, t_str, t_player,
                                    0, w, caller_team, nct, trick_no + 1,
                                    alpha, beta, nodes, sitting)
        else:
            val = _search_alone(suits, strs, n, t_suit, t_str, t_player,
                                n_in_trick + 1, _next(to_act, sitting),
                                caller_team, caller_tricks, trick_no,
                                alpha, beta, nodes, sitting)

        n[to_act] = cnt
        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp

        if maximizing:
            if val > best:
                best = val
            if best > alpha:
                alpha = best
        else:
            if val < best:
                best = val
            if best < beta:
                beta = best
        if beta <= alpha:
            break

    return best


@njit
def _validate(hands, starting_player, caller):
    """Reject inputs the search cannot handle.

    The hot loop has no bounds checking, so an out-of-range seat indexed
    straight into the (4,) / (4, 5) state arrays and segfaulted the process.
    """
    if hands.shape[0] != 4:
        raise ValueError("hands must hold exactly 4 players")
    if hands.shape[1] != _ALL:
        raise ValueError("each hand must hold exactly 5 cards")
    if hands.shape[2] != 2:
        raise ValueError("each card must be a 2-element vector")
    if starting_player < 0 or starting_player > 3:
        raise ValueError("starting_player must be in 0..3")
    if caller < 0 or caller > 3:
        raise ValueError("caller must be in 0..3")


@njit
def _setup(hands, starting_player, caller, alone):
    """
    Shared per-solve state: (suits, strs, n, sitting, leader, trick width).

    A loner's partner keeps its dealt cards in the arrays but is given a count
    of zero, so it can never play one -- that hand is out of play exactly like
    the kitty. The lead still belongs to the seat left of the dealer, so when
    that seat is the one sitting out the lead passes on to the next live seat.
    """
    suits, strs = encode_hands(hands)
    n = np.full(4, hands.shape[1], dtype=np.int64)
    sitting = -1
    leader = starting_player
    if alone:
        sitting = (caller + 2) % 4
        n[sitting] = 0
        if leader == sitting:
            leader = (leader + 1) % 4
    return suits, strs, n, sitting, leader, _trick_size(sitting)


@njit
def solve(hands, starting_player, caller, alone=False):
    """God Mode score for the calling team. Returns (score, nodes)."""
    _validate(hands, starting_player, caller)
    suits, strs, n, sitting, leader, width = _setup(
        hands, starting_player, caller, alone)
    t_suit = np.zeros((_ALL, width), dtype=np.int64)
    t_str = np.zeros((_ALL, width), dtype=np.int64)
    t_player = np.zeros((_ALL, width), dtype=np.int64)
    nodes = np.zeros(1, dtype=np.int64)
    if sitting < 0:
        v = _search(suits, strs, n, t_suit, t_str, t_player,
                    0, leader, caller % 2, 0, 0, -1000, 1000, nodes)
    else:
        v = _search_alone(suits, strs, n, t_suit, t_str, t_player,
                          0, leader, caller % 2, 0, 0, -1000, 1000, nodes,
                          sitting)
    return v, nodes[0]


@njit
def _descend(suits, strs, n, t_suit, t_str, t_player, n_in_trick, to_act,
             caller_team, caller_tricks, trick_no, nodes, sitting):
    """
    Value of the position under a card solve_line has just committed.

    Picks the right recursion for the table size; the full alpha-beta window is
    deliberate, since solve_line evaluates every candidate card rather than
    cutting off as soon as one is good enough.
    """
    if sitting < 0:
        return _search(suits, strs, n, t_suit, t_str, t_player, n_in_trick,
                       to_act, caller_team, caller_tricks, trick_no,
                       -1000, 1000, nodes)
    return _search_alone(suits, strs, n, t_suit, t_str, t_player, n_in_trick,
                         to_act, caller_team, caller_tricks, trick_no,
                         -1000, 1000, nodes, sitting)


# _position_moves is the one-ply fan-out: it plays each legal card in turn and
# solves the position that results, so the caller sees a value per candidate
# rather than a single best. solve_line uses it to pick a move, and
# position_moves exposes it for search from a partially played position, which
# is what a player who has to *choose* a card needs.
#
# The alpha-beta window is deliberately left wide open. solve_line and PIMC
# both want every candidate's true value, and a narrowing window would return
# bounds rather than values for the also-rans.
@njit
def _position_moves(suits, strs, n, t_suit, t_str, t_player, n_in_trick,
                    to_act, caller_team, caller_tricks, trick_no, sitting,
                    width, nodes):
    """
    Value of every legal card for the seat to act, from the caller's side.

    Returns (idx, vals, m): the first m entries of idx are indices into
    to_act's hand and vals holds their values. State is restored exactly.
    """
    cnt = n[to_act]
    idx = np.full(cnt, -1, dtype=np.int64)
    vals = np.zeros(cnt, dtype=np.int64)
    m = 0

    led = -1
    has_led = False
    if n_in_trick > 0:
        led = t_suit[trick_no, 0]
        for i in range(cnt):
            if suits[to_act, i] == led:
                has_led = True
                break

    last = cnt - 1
    for i in range(cnt):
        if has_led and suits[to_act, i] != led:
            continue

        t_suit[trick_no, n_in_trick] = suits[to_act, i]
        t_str[trick_no, n_in_trick] = strs[to_act, i]
        t_player[trick_no, n_in_trick] = to_act

        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp
        n[to_act] = last

        if n_in_trick == width - 1:
            w = _resolve(t_suit, t_str, t_player, trick_no, width)
            nct = caller_tricks
            if (w % 2) == caller_team:
                nct += 1
            if trick_no + 1 == _ALL:
                val = _final(nct, sitting >= 0)
            else:
                val = _descend(suits, strs, n, t_suit, t_str, t_player,
                               0, w, caller_team, nct, trick_no + 1,
                               nodes, sitting)
        else:
            val = _descend(suits, strs, n, t_suit, t_str, t_player,
                           n_in_trick + 1, _next(to_act, sitting),
                           caller_team, caller_tricks, trick_no,
                           nodes, sitting)

        n[to_act] = cnt
        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp

        idx[m] = i
        vals[m] = val
        m += 1

    return idx, vals, m


@njit
def solve_line(hands, starting_player, caller, alone=False):
    """
    Solve, and also recover one optimal line of play.

    Returns (score, play_suit, play_str, play_player, winners); the play_*
    arrays are (tricks, cards per trick), indexed [trick, position within the
    trick]. A loner's tricks are three cards wide, so they come back (5, 3).
    """
    _validate(hands, starting_player, caller)
    suits, strs, n, sitting, leader, width = _setup(
        hands, starting_player, caller, alone)
    ncards = _ALL
    t_suit = np.zeros((ncards, width), dtype=np.int64)
    t_str = np.zeros((ncards, width), dtype=np.int64)
    t_player = np.zeros((ncards, width), dtype=np.int64)
    nodes = np.zeros(1, dtype=np.int64)
    caller_team = caller % 2
    alone_flag = sitting >= 0

    play_s = np.zeros((ncards, width), dtype=np.int64)
    play_v = np.zeros((ncards, width), dtype=np.int64)
    play_p = np.zeros((ncards, width), dtype=np.int64)
    winners = np.zeros(ncards, dtype=np.int64)

    to_act = leader
    caller_tricks = 0
    trick_no = 0
    n_in_trick = 0

    while trick_no < ncards:
        idx, vals, m = _position_moves(
            suits, strs, n, t_suit, t_str, t_player, n_in_trick, to_act,
            caller_team, caller_tricks, trick_no, sitting, width, nodes)

        maximizing = (to_act % 2) == caller_team
        best_val = -1000 if maximizing else 1000
        best_i = -1
        for k in range(m):
            if maximizing:
                if vals[k] > best_val:
                    best_val = vals[k]
                    best_i = idx[k]
            else:
                if vals[k] < best_val:
                    best_val = vals[k]
                    best_i = idx[k]

        # commit the chosen card
        i = best_i
        cnt = n[to_act]
        last = cnt - 1
        t_suit[trick_no, n_in_trick] = suits[to_act, i]
        t_str[trick_no, n_in_trick] = strs[to_act, i]
        t_player[trick_no, n_in_trick] = to_act
        play_s[trick_no, n_in_trick] = suits[to_act, i]
        play_v[trick_no, n_in_trick] = strs[to_act, i]
        play_p[trick_no, n_in_trick] = to_act

        s_tmp = suits[to_act, i]
        v_tmp = strs[to_act, i]
        suits[to_act, i] = suits[to_act, last]
        strs[to_act, i] = strs[to_act, last]
        suits[to_act, last] = s_tmp
        strs[to_act, last] = v_tmp
        n[to_act] = last

        if n_in_trick == width - 1:
            w = _resolve(t_suit, t_str, t_player, trick_no, width)
            winners[trick_no] = w
            if (w % 2) == caller_team:
                caller_tricks += 1
            trick_no += 1
            n_in_trick = 0
            to_act = w
        else:
            n_in_trick += 1
            to_act = _next(to_act, sitting)

    return _final(caller_tricks, alone_flag), play_s, play_v, play_p, winners


def _decode(suit, strength):
    """(suit, strength) -> the original 2D vector representation."""
    if suit == 0:
        return [int(strength), 0]
    if suit == 2:
        return [-int(strength), 0]
    if suit == TRUMP:
        return [0, int(strength)]
    return [0, -int(strength)]


def sitting_seat(caller, alone=True):
    """The seat that sits out a loner called by `caller`; -1 if nobody sits."""
    return (int(caller) + 2) % 4 if alone else -1


def definitive_winner(dealt_hands, starting_player, caller, verbose=False,
                      alone=False):
    """
    Drop-in replacement for tree_search.definitive_winner.

    Returns the calling team's score: +2 march, +1 win, -2 euchred -- and +4
    for a march called alone. With `alone=True` the caller's partner sits out
    and its dealt cards never enter play.
    """
    dealt_hands = np.ascontiguousarray(dealt_hands, dtype=np.int64)
    if dealt_hands.ndim != 3:
        raise ValueError(
            "dealt_hands must be shaped (4, 5, 2), got %r" % (dealt_hands.shape,)
        )
    alone = bool(alone)
    if not verbose:
        return int(solve(dealt_hands, int(starting_player), int(caller), alone)[0])

    score, ps, pv, pp, winners = solve_line(
        dealt_hands, int(starting_player), int(caller), alone
    )
    print("Starting hands:\n", dealt_hands)
    if alone:
        print("Seat %d called alone; seat %d sits out."
              % (int(caller), sitting_seat(caller)))
    for t in range(ps.shape[0]):
        cards = [_decode(ps[t, k], pv[t, k]) for k in range(ps.shape[1])]
        order = [int(pp[t, k]) for k in range(pp.shape[1])]
        print("Trick %d: %s  (played by %s)" % (t + 1, cards, order))
        print("Trick %d winner: %d" % (t + 1, winners[t]))
    print("Final result:", winners.tolist())
    return int(score)


# ------------------------------------------------- partially played positions
#
# solve() and solve_line() both start from a fresh deal: five cards each, no
# trick in progress, nothing won yet. A player who has to *choose* a card is
# never in that position after the opening lead, so PIMC -- or any other
# search-at-your-turn player -- needs an entry point that takes the position as
# it actually stands.
#
# Nothing in the recursion had to change for this. `trick_no` and
# `caller_tricks` were already absolute rather than relative, and `n` was
# already a per-seat count, so entering at trick 3 with two cards each and one
# trick already won is just a different set of arguments. The two hardcoded
# constants stay true: a hand is still _ALL tricks long and the calling team
# still needs _NEEDED of them, counting the ones already in the bag.
#
# What is new is the validation, and it earns its length. njit does no bounds
# checking, so a seat index or a card count that disagrees with the trick
# number reads past the end of the state arrays and takes the interpreter with
# it -- the same failure mode `_validate` was written for.


def _encode_cards(cards):
    """(k, 2) vector cards -> (suits, strengths), each (k,)."""
    cards = np.ascontiguousarray(cards, dtype=np.int64).reshape(1, -1, 2)
    suits, strs = encode_hands(cards)
    return suits[0], strs[0]


def _check_position(hands, counts, trick_cards, trick_players, to_act, caller,
                    caller_tricks, trick_no, alone):
    """
    Reject any position the search cannot handle, with a readable message.

    The count-per-seat rule is the load-bearing one: by trick `trick_no` every
    live seat has played exactly `trick_no` cards, plus one more if it has
    already played to the trick now on the table. A position that disagrees
    describes a hand that cannot have happened, and the search would go ahead
    and solve it anyway and hand back a number.
    """
    if hands.ndim != 3 or hands.shape[0] != 4 or hands.shape[2] != 2:
        raise ValueError("hands must be shaped (4, cards, 2), got %r"
                         % (hands.shape,))
    if counts.shape != (4,):
        raise ValueError("counts must hold one card count per seat, got %r"
                         % (counts.shape,))
    if trick_cards.shape[0] != trick_players.shape[0]:
        raise ValueError("%d cards played to the trick but %d players named"
                         % (trick_cards.shape[0], trick_players.shape[0]))
    if not 0 <= trick_no < _ALL:
        raise ValueError("trick_no must be in 0..%d, got %d"
                         % (_ALL - 1, trick_no))
    if not 0 <= caller <= 3:
        raise ValueError("caller must be in 0..3, got %d" % caller)
    if not 0 <= to_act <= 3:
        raise ValueError("to_act must be in 0..3, got %d" % to_act)
    if not 0 <= caller_tricks <= trick_no:
        raise ValueError("caller_tricks must be in 0..%d by trick %d, got %d"
                         % (trick_no, trick_no, caller_tricks))

    width = 3 if alone else 4
    sitting = (caller + 2) % 4 if alone else -1
    k = int(trick_cards.shape[0])
    if k >= width:
        raise ValueError("a %d-card trick already holds %d cards" % (width, k))

    played_here = [int(p) for p in trick_players]
    if len(set(played_here)) != len(played_here):
        raise ValueError("a seat played twice to the same trick")
    if sitting >= 0 and sitting in played_here:
        raise ValueError("seat %d is sitting out and cannot have played"
                         % sitting)
    if to_act == sitting:
        raise ValueError("seat %d is sitting out and cannot be to act"
                         % sitting)
    if to_act in played_here:
        raise ValueError("seat %d has already played to this trick" % to_act)

    # The seats must have acted in turn, and to_act must be next in that order.
    for a, b in zip(played_here, played_here[1:] + [int(to_act)]):
        if _next(a, sitting) != b:
            raise ValueError("seat %d does not act after seat %d" % (b, a))

    for seat in range(4):
        held = int(counts[seat])
        if not 0 <= held <= hands.shape[1]:
            raise ValueError("seat %d holds %d cards, which does not fit a "
                             "(4, %d, 2) hand array"
                             % (seat, held, hands.shape[1]))
        if seat == sitting:
            if held != 0:
                raise ValueError("seat %d is sitting out but holds %d cards"
                                 % (seat, held))
            continue
        want = _ALL - trick_no - (1 if seat in played_here else 0)
        if held != want:
            raise ValueError("seat %d holds %d cards; at trick %d it should "
                             "hold %d" % (seat, held, trick_no, want))

    live = [tuple(int(v) for v in hands[p, i])
            for p in range(4) for i in range(int(counts[p]))]
    live += [tuple(int(v) for v in c) for c in trick_cards]
    if len(set(live)) != len(live):
        raise ValueError("the position contains duplicate cards")

    return sitting, width, k


def _position_state(hands, counts, trick_cards, trick_players, to_act, caller,
                    caller_tricks, trick_no, alone):
    """Validate, and lay out the mutable arrays one position solve runs on."""
    hands = np.ascontiguousarray(hands, dtype=np.int64)
    counts = np.ascontiguousarray(counts, dtype=np.int64)
    trick_cards = np.ascontiguousarray(
        trick_cards, dtype=np.int64).reshape(-1, 2)
    trick_players = np.ascontiguousarray(
        trick_players, dtype=np.int64).reshape(-1)

    sitting, width, k = _check_position(
        hands, counts, trick_cards, trick_players, to_act, caller,
        caller_tricks, trick_no, alone)

    suits, strs = encode_hands(hands)
    t_suit = np.zeros((_ALL, width), dtype=np.int64)
    t_str = np.zeros((_ALL, width), dtype=np.int64)
    t_player = np.zeros((_ALL, width), dtype=np.int64)
    if k:
        played_s, played_v = _encode_cards(trick_cards)
        t_suit[trick_no, :k] = played_s
        t_str[trick_no, :k] = played_v
        t_player[trick_no, :k] = trick_players

    return (suits, strs, counts.copy(), t_suit, t_str, t_player,
            sitting, width, k)


def solve_position(hands, counts, trick_cards, trick_players, to_act, caller,
                   caller_tricks, trick_no, alone=False):
    """
    God Mode value of a partially played hand, from the caller's side.

    Args:
        hands: (4, C, 2) vector cards. Seat p's live cards are the first
            counts[p] slots of row p; the rest are ignored and may be anything.
        counts: (4,) cards still held per seat. A loner's sitting partner is 0.
        trick_cards: (k, 2) cards already played to the trick now on the table,
            in the order they were played. Empty for a fresh trick.
        trick_players: (k,) the seats that played them.
        to_act: the seat whose turn it is.
        caller: the seat that called trump.
        caller_tricks: tricks the calling team has already taken.
        trick_no: which trick is on the table, 0-based.
        alone: True if the caller is playing alone.

    Returns (score, nodes), the same shape of answer as `solve`.
    """
    alone = bool(alone)
    (suits, strs, n, t_suit, t_str, t_player,
     sitting, width, k) = _position_state(
        hands, counts, trick_cards, trick_players, to_act, caller,
        caller_tricks, trick_no, alone)

    nodes = np.zeros(1, dtype=np.int64)
    v = _descend(suits, strs, n, t_suit, t_str, t_player, k, int(to_act),
                 int(caller) % 2, int(caller_tricks), int(trick_no), nodes,
                 sitting)
    return int(v), int(nodes[0])


def position_moves(hands, counts, trick_cards, trick_players, to_act, caller,
                   caller_tricks, trick_no, alone=False):
    """
    Value of every legal card for the seat to act, from the caller's side.

    Returns (indices, values, nodes). `indices` are positions in
    `hands[to_act, :counts[to_act]]`, in hand order, and `values` the God Mode
    value of playing each. One solve per candidate card -- what a player
    choosing a card needs, and what `solve` (a single value for the position)
    does not give.

    Arguments are `solve_position`'s; see it for their meaning.
    """
    alone = bool(alone)
    (suits, strs, n, t_suit, t_str, t_player,
     sitting, width, k) = _position_state(
        hands, counts, trick_cards, trick_players, to_act, caller,
        caller_tricks, trick_no, alone)

    nodes = np.zeros(1, dtype=np.int64)
    idx, vals, m = _position_moves(
        suits, strs, n, t_suit, t_str, t_player, k, int(to_act),
        int(caller) % 2, int(caller_tricks), int(trick_no), sitting, width,
        nodes)
    return idx[:m].copy(), vals[:m].copy(), int(nodes[0])
