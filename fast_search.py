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
def _resolve(t_suit, t_str, t_player, r):
    """Winner of the complete 4-card trick held in row r."""
    have_trump = False
    for k in range(4):
        if t_suit[r, k] == TRUMP:
            have_trump = True
            break
    best_s = -1
    win_idx = 0
    if have_trump:
        for k in range(4):
            if t_suit[r, k] == TRUMP and t_str[r, k] > best_s:
                best_s = t_str[r, k]
                win_idx = k
    else:
        led = t_suit[r, 0]
        for k in range(4):
            if t_suit[r, k] == led and t_str[r, k] > best_s:
                best_s = t_str[r, k]
                win_idx = k
    return t_player[r, win_idx]


@njit
def _final(caller_tricks):
    if caller_tricks == _ALL:
        return 2
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
            w = _resolve(t_suit, t_str, t_player, trick_no)
            nct = caller_tricks
            if (w % 2) == caller_team:
                nct += 1
            if trick_no + 1 == _ALL:
                val = _final(nct)
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
def solve(hands, starting_player, caller):
    """Score for the calling team under perfect play. Returns (score, nodes)."""
    _validate(hands, starting_player, caller)
    suits, strs = encode_hands(hands)
    n = np.full(4, hands.shape[1], dtype=np.int64)
    t_suit = np.zeros((_ALL, 4), dtype=np.int64)
    t_str = np.zeros((_ALL, 4), dtype=np.int64)
    t_player = np.zeros((_ALL, 4), dtype=np.int64)
    nodes = np.zeros(1, dtype=np.int64)
    v = _search(suits, strs, n, t_suit, t_str, t_player,
                0, starting_player, caller % 2, 0, 0, -1000, 1000, nodes)
    return v, nodes[0]


@njit
def solve_line(hands, starting_player, caller):
    """
    Solve, and also recover one optimal line of play.

    Returns (score, play_suit, play_str, play_player, winners); the play_*
    arrays are (tricks, 4), indexed [trick, position within the trick].
    """
    _validate(hands, starting_player, caller)
    suits, strs = encode_hands(hands)
    ncards = _ALL
    n = np.full(4, ncards, dtype=np.int64)
    t_suit = np.zeros((ncards, 4), dtype=np.int64)
    t_str = np.zeros((ncards, 4), dtype=np.int64)
    t_player = np.zeros((ncards, 4), dtype=np.int64)
    nodes = np.zeros(1, dtype=np.int64)
    caller_team = caller % 2

    play_s = np.zeros((ncards, 4), dtype=np.int64)
    play_v = np.zeros((ncards, 4), dtype=np.int64)
    play_p = np.zeros((ncards, 4), dtype=np.int64)
    winners = np.zeros(ncards, dtype=np.int64)

    to_act = starting_player
    caller_tricks = 0
    trick_no = 0
    n_in_trick = 0

    while trick_no < ncards:
        cnt = n[to_act]
        last = cnt - 1
        led = -1
        has_led = False
        if n_in_trick > 0:
            led = t_suit[trick_no, 0]
            for i in range(cnt):
                if suits[to_act, i] == led:
                    has_led = True
                    break

        maximizing = (to_act % 2) == caller_team
        best_val = -1000 if maximizing else 1000
        best_i = -1

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

            if n_in_trick == 3:
                w = _resolve(t_suit, t_str, t_player, trick_no)
                nct = caller_tricks
                if (w % 2) == caller_team:
                    nct += 1
                if trick_no + 1 == ncards:
                    val = _final(nct)
                else:
                    val = _search(suits, strs, n, t_suit, t_str, t_player,
                                  0, w, caller_team, nct, trick_no + 1,
                                  -1000, 1000, nodes)
            else:
                val = _search(suits, strs, n, t_suit, t_str, t_player,
                              n_in_trick + 1, (to_act + 1) % 4, caller_team,
                              caller_tricks, trick_no, -1000, 1000, nodes)

            n[to_act] = cnt
            s_tmp = suits[to_act, i]
            v_tmp = strs[to_act, i]
            suits[to_act, i] = suits[to_act, last]
            strs[to_act, i] = strs[to_act, last]
            suits[to_act, last] = s_tmp
            strs[to_act, last] = v_tmp

            if maximizing:
                if val > best_val:
                    best_val = val
                    best_i = i
            else:
                if val < best_val:
                    best_val = val
                    best_i = i

        # commit the chosen card
        i = best_i
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

        if n_in_trick == 3:
            w = _resolve(t_suit, t_str, t_player, trick_no)
            winners[trick_no] = w
            if (w % 2) == caller_team:
                caller_tricks += 1
            trick_no += 1
            n_in_trick = 0
            to_act = w
        else:
            n_in_trick += 1
            to_act = (to_act + 1) % 4

    return _final(caller_tricks), play_s, play_v, play_p, winners


def _decode(suit, strength):
    """(suit, strength) -> the original 2D vector representation."""
    if suit == 0:
        return [int(strength), 0]
    if suit == 2:
        return [-int(strength), 0]
    if suit == TRUMP:
        return [0, int(strength)]
    return [0, -int(strength)]


def definitive_winner(dealt_hands, starting_player, caller, verbose=False):
    """
    Drop-in replacement for tree_search.definitive_winner.

    Returns the calling team's score: +2 march, +1 win, -2 euchred.
    """
    dealt_hands = np.ascontiguousarray(dealt_hands, dtype=np.int64)
    if dealt_hands.ndim != 3:
        raise ValueError(
            "dealt_hands must be shaped (4, 5, 2), got %r" % (dealt_hands.shape,)
        )
    if not verbose:
        return int(solve(dealt_hands, int(starting_player), int(caller))[0])

    score, ps, pv, pp, winners = solve_line(
        dealt_hands, int(starting_player), int(caller)
    )
    print("Starting hands:\n", dealt_hands)
    for t in range(ps.shape[0]):
        cards = [_decode(ps[t, k], pv[t, k]) for k in range(4)]
        order = [int(pp[t, k]) for k in range(4)]
        print("Trick %d: %s  (played by %s)" % (t + 1, cards, order))
        print("Trick %d winner: %d" % (t + 1, winners[t]))
    print("Final result:", winners.tolist())
    return int(score)
