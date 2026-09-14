"""Independent pure-Python solver, written from the Euchre rules directly.

Deliberately structured differently from fast_search.py: no forced-outcome
cutoffs, different move ordering, plain recursion over a list-of-lists state.
Used to check that fast_search's search and its rule encoding agree with a
straightforward reading of the rules. See test_fast_search.py.

Loners are handled the same way here as there, but written out separately: the
caller's partner is emptied, the turn order skips it, a trick is three cards,
and taking all five alone is worth 4.
"""

def card_suit_strength(x, y):
    if x > 0:   return 0, x      # diamonds
    if x < 0:   return 2, -x     # hearts
    if y > 0:   return 1, y      # spades = trump
    return 3, -y                 # clubs


def trick_winner(played):
    """played: list of (suit, strength, player) in play order."""
    trumps = [p for p in played if p[0] == 1]
    if trumps:
        return max(trumps, key=lambda p: p[1])[2]
    led = played[0][0]
    return max((p for p in played if p[0] == led), key=lambda p: p[1])[2]


def solve_py(hands, starting_player, caller, alone=False):
    """hands: list of 4 lists of (suit, strength). Returns calling-team score."""
    caller_team = caller % 2
    hands = [list(h) for h in hands]

    sitting = (caller + 2) % 4 if alone else -1
    width = 3 if alone else 4
    march = 4 if alone else 2
    if alone:
        hands[sitting] = []
        if starting_player == sitting:
            starting_player = (starting_player + 1) % 4

    def nxt(seat):
        s = (seat + 1) % 4
        return (s + 1) % 4 if s == sitting else s

    def rec(to_act, played, caller_tricks, trick_no, alpha, beta):
        if len(played) == width:
            w = trick_winner(played)
            ct = caller_tricks + (1 if w % 2 == caller_team else 0)
            if trick_no + 1 == 5:
                return march if ct == 5 else (1 if ct >= 3 else -2)
            return rec(w, [], ct, trick_no + 1, alpha, beta)

        hand = hands[to_act]
        if played:
            led = played[0][0]
            moves = [c for c in hand if c[0] == led] or list(hand)
        else:
            moves = list(hand)
        # deliberately different ordering from solver.py
        moves.sort(key=lambda c: (c[0], -c[1]))

        maxing = (to_act % 2) == caller_team
        best = -99 if maxing else 99
        for c in moves:
            hand.remove(c)
            v = rec(nxt(to_act), played + [(c[0], c[1], to_act)],
                    caller_tricks, trick_no, alpha, beta)
            hand.append(c)
            if maxing:
                best = max(best, v); alpha = max(alpha, best)
            else:
                best = min(best, v); beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    return rec(starting_player, [], 0, 0, -99, 99)


def hands_to_py(arr):
    return [[card_suit_strength(int(c[0]), int(c[1])) for c in player] for player in arr]
