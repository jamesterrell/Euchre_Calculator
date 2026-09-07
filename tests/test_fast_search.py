"""
Correctness checks for fast_search.py.

fast_search is validated two ways, neither of which trusts tree_search.py:

  1. against reference_solver.py, a pure-Python solver written straight from
     the rules with different move ordering and no forced-outcome cutoffs;
  2. by replaying the line fast_search returns and checking every card was
     held, followed suit, and won the trick the solver claimed.

Run:  python test_fast_search.py [n_hands]
"""
import os
import sys
import time

# Run as a script from anywhere: the live modules are at the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fast_search import solve, solve_line, _decode
from reference_solver import solve_py, hands_to_py
from n_game_sim import generate_hands

TEST_HAND = np.array([
    [[0, 140], [0, 135], [0, -9], [-9, 0], [9, 0]],
    [[13, 0], [0, -14], [0, 100], [0, 110], [0, -10]],
    [[0, 90], [10, 0], [14, 0], [-12, 0], [11, 0]],
    [[-11, 0], [-14, 0], [12, 0], [0, -13], [-10, 0]],
], dtype=np.int64)


def suit_of(card):
    x, y = card
    if x > 0:
        return 0
    if x < 0:
        return 2
    if y > 0:
        return 1
    return 3


def check_line(hands, starting_player, caller, score, ps, pv, pp, winners):
    """Replay a returned line and report every rule or bookkeeping violation."""
    errs = []
    remaining = [[tuple(int(v) for v in c) for c in hands[p]] for p in range(4)]
    caller_team = caller % 2
    caller_tricks = 0

    for t in range(5):
        leader = starting_player if t == 0 else int(winners[t - 1])
        led_suit = None
        played = []

        for k in range(4):
            p = int(pp[t, k])
            if p != (leader + k) % 4:
                errs.append("trick %d seat %d: player %d, expected %d"
                            % (t + 1, k, p, (leader + k) % 4))

            card = tuple(_decode(ps[t, k], pv[t, k]))
            if card not in remaining[p]:
                errs.append("trick %d: player %d played %s which it does not hold"
                            % (t + 1, p, card))
            else:
                remaining[p].remove(card)

            s = int(ps[t, k])
            if k == 0:
                led_suit = s
            elif s != led_suit and any(suit_of(c) == led_suit for c in remaining[p]):
                errs.append("trick %d: player %d revoked" % (t + 1, p))
            played.append((s, int(pv[t, k]), p))

        trumps = [x for x in played if x[0] == 1]
        if trumps:
            w = max(trumps, key=lambda x: x[1])[2]
        else:
            w = max((x for x in played if x[0] == led_suit), key=lambda x: x[1])[2]
        if w != int(winners[t]):
            errs.append("trick %d: winner is %d, solver said %d" % (t + 1, w, winners[t]))
        if w % 2 == caller_team:
            caller_tricks += 1

    expected = 2 if caller_tricks == 5 else (1 if caller_tricks >= 3 else -2)
    if expected != score:
        errs.append("solver returned %d but the line takes %d tricks (-> %d)"
                    % (score, caller_tricks, expected))
    return errs


def main(n_hands=400, n_cross=150):
    failures = 0

    score, ps, pv, pp, wn = solve_line(TEST_HAND, 2, 0)
    assert score == 2, "test_hand.txt should be a march for the calling team, got %d" % score
    assert not check_line(TEST_HAND, 2, 0, score, ps, pv, pp, wn)
    print("test_hand.txt (starting_player=2, caller=0) -> %d  [ok]" % score)

    np.random.seed(20240607)
    hands = generate_hands(n_games=n_hands)
    cfg = [(i % 4, (i // 4) % 4) for i in range(n_hands)]

    solve(hands[0], 0, 0)  # warm the JIT before timing

    t0 = time.perf_counter()
    scores = np.zeros(n_hands, dtype=np.int64)
    for i in range(n_hands):
        scores[i] = solve(hands[i], cfg[i][0], cfg[i][1])[0]
    elapsed = time.perf_counter() - t0
    print("solved %d hands in %.3fs (%.3f ms/hand)"
          % (n_hands, elapsed, 1000 * elapsed / n_hands))

    for i in range(n_hands):
        sp, cl = cfg[i]
        sc, ps, pv, pp, wn = solve_line(hands[i], sp, cl)
        if sc != scores[i]:
            print("FAIL hand %d: solve=%d solve_line=%d" % (i, scores[i], sc))
            failures += 1
            continue
        errs = check_line(hands[i], sp, cl, sc, ps, pv, pp, wn)
        if errs:
            print("FAIL hand %d: %s" % (i, errs[:3]))
            failures += 1
    print("line legality: %d/%d hands clean" % (n_hands - failures, n_hands))

    mism = 0
    for i in range(min(n_cross, n_hands)):
        sp, cl = cfg[i]
        if solve_py(hands_to_py(hands[i]), sp, cl) != scores[i]:
            print("FAIL hand %d: pure-Python reference disagrees" % i)
            mism += 1
    print("cross-check vs reference_solver: %d/%d agree"
          % (min(n_cross, n_hands) - mism, min(n_cross, n_hands)))

    failures += mism
    print("\n%s" % ("ALL CHECKS PASSED" if failures == 0 else "%d FAILURES" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    sys.exit(main(n))
