"""
Correctness checks for fast_search.py.

fast_search is validated two ways, neither of which trusts tree_search.py:

  1. against reference_solver.py, a pure-Python solver written straight from
     the rules with different move ordering and no forced-outcome cutoffs;
  2. by replaying the line fast_search returns and checking every card was
     held, followed suit, and won the trick the solver claimed.

Every hand is put through both twice: four-handed, and again as a loner with the
caller's partner sitting out. `_search_alone` is a hand copy of `_search`, so it
needs the sweep at least as much as the original does.

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


def check_line(hands, starting_player, caller, score, ps, pv, pp, winners,
               alone=False):
    """Replay a returned line and report every rule or bookkeeping violation.

    With `alone` the line is three cards a trick, seat order skips the caller's
    partner, and all five of its cards must still be in hand at the end.
    """
    errs = []
    remaining = [[tuple(int(v) for v in c) for c in hands[p]] for p in range(4)]
    caller_team = caller % 2
    caller_tricks = 0

    sitting = (caller + 2) % 4 if alone else -1
    width = 3 if alone else 4
    if ps.shape[1] != width:
        return ["line is %d cards a trick, expected %d" % (ps.shape[1], width)]
    if starting_player == sitting:
        starting_player = (starting_player + 1) % 4

    for t in range(5):
        leader = starting_player if t == 0 else int(winners[t - 1])
        order = [s % 4 for s in range(leader, leader + 4) if s % 4 != sitting]
        led_suit = None
        played = []

        for k in range(width):
            p = int(pp[t, k])
            if p != order[k]:
                errs.append("trick %d seat %d: player %d, expected %d"
                            % (t + 1, k, p, order[k]))

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

    if alone and len(remaining[sitting]) != 5:
        errs.append("seat %d sat out but played %d cards"
                    % (sitting, 5 - len(remaining[sitting])))

    march = 4 if alone else 2
    expected = march if caller_tricks == 5 else (1 if caller_tricks >= 3 else -2)
    if expected != score:
        errs.append("solver returned %d but the line takes %d tricks (-> %d)"
                    % (score, caller_tricks, expected))
    return errs


def sweep(hands, cfg, n_cross, alone):
    """Solve every hand, replay every line, cross-check the first n_cross."""
    n_hands = len(cfg)
    failures = 0
    label = "alone" if alone else "four-handed"

    solve(hands[0], 0, 0, alone)  # warm the JIT before timing

    t0 = time.perf_counter()
    scores = np.zeros(n_hands, dtype=np.int64)
    for i in range(n_hands):
        scores[i] = solve(hands[i], cfg[i][0], cfg[i][1], alone)[0]
    elapsed = time.perf_counter() - t0
    print("%s: solved %d hands in %.3fs (%.3f ms/hand)"
          % (label, n_hands, elapsed, 1000 * elapsed / n_hands))

    for i in range(n_hands):
        sp, cl = cfg[i]
        sc, ps, pv, pp, wn = solve_line(hands[i], sp, cl, alone)
        if sc != scores[i]:
            print("FAIL hand %d: solve=%d solve_line=%d" % (i, scores[i], sc))
            failures += 1
            continue
        errs = check_line(hands[i], sp, cl, sc, ps, pv, pp, wn, alone)
        if errs:
            print("FAIL hand %d: %s" % (i, errs[:3]))
            failures += 1
    print("%s: line legality %d/%d hands clean"
          % (label, n_hands - failures, n_hands))

    mism = 0
    n_cross = min(n_cross, n_hands)
    for i in range(n_cross):
        sp, cl = cfg[i]
        if solve_py(hands_to_py(hands[i]), sp, cl, alone) != scores[i]:
            print("FAIL hand %d: pure-Python reference disagrees" % i)
            mism += 1
    print("%s: cross-check vs reference_solver %d/%d agree"
          % (label, n_cross - mism, n_cross))

    counts = {int(v): int((scores == v).sum()) for v in np.unique(scores)}
    print("%s: outcome distribution %s\n" % (label, counts))
    return failures + mism


def main(n_hands=400, n_cross=150):
    failures = 0

    score, ps, pv, pp, wn = solve_line(TEST_HAND, 2, 0)
    assert score == 2, "test_hand.txt should be a march for the calling team, got %d" % score
    assert not check_line(TEST_HAND, 2, 0, score, ps, pv, pp, wn)
    print("test_hand.txt (starting_player=2, caller=0) -> %d  [ok]" % score)

    lone, ps, pv, pp, wn = solve_line(TEST_HAND, 2, 0, True)
    assert lone == -2, "test_hand.txt alone should be a euchre, got %d" % lone
    assert not check_line(TEST_HAND, 2, 0, lone, ps, pv, pp, wn, True)
    print("test_hand.txt alone (seat 2 sits out)       -> %d  [ok]\n" % lone)

    np.random.seed(20240607)
    hands = generate_hands(n_games=n_hands)
    cfg = [(i % 4, (i // 4) % 4) for i in range(n_hands)]

    # The same hands twice: four-handed, then with the caller's partner out.
    failures += sweep(hands, cfg, n_cross, alone=False)
    failures += sweep(hands, cfg, n_cross, alone=True)

    print("%s" % ("ALL CHECKS PASSED" if failures == 0 else "%d FAILURES" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    sys.exit(main(n))
