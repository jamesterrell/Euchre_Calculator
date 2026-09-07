"""
Shared helpers for the unit tests.

Deliberately named so that `unittest discover` does not collect it: it holds no
test cases, only fixtures and independent oracles the tests check the real code
against.

The oracles here are written straight from the Euchre rules and share no code
with fast_search.py, so agreement between them is evidence rather than a
tautology.
"""
import numpy as np

# ---------------------------------------------------------------- card names

# Spades is always trump. T = ten. JC is the left bower and lives on the trump
# axis; JS is the right bower. There is deliberately no jack in the clubs range.
CARD = {
    "9H": (-9, 0), "TH": (-10, 0), "JH": (-11, 0),
    "QH": (-12, 0), "KH": (-13, 0), "AH": (-14, 0),

    "9D": (9, 0), "TD": (10, 0), "JD": (11, 0),
    "QD": (12, 0), "KD": (13, 0), "AD": (14, 0),

    "9C": (0, -9), "TC": (0, -10),
    "QC": (0, -12), "KC": (0, -13), "AC": (0, -14),

    "9S": (0, 90), "TS": (0, 100), "QS": (0, 110),
    "KS": (0, 120), "AS": (0, 130),
    "JC": (0, 135),   # left bower  -- trump, above AS
    "JS": (0, 140),   # right bower -- trump, top card in the game
}

NAME_OF = {v: k for k, v in CARD.items()}

DIAMONDS, TRUMP, HEARTS, CLUBS = 0, 1, 2, 3


def hand(*names):
    """('JS', 'JC', ...) -> (n, 2) int64 array of cards."""
    return np.array([CARD[n] for n in names], dtype=np.int64)


def deal(*hands):
    """Four hands of names -> a (4, 5, 2) int64 deal."""
    if len(hands) != 4:
        raise ValueError("a deal needs exactly 4 hands, got %d" % len(hands))
    return np.array([[CARD[n] for n in h] for h in hands], dtype=np.int64)


def name_of(card):
    return NAME_OF[(int(card[0]), int(card[1]))]


# ------------------------------------------------------------ rule oracles

def suit_strength(card):
    """A second, independent reading of the card encoding."""
    x, y = int(card[0]), int(card[1])
    if x > 0:
        return DIAMONDS, x
    if x < 0:
        return HEARTS, -x
    if y > 0:
        return TRUMP, y
    return CLUBS, -y


def suit_of(card):
    return suit_strength(card)[0]


def winner_of(played):
    """played: [(suit, strength, player)] in play order. Highest trump, else
    highest card of the led suit."""
    trumps = [p for p in played if p[0] == TRUMP]
    if trumps:
        return max(trumps, key=lambda p: p[1])[2]
    led = played[0][0]
    return max((p for p in played if p[0] == led), key=lambda p: p[1])[2]


def score_from_tricks(caller_tricks, n_tricks=5):
    """+2 march, +1 win, -2 euchred -- from the calling team's perspective."""
    if caller_tricks == n_tricks:
        return 2
    if caller_tricks >= 3:
        return 1
    return -2


class SearchTooLarge(Exception):
    """full_minimax hit its node budget. The hand is too branchy to brute force."""


def full_minimax(hands, starting_player, caller, node_limit=None):
    """
    Exhaustive minimax with no pruning and no forced-outcome cutoffs.

    This is the slow ground truth: it enumerates every legal continuation, so
    it cannot be wrong for the reason an alpha-beta search can.

    Cost is driven entirely by how often players can follow suit, since that is
    the only thing narrowing the branching factor, and it varies enormously.
    Measured over 15 random deals at ~1M nodes/s: 0.18 s at the fastest, 2.7 s
    median, and 55 s at the slowest -- a 300x spread on ordinary hands. The
    ceiling, where nobody can ever follow suit, is (5!)^4 = 207M play sequences,
    or roughly three and a half minutes.

    So pass a `node_limit` rather than trusting a hand to be cheap: a caller
    that picks hands by seed is relying on luck, not on a bound.

    Args:
        node_limit: raise SearchTooLarge once this many nodes have been visited.

    Returns (score, nodes_visited).

    Raises:
        SearchTooLarge: if node_limit is given and exceeded.
    """
    caller_team = caller % 2
    state = [[suit_strength(c) for c in player] for player in hands]
    n_tricks = len(state[0])
    nodes = [0]

    def rec(to_act, played, caller_tricks, trick_no):
        nodes[0] += 1
        if node_limit is not None and nodes[0] > node_limit:
            raise SearchTooLarge(
                "exhaustive minimax passed its %d-node budget" % node_limit)
        if len(played) == 4:
            w = winner_of(played)
            ct = caller_tricks + (1 if w % 2 == caller_team else 0)
            if trick_no + 1 == n_tricks:
                return score_from_tricks(ct, n_tricks)
            return rec(w, [], ct, trick_no + 1)

        held = state[to_act]
        if played:
            led = played[0][0]
            moves = [c for c in held if c[0] == led] or list(held)
        else:
            moves = list(held)

        values = []
        for c in moves:
            held.remove(c)
            values.append(
                rec((to_act + 1) % 4, played + [(c[0], c[1], to_act)],
                    caller_tricks, trick_no)
            )
            held.append(c)
        return max(values) if (to_act % 2) == caller_team else min(values)

    return rec(starting_player, [], 0, 0), nodes[0]


# Deals cheap enough to check against full_minimax, one per possible outcome.
#
# These are pinned rather than drawn from a seed on purpose. Exhaustive minimax
# cost swings ~300x across ordinary deals -- 0.18 s to 55 s measured -- so a
# seeded pick makes the suite's runtime a matter of luck, and one unlucky reseed
# turns a fast suite into a multi-minute one with nothing looking wrong. Pinning
# them puts the cost in plain sight instead.
#
# Node counts are as measured at ~1M nodes/s; together they run in about a third
# of a second. If you add one, measure it and record it here.
BRUTE_FORCEABLE = (
    # label,     hands,                       starting_player, caller, score, nodes
    ("euchred",
     (["KC", "QC", "AH", "JC", "9D"],
      ["JS", "QH", "AC", "TD", "KD"],
      ["TC", "9S", "TS", "KH", "JD"],
      ["KS", "JH", "AS", "9C", "QD"]), 1, 2, -2, 57_853),

    ("three tricks",
     (["JC", "KD", "QS", "KC", "JH"],
      ["9H", "AC", "TD", "AD", "JS"],
      ["TC", "9D", "9S", "JD", "QH"],
      ["AH", "QD", "AS", "QC", "9C"]), 3, 1, 1, 30_028),

    ("march",
     (["9D", "AH", "KC", "JD", "AS"],
      ["JC", "TD", "JS", "QH", "9S"],
      ["9H", "9C", "QD", "QC", "TS"],
      ["JH", "AD", "KS", "AC", "QS"]), 3, 1, 2, 241_248),
)


def brute_forceable_deals():
    """Yield (label, deal, starting_player, caller, expected_score)."""
    for label, hands, starting_player, caller, score, _nodes in BRUTE_FORCEABLE:
        yield label, deal(*hands), starting_player, caller, score


def legal_moves(held, led_suit):
    """The rules' own answer to 'what may this player play?'."""
    if led_suit is None:
        return list(held)
    following = [c for c in held if suit_of(c) == led_suit]
    return following if following else list(held)


# ------------------------------------------------------------ line replay

def replay_line(hands, starting_player, caller, score, ps, pv, pp, winners):
    """
    Replay a line returned by fast_search.solve_line and re-derive everything:
    seat order, that the card was still held, the follow-suit rule against the
    hand as it stood, the trick winner, and the final score.

    Returns a list of violations; empty means the line is sound.
    """
    from fast_search import _decode

    errs = []
    remaining = [[tuple(int(v) for v in c) for c in hands[p]] for p in range(4)]
    caller_team = caller % 2
    caller_tricks = 0
    n_tricks = hands.shape[1]

    for t in range(n_tricks):
        leader = starting_player if t == 0 else int(winners[t - 1])
        led_suit = None
        played = []

        for k in range(4):
            p = int(pp[t, k])
            if p != (leader + k) % 4:
                errs.append("trick %d seat %d: player %d played, expected %d"
                            % (t + 1, k, p, (leader + k) % 4))

            card = tuple(_decode(ps[t, k], pv[t, k]))
            if card not in remaining[p]:
                errs.append("trick %d: player %d played %s, which it does not hold"
                            % (t + 1, p, card))
            else:
                remaining[p].remove(card)

            s = int(ps[t, k])
            if k == 0:
                led_suit = s
            elif s != led_suit and any(suit_of(c) == led_suit for c in remaining[p]):
                errs.append("trick %d: player %d revoked (held the led suit)"
                            % (t + 1, p))
            played.append((s, int(pv[t, k]), p))

        w = winner_of(played)
        if w != int(winners[t]):
            errs.append("trick %d: winner is %d, solver said %d"
                        % (t + 1, w, winners[t]))
        if w % 2 == caller_team:
            caller_tricks += 1

    expected = score_from_tricks(caller_tricks, n_tricks)
    if expected != score:
        errs.append("solver returned %d but its own line takes %d tricks (-> %d)"
                    % (score, caller_tricks, expected))
    return errs


# ------------------------------------------------------------ deal checks

def deal_violations(game, deck):
    """A (4, 5, 2) deal must be 20 distinct cards, all drawn from `deck`."""
    errs = []
    if game.shape != (4, 5, 2):
        return ["deal is shaped %r, expected (4, 5, 2)" % (game.shape,)]

    flat = game.reshape(-1, 2)
    if len(np.unique(flat, axis=0)) != 20:
        errs.append("deal contains duplicate cards")

    known = {(int(c[0]), int(c[1])) for c in deck}
    for c in flat:
        if (int(c[0]), int(c[1])) not in known:
            errs.append("dealt %r, which is not in the deck" % (c.tolist(),))
    return errs
