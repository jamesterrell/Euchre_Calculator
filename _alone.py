"""Does the PIMC engine over-value going alone, against God Mode on the same deals?

Both engines price the *same pinned call* on the *same layouts*:
  PIMC     -- hand_ev with --assume order / order-alone --no-let-auction-play
  God Mode -- fastsim.order_up_value on the deals that engine generated
"""
import math
import numpy as np
import bitcore, fastsim as F, hand_ev as h, players, rotation as r

DEALS = 10000
CASES = [
    ("TH AS AD KD JD", "9H", 2, 0, "the hand that flagged it"),
    ("JS AS 9H 9D TC", "9S", 0, 3, "the README hand"),
    ("JS JC AS KS QS", "9S", 0, 3, "a laydown loner"),
    ("9C TC 9D TD 9H", "9S", 0, 3, "garbage"),
]

def cid(c): return c.suit * 6 + (c.rank - r.NINE)

def pimc(hand, up, seat, dealer, alone):
    s = h.Setup(hand=tuple(r.parse_hand(hand)), up_card=r.parse_card(up),
                seat=seat, dealer=dealer,
                player_eval_sims=players.RESEARCHED_PLAY,
                bid_eval_sims=players.RESEARCHED_BID,
                discard_eval_sims=players.RESEARCHED_DISCARD,
                pass_model=players.PASS_ZERO, allow_loners=True, stick=False,
                seed=0, epsilon=0.05,
                assume=h.ASSUME_ALONE if alone else h.ASSUME_ORDER,
                let_auction_play=False)
    recs, _, _ = h.run_fast(s, DEALS, workers=10)
    return np.array([x.value for x in recs], dtype=float)

def godmode(hand, up, seat, dealer, alone):
    """The same deals, priced in God Mode."""
    pin = 0
    for c in r.parse_hand(hand): pin |= 1 << cid(c)
    upc = cid(r.parse_card(up))
    tt = bitcore.new_tt(22); mask = bitcore.tt_mask(tt)
    stk = bitcore.new_stack(); nodes = np.zeros(F.COUNTERS, dtype=np.int64)
    rng = np.zeros(1, dtype=np.int64); hands = np.zeros(4, dtype=np.int64)
    out = np.empty(DEALS)
    for i in range(DEALS):
        F.seed_stream(rng, 0 + i)                 # the deal stream run_deals uses
        turned, _ = F.deal_around(pin, seat, upc, dealer, rng, hands)
        v = F.order_up_value(hands, turned, dealer, seat, 1 if alone else 0,
                             tt, mask, nodes, stk)
        out[i] = v if seat % 2 == 0 else -v       # onto the asking seat's scale
    return out

def ci(v): return 1.96 * v.std(ddof=1) / math.sqrt(len(v))

print("%d deals a cell, the call pinned every deal (--no-let-auction-play).\n" % DEALS)
print("  %-16s %-8s %-17s %-17s %s" % ("hand", "engine", "order", "order alone",
                                       "alone - order (paired)"))
for hand, up, seat, dealer, note in CASES:
    print("  %-16s %-8s %-17s %-17s %s"
          % (hand + " /" + up, "", "", "", "<- " + note))
    for name, fn in (("PIMC", pimc), ("God Mode", godmode)):
        o = fn(hand, up, seat, dealer, False)
        a = fn(hand, up, seat, dealer, True)
        d = a - o
        print("  %-16s %-8s %+.3f +/- %.3f  %+.3f +/- %.3f  %+.3f +/- %.3f"
              % ("", name, o.mean(), ci(o), a.mean(), ci(a), d.mean(), ci(d)))
    print()
