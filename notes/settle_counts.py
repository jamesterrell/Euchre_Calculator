"""
How many sampled worlds does a PIMC decision actually need?

A PIMC decision averages every option over N worlds and takes the best. The
average is not the answer -- the argmax is -- so the question that prices
`--player-eval-sims` is how many worlds the argmax needs before it stops
moving. `players._race` already leans on that being small; this measures the
distribution, and in particular its tail.

Method. Every decision is run to a fixed budget with racing OFF
(`epsilon=None`, so exactly `budget` worlds are drawn and nothing is dropped),
recording each option's value in each world. Then, offline:

  1. the pick after all `N_max` worlds is the reference answer;
  2. the settle point is the smallest `n` such that the running pick over the
     first `m` worlds equals the reference for every `m >= n`.

"Pick" is `players._pick`'s own rule, not a bare argmax: highest running mean,
ties to the first option for a bid and to the cheapest card for a discard or a
play. Running sums are integers, so the comparisons are exact.

**This is hindsight.** The reference answer is read off the end of the run, so
the settle point is a lower bound on what any online rule could achieve -- an
online rule does not know the answer it is converging to. Read the numbers as
"no stopping rule can do better than this", not as "this budget is enough".

**Right-censoring.** Every decision has a settle point <= N_max by
construction, because the reference is defined at N_max. That number is only
meaningful when the running pick stopped moving well before the end. A decision
whose last flip is near N_max would probably have flipped again given more
worlds, and its true settle point is unobservable. Those are counted separately
(last flip after N_max/2) rather than folded into the percentiles.

Nothing in the repo is modified: `players._race` and the three `PIMCPlayer`
decision methods are wrapped in this process only.

    python notes/settle_counts.py --nmax-play 20000 --nmax-bid 2000 \
        --deals 400 --workers 10 --minutes 240 --out notes/settle_counts.jsonl
    python notes/settle_counts.py --analyse notes/settle_counts.jsonl
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hand_ev                                              # noqa: E402
import players                                              # noqa: E402
import rotation as r                                        # noqa: E402
import table as t                                           # noqa: E402

BID, DISCARD, PLAY = "bid", "discard", "play"
KINDS = (BID, DISCARD, PLAY)


# ------------------------------------------------------------- instrumenting

_ORIG_RACE = players._race
_STATE = {"kind": None, "trump": None}
_OUT = []


def _argmax(sums, keys):
    """`players._pick`, re-expressed on running sums rather than means.

    Counts are equal across options when nothing is raced, so comparing sums
    is the same ordering as comparing means -- and exact, since they are ints.
    """
    best = max(sums)
    tied = [i for i, s in enumerate(sums) if s == best]
    if keys is not None and len(tied) > 1:
        return min(tied, key=lambda i: keys[i])
    return tied[0]


def _summarise(kind, n_options, rows, keys):
    """
    One decision, stored as the trace of its running pick.

    `changes` is [[world, option], ...] at every world where the running pick
    differs from the previous world's -- a run-length encoding of the whole
    sequence, and small, since most decisions change their mind a handful of
    times. Keeping the trace rather than a single settle number is what lets
    `settle_at(rec, budget)` re-derive the answer for any smaller budget, so
    "would a shorter run have said something different" is a re-analysis
    rather than a second sweep.
    """
    sums = [0] * n_options
    changes = []
    last = None
    for m, row in enumerate(rows, 1):
        for i in range(n_options):
            sums[i] += row[i]
        pick = _argmax(sums, keys)
        if pick != last:
            changes.append([m, pick])
            last = pick

    # The final margin: how far ahead the winner ended up, in points per
    # world. A decision that never settles should be one whose margin is
    # zero or nearly zero -- that is the claim `epsilon` rests on, so it is
    # recorded rather than assumed.
    ordered = sorted(sums, reverse=True)
    gap = (ordered[0] - ordered[1]) / len(rows) if n_options > 1 else 0.0
    return {"kind": kind, "opts": n_options, "n": len(rows),
            "gap": round(gap, 6), "changes": changes}


def settle_at(rec, budget=None):
    """
    (settle point, last flip) for this decision at `budget` worlds.

    The reference answer is the pick after `budget` worlds; the settle point is
    the first world after which the running pick never again disagrees with it.
    """
    n = rec["n"] if budget is None else min(budget, rec["n"])
    changes = [c for c in rec["changes"] if c[0] <= n]
    ref = changes[-1][1]
    last_flip = 0
    for i, (_, pick) in enumerate(changes):
        # this pick stands until the next change, or to the end of the budget
        end = changes[i + 1][0] - 1 if i + 1 < len(changes) else n
        if pick != ref:
            last_flip = end
    return last_flip + 1, last_flip


def _race_recording(options, draw, budget, epsilon=None,
                    min_worlds=players.MIN_WORLDS, z=players.Z,
                    guard=players.GUARD):
    kind = _STATE["kind"]
    if kind is None or len(options) < 2:
        # Nothing to settle -- one option, or a call from outside a decision.
        return _ORIG_RACE(options, draw, budget, epsilon, min_worlds, z, guard)

    rows = []

    def recording_draw(active):
        values = draw(active)
        rows.append(tuple(int(values[o]) for o in options))
        return values

    out = _ORIG_RACE(options, recording_draw, budget, epsilon, min_worlds,
                     z, guard)
    trump = _STATE["trump"]
    keys = (None if trump is None
            else [t.card_order(c, trump) for c in options])
    if rows:
        _OUT.append(_summarise(kind, len(options), rows, keys))
    return out


def _wrap(name, kind, use_trump):
    original = getattr(players.PIMCPlayer, name)

    def wrapper(self, turn):
        before = (_STATE["kind"], _STATE["trump"])
        _STATE["kind"] = kind
        _STATE["trump"] = turn.trump if use_trump else None
        try:
            return original(self, turn)
        finally:
            _STATE["kind"], _STATE["trump"] = before

    return wrapper


def instrument():
    """Wrap the three decision methods and `_race`, in this process only."""
    players._race = _race_recording
    players.PIMCPlayer.bid = _wrap("bid", BID, False)
    players.PIMCPlayer.discard = _wrap("discard", DISCARD, True)
    players.PIMCPlayer.play = _wrap("play", PLAY, True)


# ------------------------------------------------------------- the workload
#
# Three configurations, so the answer is not one hand at one seat. All of them
# run `hand_ev.play_one`, which is the workload the sample count is actually
# spent on, with its own defaults: pass model "zero", loners on, no stick.

HAND_A = "JS AS 9H 9D TC"        # hand_ev.py's own example; ordered up ~70%
HAND_C = "AH KH QC TD 9D"        # no trump with 9S up, so seat 0 passes and
UP = "9S"                        # the auction runs on into round two


def setups(nmax_play, nmax_bid, seed=0):
    """(label, Setup) pairs. Labels land in the checkpoint file."""
    def make(hand, dealer, s):
        return hand_ev.Setup(
            hand=tuple(r.parse_hand(hand)), up_card=r.parse_card(UP),
            seat=0, dealer=dealer, player_eval_sims=nmax_play,
            bid_eval_sims=nmax_bid, pass_model=players.PASS_ZERO,
            allow_loners=True, stick=False, seed=s, epsilon=None)

    out = [("A/dealer3", make(HAND_A, 3, seed))]
    out += [("B/dealer%d" % d, make(HAND_A, d, seed + 100 * (d + 1)))
            for d in (0, 1, 2)]
    out += [("C/dealer%d" % d, make(HAND_C, d, seed + 900 + 10 * d))
            for d in (3, 1)]
    return out


# A/dealer3 is the headline question, so it gets half the deals; the rotating
# dealer and the trumpless hand split the rest. The weights are how many jobs
# each configuration contributes per round of the job list.
WEIGHTS = {"A/dealer3": 6, "B/dealer0": 1, "B/dealer1": 1, "B/dealer2": 1,
           "C/dealer3": 2, "C/dealer1": 1}


def jobs(setup_list, deals, start=0):
    """
    Interleaved, so an early stop still leaves every configuration covered.

    `start` skips that many jobs, which is how a second run extends a
    checkpoint file rather than re-playing the deals already in it.
    """
    out = []
    seen = {label: 0 for label, _ in setup_list}
    deals += start
    while len(out) < deals:
        for label, setup in setup_list:
            for _ in range(WEIGHTS[label]):
                out.append((label, setup, seen[label]))
                seen[label] += 1
                if len(out) >= deals:
                    return out[start:]
    return out[start:]


def _init():
    instrument()


def _work(job):
    label, setup, i = job
    del _OUT[:]
    started = time.time()
    try:
        hand_ev.play_one(setup, i)
    except Exception as exc:                       # keep the sweep going
        return label, i, [], round(time.time() - started, 3), repr(exc)
    return label, i, list(_OUT), round(time.time() - started, 3), None


def collect(setup_list, deals, workers, out_path, minutes=None, quiet=False,
            start=0):
    """Run deals, appending one JSON line per deal as it finishes."""
    work = jobs(setup_list, deals, start)
    deadline = time.time() + minutes * 60 if minutes else None
    started = time.time()
    done = decisions = 0

    with open(out_path, "a", encoding="utf-8") as fh:
        def write(row):
            label, i, recs, secs, err = row
            fh.write(json.dumps({"label": label, "deal": i, "secs": secs,
                                 "error": err, "decisions": recs}) + "\n")
            fh.flush()
            return len(recs)

        if workers > 1:
            pool = mp.Pool(workers, initializer=_init)
            try:
                for row in pool.imap_unordered(_work, work, chunksize=1):
                    decisions += write(row)
                    done += 1
                    if not quiet:
                        _progress(done, len(work), decisions, started)
                    if deadline and time.time() > deadline:
                        print("  ... time limit reached, stopping",
                              file=sys.stderr)
                        break
            finally:
                pool.terminate()
                pool.join()
        else:
            _init()
            for job in work:
                decisions += write(_work(job))
                done += 1
                if not quiet:
                    _progress(done, len(work), decisions, started)
                if deadline and time.time() > deadline:
                    break

    return done, decisions


def _progress(done, total, decisions, started):
    elapsed = time.time() - started
    rate = done / elapsed if elapsed else 0
    print("  ... %d/%d deals, %d decisions, %.0fs elapsed, %.1fs/deal"
          % (done, total, decisions, elapsed, elapsed / done if done else 0),
          file=sys.stderr)


# ---------------------------------------------------------------- reading it


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values, q):
    """Nearest-rank percentile: the smallest observation at or above q."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, int(-(-q * len(ordered) // 100)))
    return ordered[min(k, len(ordered)) - 1]


CENSOR_FRACTION = 0.5   # last flip past half the budget: settle point unusable


BUDGETS = (1, 24, 100, 400, 800, 2000)


def by_kind(rows):
    out = {k: [] for k in KINDS}
    for row in rows:
        for rec in row["decisions"]:
            out[rec["kind"]].append(rec)
    out["pooled"] = [d for k in KINDS for d in out[k]]
    return out


def summarise(recs, budget=None, censor_fraction=CENSOR_FRACTION):
    """Statistics over one population of decisions, at `budget` worlds."""
    if not recs:
        return None
    pairs = [settle_at(d, budget) for d in recs]
    caps = [min(budget, d["n"]) if budget else d["n"] for d in recs]
    settles = [s for s, _ in pairs]
    censored = [i for i, (_, flip) in enumerate(pairs)
                if flip > censor_fraction * caps[i]]
    clean = [settles[i] for i in range(len(recs)) if i not in set(censored)]
    return {
        "n": len(recs),
        "nmax": sorted({c for c in caps}),
        "censored": len(censored),
        "censor_rate": len(censored) / len(recs),
        "mean": sum(settles) / len(recs),
        "mean_clean": (sum(clean) / len(clean)) if clean else None,
        "median": percentile(settles, 50),
        "p75": percentile(settles, 75),
        "p95": percentile(settles, 95),
        "p99": percentile(settles, 99),
        "p999": percentile(settles, 99.9),
        "max": max(settles),
        "within": {b: sum(1 for s in settles if s <= b) / len(recs)
                   for b in BUDGETS},
        "mean_opts": sum(d["opts"] for d in recs) / len(recs),
    }


def report(rows, budgets=(None, 1000, 250)):
    groups = by_kind(rows)
    deals = len({(row["label"], row["deal"]) for row in rows})
    errors = sum(1 for row in rows if row["error"])
    print("deals %d (%d errored), decisions %d"
          % (deals, errors, len(groups["pooled"])))
    labels = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
    print("by configuration: %s"
          % ", ".join("%s %d" % kv for kv in sorted(labels.items())))
    print("median deal time %.1fs"
          % percentile([row["secs"] for row in rows], 50))

    stats = {}
    for budget in budgets:
        print("\n== N_max %s ==" % ("as run" if budget is None else budget))
        head = ("kind", "n", "N_max", "mean", "p50", "p75", "p95", "p99",
                "p99.9", "max", "cens%", "mean*")
        print("%-8s %7s %9s %9s %6s %6s %7s %7s %7s %7s %7s %8s" % head)
        for kind in KINDS + ("pooled",):
            s = summarise(groups[kind], budget)
            if not s:
                continue
            stats[(kind, budget)] = s
            print("%-8s %7d %9s %9.1f %6d %6d %7d %7d %7d %7d %6.1f%% %8.1f"
                  % (kind, s["n"], "/".join(str(x) for x in s["nmax"]),
                     s["mean"], s["median"], s["p75"], s["p95"], s["p99"],
                     s["p999"], s["max"], 100 * s["censor_rate"],
                     s["mean_clean"] or 0.0))

    print("\nnever settled, by final margin (N_max as run)")
    print("%-8s %6s %10s %10s %10s %10s"
          % ("kind", "n", "gap=0", "0<g<=.05", ".05-.15", ">.15"))
    bands = ((0.0, 0.0), (1e-9, 0.05), (0.05, 0.15), (0.15, 9.9))
    for kind in KINDS + ("pooled",):
        recs = [d for d in groups[kind] if "gap" in d]
        if not recs:
            continue
        cells = []
        for lo, hi in bands:
            band = [d for d in recs
                    if (d["gap"] == 0 if hi == 0 else lo <= d["gap"] <= hi)]
            if not band:
                cells.append("       -- ")
                continue
            never = sum(1 for d in band
                        if settle_at(d)[1] > CENSOR_FRACTION * d["n"])
            cells.append("%5d %4.0f%%" % (len(band), 100 * never / len(band)))
        print("%-8s %6d %s" % (kind, len(recs), " ".join(cells)))
    print("  (each cell: decisions in that margin band, and what share of them"
          " never settled)")

    print("\nsettled within B worlds (N_max as run)")
    print("%-8s %6s %s" % ("kind", "opts",
                           " ".join("%8s" % ("<=%d" % b) for b in BUDGETS)))
    for kind in KINDS + ("pooled",):
        s = stats.get((kind, None))
        if not s:
            continue
        print("%-8s %6.2f %s"
              % (kind, s["mean_opts"],
                 " ".join("%7.1f%%" % (100 * s["within"][b])
                          for b in BUDGETS)))
    return stats


# --------------------------------------------------------------- the front


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--analyse", default=None,
                        help="read a checkpoint file and print the tables")
    parser.add_argument("--out", default=os.path.join(_ROOT, "notes",
                                                      "settle_counts.jsonl"))
    parser.add_argument("--nmax-play", type=int, default=20000)
    parser.add_argument("--nmax-bid", type=int, default=2000)
    parser.add_argument("--deals", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=None,
                        help="stop collecting after this long")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many jobs, to extend a checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.analyse:
        return report(load(args.analyse))

    print("settle counts: N_max play %d, bid/discard %d, %d deals, %d workers"
          % (args.nmax_play, args.nmax_bid, args.deals, args.workers))
    started = time.time()
    done, decisions = collect(setups(args.nmax_play, args.nmax_bid, args.seed),
                              args.deals, args.workers, args.out,
                              minutes=args.minutes, quiet=args.quiet,
                              start=args.start)
    print("%d deals, %d decisions, %.0fs -> %s"
          % (done, decisions, time.time() - started, args.out))
    return report(load(args.out))


if __name__ == "__main__":
    main()
