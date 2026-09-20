# The path to a front end

Written 2026-09-20, after the compiled engine landed. Everything below is
measured on this machine (6 physical cores, 12 threads, 15 GB) with the numba
cache warm.

The framing was settled in conversation and is not up for rediscovery:

- **a hand evaluator, not a bid advisor, and no game score.** "This hand plays
  for +0.9 if you order it", not "order it", and no running 10-point game.
- **absolute EV per action** -- order / order alone / pass -- each on the asking
  seat's own team's scale. Not a paired comparison between them.
- **measured over the deals the auction actually reached the seat on.** That is
  the `EV given you ordered` line `report()` already prints, taken over
  `Record.forced`. Not `--no-let-auction-play`, which prices the call on every
  deal and gives opposite advice about loners -- see "Why the population
  matters" below.
- **`pass_model="zero"`**, which is already the default.

## What one query costs

Three actions, walked auction, mean over the deals that reached the seat.
`TH AS AD KD JD`, `9H` up, seat 2, dealer 0:

| deals | wall | order | order alone | pass |
| ----- | ---- | ----- | ----------- | ---- |
|    100 |  1.1s | +/- 0.226 | +/- 0.455 | +/- 0.299 |
|    250 |  2.2s | +/- 0.137 | +/- 0.274 | +/- 0.182 |
|    500 |  3.9s | +/- 0.101 | +/- 0.192 | +/- 0.135 |
|  1,000 |  7.5s | +/- 0.078 | +/- 0.139 | +/- 0.100 |
|  2,500 | 16.7s | +/- 0.049 | +/- 0.088 | +/- 0.062 |
|  5,000 | 30.3s | +/- 0.034 | +/- 0.062 | +/- 0.044 |
| 10,000 | 57.2s | +/- 0.024 | +/- 0.044 | +/- 0.031 |

Linear, about **5.7 ms a deal** for all three actions together. So the UI can
promise a wait time honestly, and the arithmetic is `deals * 0.0057` seconds.

**Going alone needs about four times the deals for the same precision.** Its
outcomes run -2 to +4 where a four-handed call runs -2 to +2, so its interval
is roughly double at any deal count. At 1,000 deals the order/alone gap on this
hand (0.08) is smaller than the alone interval (0.14) -- the query cannot tell
you which is better. At 5,000 it can. Two options, both legitimate:

1. show the intervals and let the user see it is too close to call;
2. spend more deals on `alone` than on the other two -- same estimator, same
   reported mean, just a bigger `n` where the variance is bigger.

Do not "fix" this by reporting a paired difference. That is a different number
and the user has said it is not the one they want.

## The other sim axis, and what it is worth

`--deals` is the outer loop and the only thing the error bar is on.
`--player-eval-sims` / `--bid-eval-sims` / `--discard-eval-sims` are how many
worlds each player imagines per decision; they move the mean rather than
narrowing it.

Measured at **10,000 deals**, where the interval is +/- 0.024 and a real shift
of 0.05 would be visible:

| play / bid / discard | wall | EV given you ordered | vs default |
| -------------------- | ---- | -------------------- | ---------- |
|   33 /  58 /   66 |  9.7s | +0.909 +/- 0.024 | +0.014 |
|   66 / 115 /  133 | 15.1s | +0.920 +/- 0.023 | +0.025 |
|  132 / 231 /  266 | 22.5s | +0.895 +/- 0.024 | (default) |
|  264 / 462 /  532 | 31.0s | +0.906 +/- 0.024 | +0.011 |
|  528 / 924 / 1064 | 43.9s | +0.896 +/- 0.024 | +0.001 |

**Sixteen times the worlds moves the answer by less than the error bar, and
costs 4.5x the time.** There is no trend -- the values bounce, which is what
independent sampling noise looks like. The band is not the explanation: at
`epsilon 0.05` the stopping rule caps a decision around 2,600 worlds, well
above 528.

This is **one hand, one epsilon, one action.** It is not a flatness claim about
PIMC in general, which CLAUDE.md is right to keep warning about. What it does
support is a UI decision: **expose `deals` prominently and leave the eval
counts on their measured defaults**, with the eval counts available to anyone
who wants them but not presented as the precision dial. Spending a user's
seconds on the axis that demonstrably does not move the answer is the wrong
trade.

## Why the population matters

`--assume X` walked, over the deals that reached the seat, against the same
call pinned on every deal. 10,000 deals:

| | pinned every deal | reached you (89%) |
| --- | --- | --- |
| order | +0.791 +/- 0.024 | +0.895 +/- 0.024 |
| order alone | +0.700 +/- 0.043 | +0.982 +/- 0.044 |

**Pinned says do not go alone. Reached-you says do.** The 11% of deals the
auction never reaches are the ones an opponent opened -- so the ones where an
opponent has the cards -- and pricing a loner on those costs 0.28 against a
four-handed call's 0.10.

Reached-you is the number a player faces, so it is the one to report.
`--no-let-auction-play` keeps its existing job: comparing seats on equal
footing, where a late seat's walked number is contaminated by how often it is
preempted.

## The loner is not broken

Worth recording, because it was the thing that looked wrong and was not. The
call pinned every deal, 10,000 deals, no God Mode involved:

| hand | | march | 3-4 | euchred | EV |
| ---- | --- | ----- | --- | ------- | -- |
| `JS JC AS KS QS` | four-handed | 100.0% | 0.0% | 0.0% | +2.000 |
| both bowers, A K Q trump | alone | 100.0% | 0.0% | 0.0% | +4.000 |
| `JS JC AS 9H 9D` | four-handed | 30.8% | 69.2% | 0.0% | +1.308 |
| both bowers, ace, two rags | alone | **0.0%** | 100.0% | 0.0% | +1.000 |
| `JS AS KS 9H 9D` | four-handed | 17.9% | 62.8% | 19.3% | +0.601 |
| right, A K, two rags | alone | **0.0%** | 64.0% | 36.0% | -0.079 |

The middle row is the convincing one: three top trumps take three tricks and
two rags cannot take a fourth, so the engine marches **zero** times alone and
prices it at exactly +1.000. It reads the hand; it is not indiscriminately
loner-happy. Going alone only pays when you march, which makes the loner EV
checkable this way without appealing to God Mode at all.

## A perfect-information single-deal solve

Asked for as a feature. **It already exists, in pieces, and costs 59 ms warm.**

| piece | what it gives | warm |
| ----- | ------------- | ---- |
| `bidding.solve_bidding(deal, allow_loners=True)` | the auction God Mode would run, seat by seat, and its value | 28.9 ms |
| `bidding.first_bid_options(deal)` | `{pass, order, order alone}` for the **eldest** seat | 27.7 ms |
| `fast_search.solve_line(...)` | the optimal line: every card, who played it, who won each trick | 2.2 ms |

Sample output, dealer 3, `9H` up:

```
seat 0 passes / seat 1 passes / seat 2 passes / seat 3 orders up hearts
  -> seat 3 ordered up hearts (dealer pitched 9C) -> -1 to team 0
  AS KS QS AH  played by [0, 1, 2, 3]  -> seat 3
  9H JH KH JD  played by [3, 0, 1, 2]  -> seat 0
  AD TC 9D QH  played by [0, 1, 2, 3]  -> seat 3
  KC JS AC 9S  played by [3, 0, 1, 2]  -> seat 1
  TH TS JC KD  played by [1, 2, 3, 0]  -> seat 1
```

Two gaps, both small:

- **`solve_line` returns encoded `(suit, strength)` planes and the decoder is
  private** -- `fast_search._decode`, plus `rotation.card_from_engine`. A
  caller outside the module cannot turn a line into card names without
  reaching into a private function. Needs a public helper; four lines.
- **`first_bid_options` is eldest-only.** For an arbitrary seat the pieces are
  `order_up(deal, seat, alone)`, `name_suit(deal, seat, suit, alone)` and
  `rest_of_auction(...)` for the pass branch -- assembling them is the work,
  not computing them.

Note this runs on the **Python** path (`bidding.py` + `fast_search`), not the
compiled one. `fastsim.solve_bidding` is far quicker but returns no line, and
`bitcore` has no line recovery at all. At 59 ms that does not matter, and it is
worth *not* building compiled line recovery for a feature that is already
interactive.

## The process

Boot, with the numba cache warm on disk:

```
  import numpy                   0.13s
  import hand_ev                 0.60s   (pulls numba + the tables)
  first run_fast call            1.18s   (numba loads the cached code)
  ---------------------------------------
  boot, all in                   1.92s
  a 50-deal query after that     0.24s
  a 1000-deal query              2.80s
```

So: a **long-lived process**, because 1.9 s at boot is fine and paying it per
request is not. A small local server, or an embedded Python process behind
whatever the UI is.

Three things it has to handle:

- **A cold numba cache costs ~80 s**, not 1.9. That happens after any edit to
  `bitcore.py` or `fastsim.py`. The server should make one throwaway query at
  boot and refuse traffic until it returns, with a visible "building" state --
  otherwise the first user sees a 90-second hang.
- **Serialise queries.** One query already uses every core through numba's
  `prange`; two at once oversubscribe the thread pool and both get slower. A
  lock and a queue, not a thread pool.
- **The transposition table is 134 MB at 1,000 deals** and is allocated per
  call inside `run_fast`. Hoisting it to live for the process is an obvious
  win and needs a small change to `run_fast`'s signature -- worth doing, since
  a 1,000-deal query spends ~0.1 s of its 2.8 s zeroing it.

## Proposed API

Two functions, because there are genuinely two products.

```python
evaluate(hand, up_card, seat, dealer, deals=1000, **knobs) -> Evaluation
```

Three `run_fast` calls behind one entry point. Returns, per action: the mean
over the deals that reached the seat, its 95% interval, how often the auction
reached the seat, and the trick breakdown (march / 3-4 / euchred) -- that last
one because it is what makes a loner number checkable by eye.

```python
solve(hands, up_card, dealer, seat=None) -> Solution
```

The perfect-information answer: the auction God Mode runs, its value, the
optimal line of play, and per-option values for `seat`. Assembles the three
existing pieces plus the decoder gap above.

Deliberately **not** in the first cut: game score, defending alone, heuristic
opponents (roadmap section 2), and anything that needs `bitcore` to recover a
line.

## Suggested order

1. **`evaluate` and `solve` as plain Python, with tests.** No UI, no server.
   This is the piece that has to exist whatever the front end turns out to be,
   and it makes the whole thing drivable from a REPL. Includes the public
   decoder and the arbitrary-seat generalisation of `first_bid_options`.
2. **Hoist the transposition table** out of `run_fast` so a process can keep
   one.
3. **Decide the transport** -- and it is worth asking whether it needs one. If
   the UI is a notebook or a local script, (1) is already the whole product.
4. **The UI**, once there is something to point it at.

The open question for step 3 is who this is for. A notebook is free; a web UI
is a different project with a different shape, and the answer changes what (1)
should return -- JSON-shaped dataclasses versus whatever reads best in a REPL.
