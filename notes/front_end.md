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

| deals | first query | after | order | order alone | pass |
| ----- | ----------- | ----- | ----- | ----------- | ---- |
|    250 |  2.9s |  0.9s | +/- 0.137 | +/- 0.274 | +/- 0.182 |
|    500 |  5.5s |  2.0s | +/- 0.101 | +/- 0.192 | +/- 0.135 |
|  1,000 | 10.4s |  4.7s | +/- 0.078 | +/- 0.139 | +/- 0.100 |
|  2,000 | 19.6s | 11.8s | +/- 0.055 | +/- 0.098 | +/- 0.070 |
|  4,000 | 38.2s | 27.9s | +/- 0.039 | +/- 0.069 | +/- 0.050 |
| 10,000 |     - | 57.2s | +/- 0.024 | +/- 0.044 | +/- 0.031 |

Near enough linear to quote a wait from, at about **6 ms a deal** for all three
actions -- but **the first query of a process costs two to three times that**.
The transposition table starts empty and is never cleared, so one query pays to
fill it and every query after inherits the work. `api.SECONDS_PER_DEAL` and
`SECONDS_PER_DEAL_FIRST` are those two rates, `Engine.status` reports how many
queries have run, and the page quotes whichever applies.

That warming effect is also what made the first round of these measurements
incoherent -- runs got faster through a session and "no progress" appeared to
be slower than "with progress". Measure a fresh table and a warm one
separately, or the numbers move under you.

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

## Order of work

1. ~~**`evaluate` and `solve` as plain Python, with tests.**~~ **Done** --
   `api.py`, `tests/test_api.py`, 20 tests. It is a web app, so the results are
   frozen dataclasses with `as_dict()` returning plain JSON: cards as `"JS"`,
   suits as `"spades"`, nothing a `json.dumps` will choke on. Measured on this
   machine: `solve` 98 ms warm, `evaluate` 7.3 s for 1,000 deals on ten
   threads. The two gaps named above are closed -- `fast_search.decode_card` is
   public, and `bidding.bid_options(deal, seat)` asks the round-one question
   from any seat.
2. ~~**Hoist the transposition table.**~~ **Done** -- `hand_ev.new_table`, and
   `run_fast(..., table=)`. `api.Engine` holds one for the life of a process.
   Worth more than the allocation it saves: the table is never cleared, so
   queries warm each other. On the worked example at 1,000 deals, 7.3 s with a
   fresh table each call against 5.0 s with a shared one by the third query.
3. ~~**The server.**~~ **Done** -- `server.py`, standard library only, and
   `static/index.html` as one file with no build step. It warms in a
   background thread and answers 503 until ready, serialises queries behind
   `Engine`'s lock, and closes the connection on every refusal. That last one
   was a real bug: HTTP/1.1 reuses connections, and a request whose body was
   never read leaves bytes in the socket that the next request gets parsed out
   of. `tests/test_server.py` found it; curl by hand had not.
4. ~~**The UI.**~~ **Done** -- three tabs: evaluate, solve, and an About page
   written for a Euchre player rather than a programmer (615 words, no
   jargon). Cards are picked by dragging or clicking from a deck laid out a
   suit to a row, not typed. Checkboxes choose which actions to price, all
   three on by default. The deals slider quotes its own wait and says when the
   first query will be slower. Progress is **real**, not a guess: the page
   polls `/api/health` for the engine's own fraction and the action it is
   pricing, and rotates a line of nonsense over the top of it.

   Reporting progress costs something, and getting it wrong costs a lot. The
   sweep is split into blocks and every block ends at a barrier, so a block has
   to be big enough to keep every thread busy: 1,000 deals over 40 chunks in 20
   blocks is barely one deal per chunk per block, and it ran at 13.5s against
   5.6s. `DEALS_PER_CHUNK_PER_BLOCK` is the floor that fixes it; the remaining
   cost is under 10%.

Nothing in (2), (3) or (4) changed `api.py`'s shape, which was the point of
doing it first.

## What is still not here

- **Game score.** `Deal` is one hand; there is no running 10-point game, so
  no "we are at 9, do I order anything?". Roadmap section 1's last unchecked
  box, and probably the thing a Euchre player asks for first.
- **Heuristic opponents.** The table is four PIMC players. Roadmap section 2.
- **Defending alone.**
- **Concurrency.** Queries serialise, which is right for one user on one
  machine and wrong for anything else. So is binding to localhost with no
  authentication.

## Notes for whoever writes the server

- `evaluate(..., workers=N)` sets numba's thread count for the process. Pick it
  once at boot rather than per request.
- `deals` is the only knob worth putting in front of a user, and the wait is
  `deals * 0.0057` seconds for all three actions. Quote it before running.
- `Evaluation.best()` exists and is **not advice** -- at 1,000 deals the
  order/alone gap on the worked example is smaller than the alone interval.
  Render the intervals or the UI will overstate what it knows.
- A passed-out `Solution` has no tricks. God Mode essentially never passes out
  -- 0 of 1600 measured -- but the shape has to survive it, and
  `test_a_passed_out_solution_still_serialises` pins that.
