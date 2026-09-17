# How many sampled worlds does a PIMC decision need?

`notes/settle_counts.py`, raw records in `notes/settle_counts.jsonl`.
**2,835 deals, 40,390 decisions**, racing off (`epsilon=None`) so exactly
`N_max` worlds are drawn and nothing is dropped. Six configurations (three
hands, dealer rotating). Median deal 44.6 s.

Measured at `93af317`, after the up-card stopped being a legal discard — so a
discard decision has 5 options, not 6.

## The headline table, and why half of it is not real

`N_max = 2000`:

| kind | n | mean | p50 | p75 | p95 | p99 | p99.9 | censored | mean (uncensored) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bid | 6,091 | 231.1 | 5 | 113 | 1552 | 1975 | **2000** | 10.2% | 80.5 |
| discard | 2,617 | 265.8 | 14 | 184 | 1698 | 1968 | **1999** | 11.3% | 97.7 |
| play | 31,682 | 132.3 | 1 | 22 | 1084 | 1931 | **1997** | 5.4% | 49.1 |
| **pooled** | 40,390 | **155.9** | 2 | **32** | 1284 | 1946 | **1999** | 6.5% | 56.6 |

**The mean and p75 are the numbers to use. p95 and above are artifacts of the
budget.**

Look at what happens when the budget changes:

| N_max | pooled p95 | pooled p99 | pooled p99.9 |
| --- | --- | --- | --- |
| 250 | 202 | 245 | **250** |
| 1,000 | 744 | 981 | **999** |
| 2,000 | 1,284 | 1,946 | **1,999** |

The p99.9 is N_max every time. It is not measuring the distribution, it is
reporting the budget back. p95 and p99 do the same thing more slowly.

## Why the tail cannot be measured this way — or at all

Two reasons, and the second is the real one.

**Censoring.** The settle point is defined against the pick at `N_max`, so
every decision has one by construction. That number is only meaningful if the
running pick stopped moving well before the end; the script counts a decision
as censored when its last flip lands after `N_max/2`. That is **6.5% pooled and
10–11% for bid and discard**. Anything above roughly the 93rd percentile
(pooled) or the 89th (bid) is therefore built on decisions whose true settle
point was never observed. The p95 row is already inside that zone.

**The tail is ties, and ties never settle.** This is the part worth keeping.
Break the decisions down by their final margin — the gap between the best and
second-best option after all 2,000 worlds — and ask what fraction never settled:

| kind | gap = 0 | 0 < gap ≤ 0.05 | 0.05–0.15 | gap > 0.15 |
| --- | --- | --- | --- | --- |
| bid | 100% | 49% | 5% | **0%** |
| discard | 6% | 32% | 0% | **0%** |
| play | 2% | 25% | 0% | **0%** |
| pooled | 2% | 29% | 1% | **0%** |

**No decision with a real margin ever failed to settle.** Every one that did
was a near-tie. And for a genuinely tied pair of options the running argmax
keeps flipping forever — the settle point is not large, it is *undefined*. The
distribution has an atom at infinity.

So there is no finite p99.9 to go and find. Raising `N_max` to 20,000 would
report p99.9 ≈ 20,000, and to 10⁶ would report ≈ 10⁶, and none of those would
be the answer to the question.

## What this says about `epsilon`

This is the measurement that justifies the band, and it justifies it more
sharply than the earlier 12-deal spot-check did.

- **You cannot buy the tail with budget.** The decisions that need more worlds
  are precisely the decisions where the options are worth the same, so more
  worlds buy a more confident answer to a question whose answer does not
  matter. An indifference band is not a shortcut around the tail — it is the
  only thing that addresses it at all.
- **`min_worlds = 24` is about right.** 72.9% of decisions pooled have settled
  by 24 worlds (play 75.7%, bid 65.2%, discard 57.3%). Below that the floor
  would be dropping options on genuinely thin evidence; far above it and most
  decisions are paying for confirmation.
- **Budget coverage**, if a band is not used:

| settled within | bid | discard | play | pooled |
| --- | --- | --- | --- | --- |
| ≤ 1 world | 37.1% | 25.5% | 54.0% | 49.6% |
| ≤ 24 | 65.2% | 57.3% | 75.7% | 72.9% |
| ≤ 100 | 74.4% | 69.9% | 84.5% | 82.0% |
| ≤ 400 | 82.6% | 81.0% | 90.5% | 88.7% |
| ≤ 800 | 88.1% | 86.0% | 93.5% | 92.2% |

Getting from 800 worlds to 2,000 buys the last ~8%, and that last 8% is made
almost entirely of ties.

- **Bidding really is the expensive half**, as CLAUDE.md says: mean 231 worlds
  against play's 132, and only 37.1% settled at one world against play's 54.0%.
  Discard is the worst of the three (mean 265.8), which is a fair bit of work
  for a decision with five options that are frequently interchangeable.

## Is the sample big enough, and what budget follows

Yes, comfortably — for the statistic that matters.

| kind | n | mean settle | standard error | 95% CI |
| --- | --- | --- | --- | --- |
| bid | 6,091 | 231.1 | 6.30 | ± 12.3 |
| discard | 2,617 | 265.8 | 10.16 | ± 19.9 |
| play | 31,682 | 132.3 | 2.15 | ± 4.2 |
| pooled | 40,390 | 155.9 | 2.06 | ± 4.0 |

Every mean is pinned to better than ±5%. More deals would not move them.

**But the mean is not what justifies a budget** — it is inflated by the ties,
which sit at `N_max` and would sit at any other `N_max` too. The number that
justifies a budget is how often the leader *stays* the leader among decisions
that have a real margin. "Leader stays leader" is exactly the settle criterion:
the pick after `B` worlds equals the pick after 2,000.

**Share of decisions whose pick at `B` equals the final pick:**

| kind | margin band | n | B=24 | B=100 | B=156 | B=266 | B=800 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bid | **real (>0.15)** | 4,087 | 90.6% | 98.4% | **99.4%** | 99.8% | 100% |
| bid | slim (0.05–0.15) | 827 | 22.5% | 43.4% | 52.6% | 64.1% | 92.3% |
| bid | near-tie (≤0.05) | 1,173 | 7.1% | 13.0% | 16.8% | 22.3% | 43.8% |
| discard | **real (>0.15)** | 867 | 90.5% | 99.4% | **99.8%** | 100% | 100% |
| discard | slim | 573 | 55.0% | 75.6% | 83.4% | 91.3% | 99.5% |
| discard | near-tie | 866 | 22.2% | 32.7% | 36.8% | 43.9% | 60.9% |
| play | **real (>0.15)** | 9,199 | 89.7% | 99.0% | **99.7%** | 99.9% | 100% |
| play | slim | 3,660 | 63.4% | 81.9% | 87.7% | 92.7% | 98.8% |
| play | near-tie | 5,985 | 32.6% | 44.1% | 48.7% | 55.1% | 71.1% |

Read the bold row. **A budget of ~150 holds 99.4–99.8% of the decisions that
have a real margin.** The near-tie rows keep drifting at every budget, which is
the expected behaviour and not a defect: those are the decisions where the
options are worth the same, so whichever one the drift lands on costs nothing.

Note also that going from 24 to 156 buys about 9 percentage points of
real-margin decisions, and going from 156 to 800 buys 0.3–0.6. The curve is
almost entirely flat past ~150 for anything that matters.

## The defaults that follow

`hand_ev.py` now defaults each kind to its own measured mean:

```python
PLAYER_EVAL_SIMS = 132          # a card:    mean 132.3 +/- 4.2
BID_EVAL_SIMS = 231             # a bid:     mean 231.1 +/- 12.3
DISCARD_EVAL_SIMS = 266         # a discard: mean 265.8 +/- 19.9
```

`PIMCPlayer` gained a `discard_samples` knob for the third of those; it
defaults to `bid_samples`, so `pimc_sweep.py`, `pimc_example.py` and every
measurement already in CLAUDE.md are unchanged. Only `hand_ev.py` adopts the
new numbers.

Two things worth being clear about:

- The old `hand_ev` default was **10** worlds per card decision, which is below
  even the 24-world floor at which `_race` will drop anything — so the old
  default could never race at all, and was running a tenth of the evidence a
  card decision wants. The new defaults are a genuine quality change. Measured
  cost on the pinned hand: **1.052 s/deal at `epsilon 0.05`, 0.481 at
  `epsilon 0.40`**, against 0.39 s/deal for the old 10-sim default. So roughly
  2.7x for 13x the worlds per card decision — the band absorbs most of it.
- Using the mean is a conservative choice rather than a tuned one. The
  real-margin coverage table says ~150 would do for all three kinds; the
  per-kind means (132/231/266) all sit at or above that, so bidding and
  discarding are being given more than the evidence demands. That is the safe
  direction, and cheap, since `epsilon` stops most of those decisions early
  anyway.

## What this is not

- **Hindsight, so a lower bound.** The reference answer is read off the end of
  the run. A real stopping rule does not know what it is converging to, so it
  must always do worse than these numbers. Read them as "no stopping rule can
  beat this", never as "this budget is enough".
- **Not a statement about EV.** A decision settling late is not a decision being
  got wrong — by the margin table, late-settling decisions are the ones where
  either answer is worth the same.
- **Sample sizes are uneven.** Play has 31,682 decisions, discard only 2,617,
  and the six configurations are not equally weighted (one contributes 1,420 of
  2,835 deals). The per-kind means are solid; fine structure within `discard`
  is thinner than it looks.
