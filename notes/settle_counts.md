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
