<!-- Moved out of CLAUDE.md on 2026-10-16 so it loads on demand rather than
     in every session. Content is verbatim; the sample counts it was taken at
     are superseded -- see the caveat below and CLAUDE.md's "Commands". -->

# What honest players actually do

Measured by `pimc_sweep.py` over 60 deals at 20 play samples and 10 bid samples
per decision, loners allowed, dealer rotating. Sampling error is large at this
size -- these are shapes, not constants.

|                            | PIMC sim     | God Mode |
| -------------------------- | ------------ | -------- |
| passed out                 |  0%          |  0%      |
| called alone               | 10%          |  1.7%    |
| ordered up in round one    | 57/60        | 41/60    |
| named a suit in round two  |  3/60        | 19/60    |
| euchred                    | 43% of calls | 10%      |
| marched                    | 10% of calls | 28%      |
| mean tricks to the caller  | 2.75         | 3.55     |
| mean points to the caller  | -0.10        | +1.02    |

The two auctions land on the same trump suit 67% of the time and the same
caller 55% of the time.

**Seeing the other hands is worth about 1.26 points a deal.** `--head-to-head`
puts the PIMC sim on one team and God Mode on the other, and plays each deal
twice with the teams swapped so seat and dealer advantages cancel exactly rather
than statistically. Over 50 deals: **-1.26 +/- 0.36 points per deal** to the
sim. A euchre is worth 2, for scale.

**The sim over-calls.** 43% of its contracts are euchred against God Mode's
10%. How much of that is the optimizer's curse -- "take the best of several
noisy averages" -- rather than the pass model below is **not settled**, and how
far the sample count per decision has to go before a PIMC player's bidding
stops moving is unmeasured. Do not assume it is flat.

**Most of the over-calling is the pass model.** Same 40 deals with
`pass_model="zero"`: the euchre rate falls to 20-28% and the average call goes
from -0.17 to +0.65 points. Pricing a pass by running the rest of the auction in
God Mode makes declining look worse than it is, because God Mode essentially
always finds a call -- so the pass branch nearly always reads "an opponent ends
up calling this", and never "it comes back around to me".

**But the better-behaved bidder is not a better player, and this is a trap
worth knowing about.** Head to head against God Mode, teams swapped on every
deal: `"god"` scores -1.26 +/- 0.36 and `"zero"` scores -1.34 +/- 0.36 over the
same 50 deals. Indistinguishable. Mean points *per call* flattered `"zero"`
only because it is averaged over the deals a player chose to call, and silently
drops what the deals it passed on cost it. That is `bidding.py`'s "passing is
not free" turning up as a measurement trap rather than a bidding one -- do not
read a per-call average as a strength number. What `"zero"` reliably is, is
about 4x faster.

**Two more pass models were tried, and both are worse. The attempt is worth
recording because of what it rules out.**

Both target the same flaw: `"god"` prices a pass at whatever the rest of the
auction does, and inside a sampled world the other seats can see the hand this
player is hiding, so the pass branch reads "an opponent calls this and makes
it" far more often than a real table would. A seat with a bad hand then makes a
desperate call because declining looked worse.

- `"floor"` clamps a pass at zero **per sampled world**: `sum(max(g_w, 0))`.
- `"guard"` is `"god"` plus one **decision-level** override: if every call is
  negative once all the worlds are averaged, pass; otherwise price the pass
  exactly as `"god"` does.

They are not the same rule -- by Jensen the per-world clamp is systematically
kinder to passing -- but they behave almost identically, which is the first
thing the measurement settled.

Profile over 60 deals, 20 play / 10 bid samples:

|                           | `zero` | `god` | `floor` | `guard` |
| ------------------------- | ------ | ----- | ------- | ------- |
| euchred                   | 11.7%  | 36.7% | 15.3%   | **21.7%** |
| ordered up in round one   | 58     | 54    | 33      | 37      |
| named a suit in round two |  2     |  6    | 26      | 23      |
| called alone              |  5.0%  |  8.3% | 18.3%   | 18.3%   |
| passed out                |  0     |  0    | **1**   |  0      |
| mean tricks to the caller | 3.57   | 2.85  | 3.54    | 3.40    |

Both produce a far more realistic-looking auction than `"god"` -- round-one
orders roughly halve and round-two calls rise five-fold, because a seat that is
not frightened of declining turns the up-card down and names a better suit.
`"floor"` produced **the first pass-out ever seen in a `pimc_sweep`**.

**And both are significantly weaker players.** Head to head against God Mode,
teams swapped on every deal, **250 deals each**:

| pass model | margin per deal |
| ---------- | --------------- |
| `zero`     | **-1.064 +/- 0.152** |
| `god`      | **-1.112 +/- 0.151** |
| `guard`    | -1.388 +/- 0.167 |
| `floor`    | -1.468 +/- 0.164 |

`zero` and `god` are indistinguishable (0.048 apart), which reproduces the
50-deal finding above at five times the sample. Both new models lose to both
old ones by about 0.3 points a deal, z ~ 2.4-2.8. `guard` and `floor` are
indistinguishable from each other (0.080 apart, z ~ 0.7), so **the per-world
versus decision-level distinction accounts for almost none of the damage** --
worth knowing, because it was the obvious hypothesis and it is wrong.

**What actually costs the points.** `"guard"` differs from `"god"` in exactly
one situation: when every call is negative and the least-bad call is still
better than `"god"`'s estimate of passing, `"god"` takes that call and
`"guard"` passes instead. That single change is the whole -0.276. So taking the
**least-bad losing call beats passing**, and the intuition that a bad hand
should be thrown in is simply wrong here. It is `bidding.py`'s "passing is not
free" again, now measured on the PIMC side: if you pass, somebody else calls
and scores, and `"god"`'s pessimism about that is closer to right than a floor
at zero is.

The euchre rate is not a strength metric. Across four models it is
**uncorrelated** with head-to-head margin -- `zero` has the best euchre rate
and the best margin, but `floor` has the second-best euchre rate and the worst
margin. Judge a bidder head to head or not at all.

**Nothing ever passes out, under either of the two original models** -- 0 of 60
in the main sweep and 0 of 40 in all five diagnostic configurations. (`"floor"`,
added later and covered above, does pass hands in.) That was the one prediction in
this file that did not come true, and the pass model does not explain it:
`"zero"` prices a pass at exactly nothing and still never throws a hand in. The
reason is the number of chances. Eight seats bid in turn, and each round-two
seat is choosing among three suits -- six options once loners are on. Somebody
almost always finds something that looks positive, especially since the sim's
estimates are optimistic to begin with. Getting a table to pass a deal out
needs a model of what the *other* seats will do with it, which is exactly what
neither pass model has. That is the next thing worth building.

**Amended: it passes out about 0.2% of the time.** The sweeps above were too
small to see it. A 10,000-deal `hand_ev.py` run on `JS AS 9H 9D TC` with `9S`
up (pass model `"zero"`, loners on, epsilon 0.40) passed out **21 of 10,000**
deals -- at which rate 60 deals predicts 0.1 pass-outs and 40 predicts 0.08, so
seeing none of them was never evidence of never. The shape of the claim
survives and the absolute version of it does not: passing out is rare rather
than impossible, and it took a sweep three orders of magnitude bigger than the
ones above to tell the difference. Worth re-reading as a caution about every
other 0-of-60 in this file.

**Loners move the way real tables move** -- 1.7% to 10%. Some of that is honest
optimism about hands that might run. Some is mechanical: averaging over sampled
worlds destroys the exact ties that made the God Mode auction decline a loner
worth no more than the same call four-handed.
