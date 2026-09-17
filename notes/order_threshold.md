# What a hand needs before ordering up is not a losing bid

`notes/order_threshold.py`, 6,000 deals, God Mode, every seat's order-up valued
on its own team's scale. Two numbers per bucket:

- **order** — what the seat's team scores if it orders up.
- **gain** — ordering *minus passing*, where passing is worth whatever the rest
  of the auction does. This is the real decision criterion and it is stricter:
  a call can score above zero and still be worse than declining.

## Headline: the premise of question 1 is false

> "Of course a player would never order a suit they have no trump in."

They would, and sometimes they should.

```
3227 zero-trump order-ups. mean order -1.133, mean gain -0.452
order >= 0 in 883 of 3227 (27.36%); gain >= 0 in 1754 (54.35%)
best zero-trump case seen: order +2, gain +0,
    hand 9S QS AH AC TS, up-card JD
```

Look at that hand. Trump is diamonds, so the up-card **JD is the right bower**,
and the seat ordering is the dealer's partner. Ordering with no trump at all
hands your partner the best card in the deck. That is not a freak of the
sampler, it is the standard *order up your partner* play, and the table below
shows it is systematic rather than anecdotal:

| position, 0 trump | n | order | ≥0 | gain | gain ≥0 |
| --- | --- | --- | --- | --- | --- |
| dealer's partner | 1121 | −0.471 | 48.1% | −0.288 | **71.2%** |
| third hand | 1019 | −1.453 | 17.3% | −0.507 | 47.8% |
| eldest | 1087 | −1.516 | 15.5% | −0.569 | 43.1% |

So a zero-trump order is a blunder from eldest and roughly a coin-flip from the
dealer's partner. **"Never order without trump" cannot be adopted as an axiom.**
Pruning on it would throw away a real and well-known play.

## Trump count

| effective trump | n | order | ≥0 | gain | gain ≥0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 3227 | −1.133 | 27.4% | −0.452 | 54.4% |
| 1 | 8868 | −0.865 | 35.5% | −0.374 | 54.5% |
| 2 | 7865 | −0.139 | 57.2% | +0.069 | 62.9% |
| 3 | 3342 | **+0.769** | 83.2% | +1.013 | 82.3% |
| 4 | 655 | +1.382 | 97.9% | +2.076 | 95.0% |
| 5 | 43 | +1.674 | 100% | +2.814 | 100% |

The dealer's count includes the up-card it is about to take, since ordering
guarantees it — counting the dealt hand instead would make the dealer's rows
mean something different from everyone else's.

**Mean order crosses zero between 2 and 3 trump.** But that is the least useful
way to read this table, because:

## Position dominates hand strength

| position | n | order | ≥0 | gain | gain ≥0 |
| --- | --- | --- | --- | --- | --- |
| dealer | 6000 | +0.162 | 65.8% | **+1.327** | 78.5% |
| dealer's partner | 6000 | +0.162 | 65.8% | −0.187 | 81.3% |
| third hand | 6000 | −0.902 | 34.2% | −0.482 | 44.7% |
| eldest | 6000 | −0.902 | 34.2% | −0.553 | 44.7% |

Two structural facts fall straight out of this, and both are worth keeping:

**1. `order` is identical within a team.** Dealer and dealer's partner both read
+0.162 / 65.8%; eldest and third hand both read −0.902 / 34.2%. Not a
coincidence and not a bug — in God Mode the value of ordering up depends on
*which team* calls, not which of the two partners says it. Trump is fixed by the
up-card, the dealer picks up either way, play starts left of the dealer either
way, and scoring is by team. Only `gain` separates the partners, because passing
leads them into different continuations.

**2. Ordering as the dealer's partner is never a gain on average** — −0.288,
−0.213, −0.129, −0.041, 0.000, 0.000 at 0..5 trump. It is *tied* most of the
time (gain ≥ 0 in 71–100%) and occasionally very bad. The reason is that passing
usually reaches the same contract anyway: your partner the dealer can order it
themselves, with a turn more information. Ordering early mostly forfeits that
option without buying anything.

## Position × trump count — where the boundary actually is

| position | first count with mean `order` ≥ 0 | first with mean `gain` ≥ 0 |
| --- | --- | --- |
| dealer | 2 (−0.088, so really 3) | **1** (+0.129) |
| dealer's partner | 2 (+0.576) | never (best is 0.000 at 4+) |
| third hand | 3 (+0.571) | never (best is −0.116 at 4) |
| eldest | 3 (+0.441) | never (best is −0.289 at 4) |

So the answer to "what is the minimum strength for a +EV call" is **there isn't
one number, and the honest form of the answer is a pair**: on the dealer's team,
2 trump; off it, 3 trump. Against perfect defence, an off-team order is never a
gain over passing at any trump count this sample reached.

## Why this does not shrink PIMC bidding, which is what it was for

Two reasons, and the second is the one that matters.

**God Mode defends perfectly.** Every number here assumes the opposition sees
your hand and never misdefends, which makes calling look worse than it is at a
real table. A threshold derived here is therefore *too strict*. Using it to
prune a PIMC bidder would cut calls the PIMC player would rightly make — the
wrong direction for a prune, which needs to be conservative.

**The distributions overlap far too much.** Even at 0 trump from eldest — the
worst bucket in the study — 15.5% of orders still score ≥ 0. There is no
(position, count) cell where ordering is *never* right, so there is no cell that
can be struck off the way Axiom 1 strikes off a discard. Anything built on this
table is a **heuristic with a measurable loss rate**, not a search-space
reduction, and it would change what the bidder does rather than making it
cheaper at the same answer.

That is the substantive finding: Axiom 1 prunes because it is a claim about
*ties* (some equally good discard always exists). This is a claim about
*averages*, and averages do not prune.

## What would actually be worth measuring next

- The same table under the **PIMC** play model rather than God Mode, which is
  the model any prune would have to be safe against. Expensive, but it is the
  only version that could justify a bidding prune.
- Whether a cheap **hand-strength score** predicts `gain` well enough to replace
  the pass-model solve entirely. Pricing a pass is still the dominant cost in
  bidding — far more than the discard candidates Axiom 1 removes — so a
  regression that predicts the pass branch is the lever worth chasing.

## Caveats

- 6,000 deals, so the 4- and 5-trump rows (655 and 43 samples) are thin.
- Loners are off in the pass branch (`allow_loners=False`), matching the default
  elsewhere. Allowing them would make passing look worse and calling relatively
  better.
- `gain` uses `rest_of_auction`, which is God Mode all the way down — the same
  inconsistency CLAUDE.md notes for `pass_model="god"`.
