# Axiom 1 (rejected): the dealer never has to discard a top trump

**Status: FALSE.** Proposed as an axiom, nearly adopted on 10,000 deals, and
killed by extending the same sweep to 100,000. The counterexample is pinned in
`tests/test_axioms.py`.

It is left written up rather than deleted, because the way it failed is the
useful part: it survived 15,515 decisive positions and died at roughly 1 in
100,000, which is what "no counterexample in a large sweep" is actually worth
when the sweep is uniform and the counterexample is structural.

## The claim

When the dealer picks up, the **right bower, the left bower and the ace of
trump** can be struck off the list of discard candidates without ever losing an
optimal discard.

Note the form. It is *not* "discarding a top trump is bad" -- it is:

> there is always an equally good discard that is not a top trump.

Ties are fine. A pruner only has to keep *one* optimal card, so the claim it
needs is that a top trump is never **uniquely** optimal. That is what was
measured.

## The counterexample

`game.deal_from_order`, seed 94137 dealt by seat 1. Trump is hearts (KH up).

```
dealer 1, caller 1, trump hearts, up-card KH
dealer's hand  TD KD JH 9D AH
  pitch AH  -> +2 to the dealer's team   <-- optimal, and the ace of trump
  pitch TD  -> +1
  pitch KD  -> +1
  pitch JH  -> +1
  pitch 9D  -> +1
```

Pitching the **ace of trump** is the only discard that makes a march, and the
prune cannot see it. Same deal, same result with caller 3, and again when the
contract is played alone.

**Why it works**, because this is the shape any further counterexample will
have: the dealer holds JH (the right bower), AH (the ace of trump) and three
diamonds. Pitching the ace keeps three diamonds and two trump; pitching a
diamond keeps three trump and two diamonds. The ace is redundant sitting behind
the right bower, and the third diamond is worth more as *length* than the ace is
as a winner -- so the holding with fewer and weaker trump takes all five tricks
and the one with more takes four.

That is a blocking-and-length effect. It is precisely the structure that uniform
random dealing almost never produces, and precisely why 10,000 deals saw
nothing.

## What was measured

`notes/discard_dominance.py`. For each deal and each caller, every legal
discard is solved in God Mode and scored on the **dealer's own team's scale**
(the dealer chooses the discard for its own side -- see CLAUDE.md, "Bidding").
A counterexample is a position where *every* optimal discard is a top trump.

Measured at `fb3b7bf` + the four-bullets nit, after the up-card stopped being a
legal discard, so the dealer chooses among its five dealt cards.

10,000 deals, dealer rotating, both four-handed and called alone:

| | right bower | left bower | ace of trump |
| --- | --- | --- | --- |
| **four-handed** — held | 7,556 | 8,792 | 7,080 |
| among the optimal discards | 3,522 (46.6%) | 5,308 (60.4%) | 4,698 (66.4%) |
| **uniquely optimal** | **0** | **0** | **0** |
| **alone** — held | 5,667 | 6,594 | 5,310 |
| among the optimal discards | 2,788 (49.2%) | 4,109 (62.3%) | 3,595 (67.7%) |
| **uniquely optimal** | **0** | **0** | **0** |

No counterexample in **9,006 decisive four-handed positions** and **6,509
decisive alone positions**.

### ...and then at 100,000

| | four-handed | alone |
| --- | --- | --- |
| (deal, caller) pairs with a top trump held | 195,964 | 146,973 |
| **counterexamples** | **2** (0.0010%) | **1** (0.0007%) |

All three are the same deal. Right bower: 0. Left bower: 0. **Ace of trump: 2
and 1.** So the claim is false for the ace and survives -- so far, and only so
far -- for the two bowers.

## Read the denominator carefully

"10,000 deals" overstates the evidence by about a factor of two, and the script
reports the honest number instead.

Of the 19,788 four-handed (deal, caller) pairs where the dealer held a top
trump, only **9,006 were decisive** -- positions where the discard changes the
value at all. In the other 10,782 the outcome is already forced and *every*
discard ties, so "a top trump was among the optimal discards" is vacuously
true and tells us nothing. That is also why the middle row of the table is so
high: the right bower is "among the optimal discards" 46.6% of the time mostly
because half of all positions have no wrong answer.

The evidence is 15,515 decisive positions. Not 10,000 deals, and not 34,629
pairs.

## Why this is weaker evidence than the sample size suggests

Uniform random dealing is close to the **weakest** way to test this claim.

The reason a counterexample is conceivable at all is that in trick-taking games
a higher card is not always weakly better to hold. Winning a trick you would
rather have ducked puts you on lead, and leading away from a tenace can cost
your side a trick; a high card can also block your partner's suit. Euchre's
scoring is monotone in tricks, which helps, but tempo is not.

Those structures -- blocking, endplays -- essentially never arise by chance.
A uniform sweep spends nearly all its samples on positions that could not have
broken the claim even in principle. So:

- **100,000 random deals is not 10x the evidence of 10,000.** It is mostly more
  of the same easy positions.
- A **targeted adversarial search** -- hill-climbing on deals to maximise the
  gap in favour of discarding a top trump, or constraining shapes so the
  dealer's bower is exactly the card that jams its partner's suit -- would be
  worth far more per CPU-second, and has **not been done**.
- An **exhaustive proof on a reduced game** (3 cards each, all 12!/(3!^4) =
  369,600 deals enumerated) would be a real theorem about 3-trick Euchre and
  would surface any counterexample structure in miniature. Also **not done**.

Both were considered and deliberately deferred. If this axiom ever has to carry
weight -- if a result depends on it rather than merely runs faster because of
it -- do one of those before trusting it further.

## What it buys: less than it looks like

Measured on `hand_ev.py`'s pinned hand, 40 deals at 200 eval sims, epsilon
0.40, `--prune-top-trumps` against the same run without it:

| | s/deal | bidding solves/deal |
| --- | --- | --- |
| exact | 0.893 | 907 |
| pruned | 0.823 | 839 |

**1.09x faster, 7.5% fewer bidding solves, and 40 of 40 deals came out
identical.**

That is well short of the "three of five candidates disappear" arithmetic, and
the reason is worth writing down so nobody re-derives the optimistic version:
the dealer holds *no* top trump in about a third of deals, and when it holds
one it usually holds exactly one. So the average saving is around one candidate
in five, not three -- and discard solves are only a part of round one, which is
itself only a part of bidding once the epsilon band has taken out the rest.

It is a **search-space reduction, not a model change**: if the axiom holds the
pruned auction returns exactly what the unpruned one did, which is what the
40/40 agreement and `tests/test_axioms.py` are checking.

Worth keeping -- it is free and it composes with everything else -- but it is
not the lever that shrinks bidding. Pricing a pass still is.

## How it is wired in, and why that way

`bidding.solve_bidding` is the project's *exact* baseline -- CLAUDE.md opens by
calling God Mode "the exact baseline, not a model of a real table", and
`tests/test_table.py` pins a table of four `GodModePlayer`s to it. Pruning
`order_up` on an unproven axiom makes that baseline heuristic, and silently so:
if the axiom is false, the baseline is wrong in a way nothing would catch,
because the thing it would be checked against is itself.

So the prune is **opt-in and cross-checked**. `bidding.order_up`,
`solve_bidding`, `rest_of_auction` and `best_discard` all take `prune=False`;
`players.PIMCPlayer` takes `prune_discards=False`; `hand_ev.py` exposes
`--prune-top-trumps`. Nothing switches it on by itself, so every God Mode
number already in CLAUDE.md is unaffected.

`tests/test_axioms.py` is what keeps the axiom falsifiable rather than merely
believed: it runs both paths over 250 deals (four-handed and alone) and asserts
they agree, and compares `order_up` directly for every caller besides. A
counterexample surfaces as a failing test naming the deal, not as a number that
is quietly a bit wrong.

## Statement, as it now stands

> ~~**Axiom 1.** When the dealer picks up, some optimal discard is not the right
> bower, the left bower, or the ace of trump.~~ **False.** Witness: seed 94137,
> dealer 1, hearts. The ace of trump is the uniquely optimal discard.

What survives is a much weaker, and merely empirical, statement:

> The top-trump prune changes the value of ordering up on about **1 position in
> 100,000**, and when it does it costs a march (+1 where the truth is +2).

That is a heuristic with a measured error rate, not an axiom. It is off by
default and `tests/test_axioms.py` pins the witness so the distinction cannot
quietly erode.

**The lesson worth keeping.** The estimate that mattered was never the sample
size. It was that uniform random dealing cannot reach the positions where the
claim breaks -- stated in this file *before* the counterexample turned up, and
then confirmed by the counterexample being exactly a blocking-and-length
position. A targeted adversarial search or the exhaustive reduced-game proof
would have found this in minutes rather than in 100,000 deals. Do that first
next time.
