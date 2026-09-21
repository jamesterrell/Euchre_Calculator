# Is the reported EV of passing too optimistic?

Written 2026-09-21, on branch `pass_ev_audit`, after `JS KS QS TS 9S` with
`AS` up from seat 3 (dealer 0) priced **passing at slightly positive** and that
looked wrong.

Short answer: **no.** On that hand passing really is worth about `+0.4` with
exact play, and the calculator reports `+0.03` -- it is a third of a point
*pessimistic*, not optimistic. The sign convention is right, the population is
right, and the one number on that hand's report that is badly wrong is
**`order alone`**, which is a full point too high.

## The three things that were suspected, and what each measured

All of it on `JS KS QS TS 9S`, `AS` up, seat 3, dealer 0, bidding order
`[1, 2, 3, 0]`, `pass_model="zero"`, loners on, `epsilon 0.05`, default
budgets.

### 1. Does a successful opponent call come back as a loss?

Yes, exactly as it should. Over 4,000 deals under `--assume pass`, 3,264 of
which reached the seat, **0 records** had a value that disagreed with
`-caller_score` for an opposing caller. Read straight off the cross-tab:

| role            | caller tricks | caller score | value to your team | n   |
| --------------- | ------------- | ------------ | ------------------ | --- |
| opponent called | 4             | +1           | **-1**             | 40  |
| opponent called | 5             | +2           | **-2**             | 28  |
| opponent called | 5             | +4 (alone)   | **-4**             | 21  |
| opponent called | 1             | -2           | **+2**             | 302 |
| partner called  | 2             | -2           | **-2**             | 285 |

`bidding.net_to_team0` then `bidding.value_to` in the Python engine, and
`fastsim.net_to_team0` / `fastsim.value_to` in the compiled one. Both were
checked; there is nothing wrong here.

### 2. Is the +0.03 an artefact of PIMC's play?

No -- it is **lower** than the truth. Paired, same 400 layouts, same seats
pinned to the same opening bids, PIMC's walked outcome against the exact
God-Mode continuation value of the identical layout (`--engine python`, so
`Setup.deal(i)` is shared):

| action      | PIMC walked | exact, same pin | paired PIMC - exact  |
| ----------- | ----------- | --------------- | -------------------- |
| pass        | +0.093      | +0.414          | **-0.321 +/- 0.209** |
| order       | +1.420      | +1.636          | **-0.216 +/- 0.046** |
| order alone | +2.176      | +1.083          | **+1.093 +/- 0.157** |

So PIMC is mildly pessimistic about passing and about ordering, and **wildly
optimistic about the loner**. That is the familiar shape -- the loner's value
lives entirely in whether the defence can find its two stoppers, and PIMC
defenders cannot. Exact play says a lone march happens on 12 of 400 layouts
(3%); PIMC reports 1,235 of 3,264 (38%).

The sceptical eye on this hand's report was pointed at the wrong row.

### 3. Then why *is* passing positive? Decomposition

Because the hand holds **5 of the 6 spades and the up-card is the sixth**. Two
consequences, and they pull in opposite directions:

- if spades ends up trump, no opponent contract in it can survive;
- if spades is turned down, the seat is void in all three live suits and the
  6th spade is buried, so **no other hand holds a spade either** and the five
  cards are worth nothing at all.

The +0.03 is those two facts fighting, over the 3,264 deals that reached the
seat:

| what happened after the pass | n   | mean       | contribution |
| ---------------------------- | --- | ---------- | ------------ |
| dealer ordered up spades     | 542 | **+2.000** | **+0.332**   |
| partner named clubs          | 918 | +0.266     | +0.075       |
| partner named hearts         | 785 | -0.499     | -0.120       |
| partner named diamonds       | 778 | -0.528     | -0.126       |
| opponents named a red suit   | 194 | -1.97      | -0.118       |
| opponents named clubs        |  43 | -1.000     | -0.013       |
| passed out                   |   4 |  0.000     |  0.000       |

Three things to read off it.

**The dealer walks into it 16.6% of the time and is euchred 542 of 542.** Not
"usually" -- every single time. The dealer holds `AS` and picks it up, and the
other five trumps are all sitting in one hand it cannot see. That +0.332 is
most of what makes passing positive.

**Clubs is the only round-two suit the seat survives.** With clubs trump `JS`
is the left bower, so the hand has exactly one trick in it; in hearts or
diamonds it has none, and partner must take three tricks against two opponents
with a dead partner. Partner is euchred on 46% of the contracts it takes.

**Nothing the seat does can rescue a red-suit contract**, which is why `roles`
under `--assume pass` shows `you called: 0` here: with spades turned down there
is nothing to name.

One layout, exactly:

    seat 0  KC JC JD TC TD   (dealer)
    seat 1  9H TH AD 9D QC
    seat 2  JH QH 9C QD AH
    seat 3  JS KS QS TS 9S   <- you
    up AS   kitty KD AC KH

    after you pass: seats 0, 1, 2, 3 all pass | seat 0 names diamonds
    -> -2 to your team.  Ordering it up instead: +1.

That is the losing half. The winning half is the same deal with a dealer that
orders the ace up instead of turning it down.

## What passing is actually sensitive to: the other seats' bidding model

This is the caveat worth carrying away. The pass number is the value of what
the rest of the table does, so it moves with **how the other three bid**. Same
hand, 3,000 deals:

| `pass_model` | pass EV              | reached   | who called after the pass  |
| ------------ | -------------------- | --------- | -------------------------- |
| `zero`       | +0.049 +/- 0.073     | 2442/3000 | partner 1831, opponent 608 |
| `god`        | **+1.390 +/- 0.049** | 3000/3000 | partner 634, opponent 2366 |

`order` is unmoved (+1.430 against +1.435). Under `"god"` the other seats are
frightened of declining, so they call far more often, and on this hand calling
against a seat holding five spades is suicide -- passing becomes nearly as good
as ordering. A 1.34-point swing in the pass number from a knob that is not
about this seat at all.

`api.evaluate` defaults to `"zero"`, which is the more selective bidder and so
the more conservative estimate of what passing is worth. That is the right
default for this, and it should not be read as the value of passing at a real
table, only at a table of these players.

## A hand where passing is negative

`9H TH 9D TD 9C`, `AS` up, eldest seat (seat 1), dealer 0 -- five nines and
tens, nothing in the up-card's suit, and the dealer holding the ace of trump.

| action      | calculator, 1,500 deals | exact, 300 layouts   |
| ----------- | ----------------------- | -------------------- |
| pass        | **-1.288 +/- 0.064**    | **-1.243 +/- 0.125** |
| order       | -1.727 +/- 0.046        | -1.697 +/- 0.106     |
| order alone | -2.000 +/- 0.000        | --                   |

An opponent takes the contract on 1,330 of 1,500 (exact: 219 of 300), and there
is nothing in the hand to punish them with. Note that passing is still the
**best** of the three actions -- a negative pass EV is not advice to bid, it is
the price of the seat.

`AH KH 9D TD 9C`, `JS` up, seat 1, dealer 0 is the same story with a little
defence in it: pass -1.095 +/- 0.061 measured, -1.207 +/- 0.116 exact.

And the reverse case, where passing beats ordering and both are large:
`JS 9S AH AD AC`, `TS` up, seat 1 -- pass +1.088, order +0.707 measured;
+1.370 and +1.107 exact. Turning the ten down and naming spades in round two
keeps the right bower and does not hand the dealer the `TS`, and 157 of 300
exact layouts have this seat calling in round two. **`--assume pass` pins only
the opening bid**, so "pass" on a strong hand routinely means "decline the
up-card and call something better", which is most of why passing looks good on
hands that have a second suit.

## What to be suspicious of instead

`order alone`. It is +1.09 +/- 0.16 too high on the audited hand, it is the row
with the widest interval at any deal count, and `Evaluation.best()` will happily
hand it back as the winner. `notes/front_end.md` already says show the
intervals; this adds a reason that is not about sampling error at all.
