# The one-trump euchre

A double-dummy position where the dealer is better off holding **one** trump
than two -- and it does not matter which one they keep. Found while checking
whether the solver had a bug. It does not: three independent implementations
agree, including an exhaustive no-pruning enumeration of the whole tree.

Worth playing out with real cards.

## The deal

**Trump is spades.** `9S` was turned up, seat 0 ordered it, seat 3 deals.
Play starts to the dealer's left, so **seat 0 leads**.

Teams: seats 0 + 2 (the callers) versus seats 1 + 3 (the defenders).

```
  seat 0  CALLER   JS  AS  9H  9D  TC
  seat 1           AC  JC  TD  KH  AH
  seat 2           TH  JH  QH  QD  TS      <- caller's partner
  seat 3  DEALER   KS  JD  AD  KC  9C      <- picks up 9S, then discards one

  up-card 9S       buried: KD  QC  QS      (QS is out of play)
```

Trump, high to low. Note `JC` -- the jack of clubs is the **left bower** when
spades is trump, and it beats the ace of spades:

```
  JS   right bower   seat 0
  JC   left bower    seat 1
  AS                 seat 0
  KS                 seat 3
  QS                 buried, out of play
  TS                 seat 2
  9S                 up-card -> seat 3
```

## The puzzle

The dealer holds six cards after picking up `9S` and must throw one. Scores are
to the **calling team**: `-2` means the callers are euchred.

```
  DUMP A TRUMP  ->  EUCHRE
     pitch KS -> keeps JD AD KC 9C + 9S     -2
     pitch 9S -> keeps JD AD KC 9C + KS     -2

  DUMP A SIDE CARD  ->  the callers make it, whichever one
     pitch JD -> keeps    AD KC 9C + KS 9S  +1
     pitch AD -> keeps JD    KC 9C + KS 9S  +1
     pitch KC -> keeps JD AD    9C + KS 9S  +1
     pitch 9C -> keeps JD AD KC    + KS 9S  +1
```

The two euchres are exactly the two lines where the dealer keeps **all four**
of his off-suit cards `JD AD KC 9C`. With five cards he cannot keep four side
cards and both trump, so something has to go -- and the only choice that
euchres is a trump. Either trump; `KS` and `9S` work equally well.

## Things that are checked, so you can skip re-checking them

- It is not a tie or a near miss. All four side-card discards give `+1`, both
  trump discards give `-2`.
- The callers are not misplaying. In the `KS`-pitched position **every** opening
  lead gives `-2`, and after the forced first trick every trick-2 lead also
  gives `-2`.
- Seat 3 is never forced into a particular lead. In the `KS`-kept line all three
  of his trick-3 leads (`KS`, `KC`, `9S`) give `+1`.
- Once the trump is gone the callers are provably stuck, by inspection: the
  defenders hold the master card in every side suit -- `AH`/`KH` and `AC` with
  seat 1, `AD` with seat 3.

## One line worth staring at

From the `KS`-kept position, trick 3, after seat 2 has ruffed a club and seat 3
has won with `AD`:

```
  t3: s3 KC   s0 9H   s1 JC   s2 TH   -> seat 1
```

Seat 0 is void in clubs and holds `JS`. It could ruff and win the trick, and
**declines** -- throwing `9H` away and letting seat 1's left bower take it.
Because if `JS` ruffs here, seat 0 is left leading `AS` into a live `JC`, which
beats it. The callers deliberately lose a trick to draw the bower, and only
then are `JS` and `AS` both clean.

## Reproducing it

```python
import random
import game, bidding as b, rotation as r

my_hand, up = r.parse_hand("JS AS 9H 9D TC"), r.parse_card("9S")
rng = random.Random(7)
for _ in range(3000):
    d = game.deal_around(known_hand=my_hand, seat=0, up_card=up, rng=rng, dealer=3)
    if r.hand_name(d.hands[3]) == "KS JD AD KC 9C":
        break

for c in list(d.hands[3]) + [d.up_card]:
    after = d.pick_up(discard=c)
    print("pitch %-3s -> keeps %-22s callers %+d"
          % (r.card_name(c), r.hand_name(after.hands[3]),
             b.play_value(after, up.suit, 0)))
```
