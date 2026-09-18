## TL/DR

- The engine knows the rules of Euchre and nothing else. Every number it prints
  is the result of monte carlo tree search.
- Played face up, Euchre is solvable exactly, quickly. That answer is correct
  and it is not advice, because it assumes you can see through the backs of the
  cards.
- Take the knowledge away and the same solver becomes a player: imagine the
  unseen cards a few hundred ways, solve each, and take the option that does
  best on average.
- The hardest question for that player is not which card to lead. It is **what
  a pass is worth**, because passing hands the deal to somebody else and the
  engine has to guess what they will do with it.
- Value a pass by playing the auction out and you get a nervous bidder who
  calls too much and is euchred for it. Value a pass at nothing and you get a
  bidder that behaves far more like a person -- and, oddly, costs you nothing
  in points.
- And the one result to take to a table: holding the right bower and the ace of
  trump with the nine turned, **ordering it up is worth -0.67 points a deal and
  gets you euchred 58% of the time.** A face-up table declines it ten thousand
  times out of ten thousand, and calls next instead.

# A Euchre engine that never guesses

Ask most Euchre software what a hand is worth and somewhere inside it there is
a table of rules: count your trump, add a point for an off-suit ace, order it
up from third seat with two and a face card. Those rules came from somewhere --
usually from a good player's judgment -- and they are what the program actually
knows.

This engine has none of them. It knows the rules of Euchre and nothing else.
No hand-strength count, no "always call next", no opening-lead convention, not
even the idea that the right bower is a good card. When it tells you a hand is
worth two points, it has worked that out by playing the hand to the end, every
legal way it could go, and counting.

---

## How it thinks: play every hand, then work backwards

Start with a deal where all 24 cards are face up. Trump is fixed, somebody has
called, and there are five tricks to play.

The engine plays it out. Not one line, but *every* line: every legal card the
leader could lead, against every legal reply, all the way down to the last card
of the fifth trick, where it counts the tricks and scores the hand.

Then it works backwards up the tree, and that is where the real work happens.
Take a single decision on its own. Say a seat is on lead with three legal cards,
and each one has already been played out to the end:

```
                       a seat is on lead
                               |
         +---------------------+---------------------+
         |                     |                     |
    lead the bower         lead the ace         lead the nine
         |                     |                     |
         |   ... every legal continuation, played to the end ...
         |                     |                     |
        +2                    +1                    -2
         |                     |                     |
         +---------------------+---------------------+
                               |
                  this seat called the hand,
                  so it keeps the best of them:     +2
```

That lead is now worth `+2`, and that number gets handed up to whoever chose the
card before it. If that seat is a **defender**, it keeps the *worst* number it
can reach instead, because a defender wants the calling team to score as little
as possible.

Twenty cards get played in a Euchre hand, so that stack of best-worst-best-worst
runs twenty deep. Every seat picks its card assuming everyone after it plays as
well as they possibly can.

What falls out at the top is not an estimate. It is the score the hand produces
when all four players are perfect: `+2` for a march, `+1` for three or four
tricks, `-2` if the calling team is euchred, and `+4` for a march called alone.
There is no
`+3` and no `0` -- those scores don't exist in Euchre, and they never come out
of the engine either.

**The only thing the engine ever refuses to do is play an illegal card.** It
follows suit when it can, and that is the entire extent of its Euchre knowledge.
It has no opinion on whether leading a singleton ace is clever. If leading the
ace is right, it is right because the playing-out says so.

### Why that finishes before the heat death of the universe

Playing out every line sounds impossible, and very nearly is. On the engine's
standard test hand there are about **three million** distinct ways the cards can
fall.

Two shortcuts bring that down, and neither one throws away a correct answer.

The first is the one every game-playing program uses: **stop reading a line the
moment it cannot change the answer.** Suppose you are choosing between leading
the right bower and leading a nine, and you have already established the bower
is worth at least a point. As soon as the nine's best remaining case drops below
a point, you can stop reading it. You never need to learn *how* bad the nine is,
only that it loses. Whole branches disappear this way.

The second is specific to Euchre's scoring. The calling team needs three tricks.
So the instant the defenders have three, the hand is worth exactly `-2` no
matter what happens next, and there is no reason to play the last two tricks.
Equally, once the calling team has three tricks *and* has already dropped one, a
march is off the table and the hand is worth exactly `+1` whatever follows. Both
checks fire between tricks, and both are certainties rather than guesses.

Measured on that test hand: **14,478 positions examined instead of 3,004,277 --
about half of one percent, or 208 times fewer.** Same answer both ways, which
was checked directly against a version of the search with every shortcut ripped
out. Two thousandths of a second instead of nine seconds.

A full deal solves in about half a millisecond. That number is worth
remembering, because everything later in this document is built out of hundreds
of thousands of them.

---

## God Mode: the whole deal, face up. Completely Unrealistic

Trick play is only half a deal. The engine solves **the auction** the same way,
with every seat seeing every hand and bidding in full knowledge of how the rest
of the bidding and all five tricks will go.

That is all God Mode means: **the whole deal solved under perfect information.**
Who calls, what trump, what the dealer buries, and every card of every trick,
worked out exactly, with nothing hidden from anybody.

It is cheap, too. Round one is four order-or-pass decisions and an order ends
it; round two is four name-or-pass decisions. Nobody's decision multiplies
anyone else's, so the work adds up rather than exploding: **exactly 32 solved
hands to settle an auction**, or 60 once loners are allowed. A complete auction
takes about a fortieth of a second.

Everything in this document that says "face up" or "perfect" came out of God
Mode. It is the yardstick the rest gets measured against.

---

## Why face-up Euchre isn't Euchre

Everything above is exact, and none of it describes a game anybody has played.
Four players who can each see all 24 cards aren't playing Euchre; they are
jointly performing a solved line. Two symptoms of how far that is from a real
table, both measured:

**A perfect table essentially never passes a deal in.** Over 1,600 auctions:
**zero** throw-ins. With every card visible, somebody can nearly always find a
call that is at worst harmless. Real tables pass hands in all the time.

**When a perfect table does get euchred, it is on purpose.** Over 300 deals, God
Mode was euchred 39 times. Check what those 39 seats were avoiding and it is the
same thing every time: in all 39, passing let an opponent call a successful
loner.

The arithmetic is brutal and correct. Being euchred hands over exactly 2, which
is also what letting them march costs you -- so the only outcome worse than a
euchre is an opposing **lone** march at 4. Those 39 seats each took a euchre to
save two points. Every one was the cheapest available disaster.

No human table produces that behaviour, because no human knows what the
alternative was. God Mode is the yardstick everything else is measured against.
It is not a player.

---

## Playing blind: imagining the other hands

The other mode models a real table. Each of the four seats sees **only its own
five cards, the up-card, and what has been played** -- exactly what you see.

When it is your turn, the engine does this:

> Deal the unseen cards into the other three hands and the kitty, in some way
> that fits everything you have watched happen. Solve that imagined deal face
> up, perfectly, and write down what each of your options was worth. Now
> reshuffle the unseen cards and do it again. And again, a couple of hundred
> times. Play whichever card did best on average.

Guess, solve, repeat, vote. That's the whole method.

The word doing the work there is *fits*. These are not random deals. If a seat
has already shown out of hearts, no imagined deal ever gives it a heart again;
the same goes for how many cards each player has left, and for where the turned
card ended up. That constraint is the entire difference between this and
guessing -- deal the unseen cards at random and you would spend most of your
effort on hands that cannot exist. There is a live example further down, where
a seat ruffs a heart lead and the whole table writes it down.

### What it cannot do

Two honest gaps, and the first is the bigger one.

**It does not read the bidding.** Imagined deals are drawn as though the auction
told you nothing at all. A seat that has just ordered up trump is not assumed to
hold any. Real players obviously don't think this way, and it makes the engine
underrate whoever called. The gap is left open deliberately, because closing it
needs a model of what bidders hold -- which is the very thing this project is
trying to produce.

**Inside each imagined deal, everyone is a mind reader.** The imagined hand is
solved face up, so in that imagined world the opponents can see your cards. Two
specific distortions follow. The engine credits itself with plans that only work
if it already knows which imagined deal is the real one -- it finesses with
certainty. And it expects opponents to find defences no human could see, because
in the imagined world they can see everything.

Both make it **optimistic**. Neither makes it weak. Keep it in mind when reading
the numbers below: part of every value it reports is a player believing in a
plan it could not really have executed.

---

## One deal, out loud

The engine can narrate a single deal: what each seat could see, what each option
was worth, and which it took. Every number is an average over a couple of
hundred imagined deals, scored for that seat's own team, so bigger is better and
the largest number is the move taken.

The referee prints all four hands first. **No player below ever sees this.**

```
  seat 0: 9S JS JD AC TC
  seat 1: 9D JC 9H KH AH
  seat 2: QC KC AS KD QS
  seat 3 (dealer): TS 9C JH TD TH
  up-card: QD
  buried:  KS QH AD
```

Round one. The queen of diamonds is turned, and it comes round to the dealer:

```
    seat 0 (eldest) holds 9S JS JD AC TC
        orders up diamonds alone    -1.96
        orders up diamonds          -0.65
        pass                        +0.08  <-- taken

    seat 3 (dealer) holds TS 9C JH TD TH
        pass                        -1.69
        orders up diamonds alone    -1.13
        orders up diamonds          +0.45  <-- taken
```

Now the discard, chosen by the dealer for the dealer's own side:

```
    seat 3 (dealer) picks up QD and holds six: TS 9C JH TD TH
        JH                          -1.27
        TD                          -0.95
        TS                          -0.10
        9C                          -0.10
        TH                          -0.06  <-- taken
      seat 3 buries TH
```

Look at what it refuses to throw. With diamonds trump the jack of hearts is
the second-best card in the deck, and pitching it is priced at `-1.27`,
comfortably the worst option on the table. Nobody told the engine that. It
imagined a few hundred deals, solved each one, and the card came back
expensive.

### Every trick, every card priced

Here is the whole hand. Diamonds are trump, seat 3 is the caller, so seats 0 and
2 are defending. Every number is scored for that seat's *own* team, so all four
seats are trying to make their own number as large as possible. Where a seat has
a choice, every legal card it holds is listed.

(The transcripts also print a running "it has seen: seat N has no X" note under
each decision. Those are kept below where they carry the point and trimmed
elsewhere for length. Nothing else has been touched.)

**Trick 1.** Seat 0 is on lead with a completely free choice:

```
    Trick 1 -- seat 0 leads.
      seat 0 holds 9S JS JD AC TC -- anything is legal
          TC                          +0.16
          9S                          +0.30
          JS                          +0.30
          JD                          +0.43
          AC                          +0.45  <-- taken
      seat 0 plays AC
      seat 1 holds 9D JC 9H KH AH -- only JC is legal
      seat 1 plays JC
      seat 2 holds QC KC AS KD QS -- must follow clubs, so: QC KC
          QC                          +0.23  <-- taken
          KC                          +0.23
      seat 2 plays QC
      seat 3 holds TS 9C JH TD QD -- only 9C is legal
      seat 3 plays 9C
      ---- trick 1 to seat 0 with AC     (team 0: 1, team 1: 0)
```

Seat 0 leads its off-suit ace and it holds. Worth noting what it turned down:
the right bower was right there at `+0.43`, barely behind the ace at `+0.45`.
The engine has no rule about cashing aces early or about holding bowers back; it
priced both and they came out nearly level.

Seat 2 is the first tie of the hand. Queen and king of clubs both score `+0.23`,
because on this trick they are the same card -- both lose to the ace. When two
options tie exactly, the engine has genuinely found no difference between them,
and it breaks the tie by playing the cheaper. That is why the queen goes.

**Trick 2.** Seat 0 leads again and the defence takes it with a trump nine:

```
    Trick 2 -- seat 0 leads.
      seat 0 holds 9S JS JD TC -- anything is legal
          JD                          +0.48
          TC                          +0.50
          9S                          +0.55  <-- taken
          JS                          +0.55
      seat 0 plays 9S
      seat 1 holds 9D 9H KH AH -- anything is legal
          9H                          -1.07
          KH                          -1.07
          AH                          -1.07
          9D                          -0.84  <-- taken
      seat 1 plays 9D
      seat 2 holds KC AS KD QS -- must follow spades, so: AS QS
        (it has seen: seat 1 has no spades)
          AS                          -0.07
          QS                          -0.05  <-- taken
      seat 2 plays QS
      seat 3 holds TS JH TD QD -- only TS is legal
        (it has seen: seat 1 has no spades)
      seat 3 plays TS
      ---- trick 2 to seat 1 with 9D     (team 0: 1, team 1: 1)
```

Look at seat 1. All three of its hearts price identically at `-1.07` -- they are
interchangeable rubbish here -- and ruffing with the nine of trump is worth
`-0.84`. Still a losing number, because seat 1 is on the calling side and this
hand is not going well, but a quarter of a point better than throwing away. It
trumps in, and a nine takes the trick.

Seat 2 holds the ace of spades and declines to spend it, playing the queen for a
hundredth of a point. There is no "save your ace" rule in there either.

**Trick 3.** The best trick of the hand, three trumps deep:

```
    Trick 3 -- seat 1 leads.
      seat 1 holds 9H KH AH -- anything is legal
          9H                          -1.34
          KH                          -1.16  <-- taken
          AH                          -1.16
      seat 1 plays KH
      seat 2 holds KC AS KD -- anything is legal
        (it has seen: seat 1 has no spades)
          KC                          +0.14
          AS                          +0.14
          KD                          +0.52  <-- taken
      seat 2 plays KD
      seat 3 holds JH TD QD -- anything is legal
        (it has seen: seat 1 has no spades, seat 2 has no hearts)
          TD                          -0.75
          QD                          -0.75
          JH                          +0.41  <-- taken
      seat 3 plays JH
      seat 0 holds JS JD TC -- anything is legal
        (it has seen: seat 1 has no spades, seat 2 has no hearts,
                      seat 3 has no hearts)
          JS                          -0.36
          TC                          -0.32
          JD                          +0.27  <-- taken
      seat 0 plays JD
      ---- trick 3 to seat 0 with JD     (team 0: 2, team 1: 1)
```

**Seat 1 leads and hates all three of its cards** -- `-1.34`, `-1.16`, `-1.16`.
It is on lead with three hearts and no good answer, so it leads the king. King
and ace tie exactly, and the cheaper one goes.

**Seat 2 is out of hearts and ruffs.** Its two off-suit cards, the king of clubs
and the ace of spades, are both worth `+0.14` -- another exact tie, because both
are simply discarded on this trick. Trumping with the king of diamonds is worth
`+0.52`, more than three times better. So it trumps.

**Seat 3 over-ruffs with the jack of hearts.** Its two plain trumps tie at
`-0.75`; the jack scores `+0.41`. That is a swing of more than a full point, and
seat 3 is the caller, so it spends its big trump rather than let the defence
have the trick.

**Seat 0 takes it with the right bower.** Jack of spades `-0.36`, ten of clubs
`-0.32`, jack of diamonds `+0.27`. Only one of the three is positive, and seat 0
is defending. This is the trick where the defence gets paid.

Catch the note on that last block, too: seat 0 has already written down that
**seat 3 has no hearts**, and from here on no imagined deal ever gives seat 3 a
heart again. Every seat keeps that book on every other seat.

**Trick 4.** The caller trumps in and levels the hand:

```
    Trick 4 -- seat 0 leads.
      seat 0 holds JS TC -- anything is legal
        (it has seen: seat 1 has no spades, seat 2 has no hearts,
                      seat 3 has no hearts)
          JS                          +0.25
          TC                          +0.27  <-- taken
      seat 0 plays TC
      seat 1 holds 9H AH -- anything is legal
        (it has seen: seat 0 has no hearts, seat 2 has no hearts,
                      seat 3 has no hearts)
          9H                          -1.02  <-- taken
          AH                          -1.02
      seat 1 plays 9H
      seat 2 holds KC AS -- only KC is legal
        (it has seen: seat 0 has no hearts, seat 1 has no clubs/spades,
                      seat 3 has no hearts)
      seat 2 plays KC
      seat 3 holds TD QD -- anything is legal
        (it has seen: seat 0 has no hearts, seat 1 has no clubs/spades,
                      seat 2 has no hearts)
          TD                          +0.45  <-- taken
          QD                          +0.45
      seat 3 plays TD
      ---- trick 4 to seat 3 with TD     (team 0: 2, team 1: 2)
```

Seat 1's ace of hearts, which it has been holding all hand, is now worth exactly
what the nine is worth: `-1.02` either way. Nobody can follow hearts any more,
so the ace wins nothing. Seat 3 holds two trumps that tie at `+0.45` and takes
the trick with the lower one.

**Trick 5.** Nobody has a decision left:

```
    Trick 5 -- seat 3 leads.
      seat 3 holds QD -- only QD is legal
      seat 3 plays QD
      seat 0 holds JS -- only JS is legal
      seat 0 plays JS
      seat 1 holds AH -- only AH is legal
      seat 1 plays AH
      seat 2 holds AS -- only AS is legal
      seat 2 plays AS
      ---- trick 5 to seat 3 with QD     (team 0: 2, team 1: 3)
```

One card each, no options priced, because there is nothing to decide. Seat 1
finally plays that ace of hearts into a trumped trick, and seat 2 plays the ace
of spades it protected on trick 2. The queen of trump takes it, and the calling
side has three tricks.

### What it cost, and what it missed

```
  seat 3 ordered up diamonds (dealer pitched TH)
  The calling team took 3 of 5 tricks -> +1 to them, -1 to team 0.

  Between them the four seats solved 8722 complete Euchre hands to play this one.
  Played in God Mode -- every hand face up -- the same contract is worth -2.
```

Two numbers worth keeping. **The four players solved 8,722 complete Euchre hands
between them to play one deal.** That is what having no heuristics costs.

And the contract the blind table brought home for a point was, in truth, a
euchre: played face up it is `-2`. The defenders had a line and could not see
it. That is the price of not being able to see, measured on a single deal.

---

## Pricing a pass

Now the part that matters most, and the part a Euchre player will find most
familiar.

Every bid is compared on one number, **passing included**, and the biggest
number wins. So how you value a pass decides how freely you bid. Get it wrong in
one direction and you never call anything; get it wrong in the other and you
call everything and get euchred for it.

The engine has two ways of putting a number on a pass.

**"Play it out."** In each imagined deal, hand the rest of the auction to a
face-up perfect table and see what happens to you. If the answer is "the
opponents call and make it", your pass is worth a minus number, and even a
losing call may beat it.

**"Nothing."** A pass is worth zero. Call only if calling beats zero.

"Play it out" is obviously the more principled of the two. Passing genuinely
isn't free: a call worth `-1` is still right if passing hands the opponents a
march for `-2`. That is defensive bidding, and "nothing" cannot express it at
all.

**But "play it out" has a bias baked into it, and it is a big one.** The
imagined continuation is played face up by everybody, so the answer it gives
you comes from a table where all three other seats bid better than any of them
could in reality. That prices a pass wrongly, and not always in the same
direction. The deal below shows it going both ways in a single auction, and the
section after that explains why.

### The same deal, priced both ways

Same 24 cards, same seats. The only thing that changed is what a pass is worth:

```
  seat 0: TH TS JS QC KS
  seat 1: JD 9C AD KC TD
  seat 2: TC JC 9H QD KD
  seat 3 (dealer): QH AC 9D 9S KH
  up-card: QS
```

Here is the **entire** auction with a pass priced by playing it out. Every seat,
every option:

```
  Round one -- order up spades, or pass.

    seat 0 (eldest) holds TH TS JS QC KS
        orders up spades alone      -0.35
        orders up spades            +0.75
        pass                        +0.96  <-- taken

    seat 1 holds JD 9C AD KC TD
        orders up spades alone      -1.99
        orders up spades            -0.78
        pass                        -0.40  <-- taken

    seat 2 holds TC JC 9H QD KD
        orders up spades alone      -2.00
        orders up spades            -1.34
        pass                        -1.10  <-- taken

    seat 3 (dealer) holds QH AC 9D 9S KH
        pass                        -1.74
        orders up spades alone      -1.51
        orders up spades            -0.56  <-- taken

    seat 3 (dealer) picks up QS and holds six: QH AC 9D 9S KH
        9S                          -1.11
        AC                          -0.98
        QH                          -0.94
        KH                          -0.94
        9D                          -0.45  <-- taken
      seat 3 buries 9D

  Contract: seat 3 ordered up spades (dealer pitched 9D)
```

Follow it down the table. **Seat 0 has the best hand of the four and passes
anyway**, because its pass scores `+0.96` and its call only `+0.75`. Seats 1 and
2 pass on hands they can see are bad. Then it reaches the dealer, and this is
the part worth reading twice:

**The dealer orders up a hand it expects to lose.** It has priced its own
contract at `-0.56` and taken it anyway, because it priced passing at `-1.74`.
It is not optimistic about the contract; it is frightened of declining. Every
option on its board is negative, and it picks the least bad one.

Here is what happens to it. Spades are trump, seat 3 is the caller, so seats 0
and 2 are defending:

```
    Trick 1 -- seat 0 leads.
      seat 0 holds TH TS JS QC KS -- anything is legal
          TS                          +0.84
          KS                          +0.84
          QC                          +1.09
          TH                          +1.20
          JS                          +1.39  <-- taken
      seat 0 plays JS
      seat 1 holds JD 9C AD KC TD -- anything is legal
          AD                          -1.61
          KC                          -1.41
          9C                          -1.39
          JD                          -1.36
          TD                          -1.36  <-- taken
      seat 1 plays TD
      seat 2 holds TC JC 9H QD KD -- only JC is legal
      seat 2 plays JC
      seat 3 holds QH AC 9S KH QS -- must follow spades, so: 9S QS
          QS                          -1.77
          9S                          -1.68  <-- taken
      seat 3 plays 9S
      ---- trick 1 to seat 0 with JS     (team 0: 1, team 1: 0)

    Trick 2 -- seat 0 leads.
      seat 0 holds TH TS QC KS -- anything is legal
          TS                          +0.84
          KS                          +1.25
          TH                          +1.41  <-- taken
          QC                          +1.41
      seat 0 plays TH
      seat 1 holds JD 9C AD KC -- anything is legal
          AD                          -1.14
          JD                          -0.98
          KC                          -0.95
          9C                          -0.93  <-- taken
      seat 1 plays 9C
      seat 2 holds TC 9H QD KD -- only 9H is legal
      seat 2 plays 9H
      seat 3 holds QH AC KH QS -- must follow hearts, so: QH KH
          QH                          -1.32  <-- taken
          KH                          -1.32
      seat 3 plays QH
      ---- trick 2 to seat 3 with QH     (team 0: 1, team 1: 1)

    Trick 3 -- seat 3 leads.
      seat 3 holds AC KH QS -- anything is legal
          KH                          -1.82
          QS                          -1.48
          AC                          -1.45  <-- taken
      seat 3 plays AC
      seat 0 holds TS QC KS -- only QC is legal
      seat 0 plays QC
      seat 1 holds JD AD KC -- only KC is legal
      seat 1 plays KC
      seat 2 holds TC QD KD -- only TC is legal
      seat 2 plays TC
      ---- trick 3 to seat 3 with AC     (team 0: 1, team 1: 2)

    Trick 4 -- seat 3 leads.
      seat 3 holds KH QS -- anything is legal
          KH                          -1.20
          QS                          -0.86  <-- taken
      seat 3 plays QS
      seat 0 holds TS KS -- anything is legal
          TS                          -0.14
          KS                          +1.48  <-- taken
      seat 0 plays KS
      seat 1 holds JD AD -- anything is legal
          AD                          -0.86
          JD                          -0.41  <-- taken
      seat 1 plays JD
      seat 2 holds QD KD -- anything is legal
          QD                          +0.55  <-- taken
          KD                          +0.55
      seat 2 plays QD
      ---- trick 4 to seat 0 with KS     (team 0: 2, team 1: 2)

    Trick 5 -- seat 0 leads.
      seat 0 holds TS -- only TS is legal
      seat 0 plays TS
      seat 1 holds AD -- only AD is legal
      seat 1 plays AD
      seat 2 holds KD -- only KD is legal
      seat 2 plays KD
      seat 3 holds KH -- only KH is legal
      seat 3 plays KH
      ---- trick 5 to seat 0 with TS     (team 0: 3, team 1: 2)
```

Seat 0, the seat that passed, wins the first trick with the right bower and
never really lets go. Every card seat 1 held on trick 1 priced between `-1.36`
and `-1.61`: the whole calling side knew from the first card that this was
going badly. Trick 4 is where it ends -- seat 3 leads its last trump, seat 0
beats it with the king (`+1.48`, against `-0.14` for holding back), and the
defence has the third trick it needs.

```
  The calling team took 2 of 5 tricks -> -2 to them.
  Euchred: they called it and could not make three.
  Played in God Mode -- every hand face up -- the same contract is also worth -2,
  so nothing was lost in the play.
```

The last line matters. Played perfectly, with every hand face up, this contract
is *still* worth `-2`. Nobody misplayed it. The two points were lost in the
auction, by a dealer that talked itself into a call.

Now the same 24 cards, with a pass priced at nothing. The whole auction:

```
  Round one -- order up spades, or pass.

    seat 0 (eldest) holds TH TS JS QC KS
        orders up spades alone      -0.35
        pass                        +0.00
        orders up spades            +0.75  <-- taken
      seat 0 orders up spades

    seat 3 (dealer) picks up QS and holds six: QH AC 9D 9S KH
        9S                          -0.40
        AC                          -0.31
        QH                          -0.18
        KH                          -0.18
        9D                          +0.17  <-- taken
      seat 3 buries 9D

  Contract: seat 0 ordered up spades (dealer pitched 9D)
```

That is the entire auction. Seat 0 compares `+0.75` against `0.00`, takes its
own call, and **the bidding never reaches seats 1, 2 or the dealer at all**.

The same 24 cards, the same trump, but now seat 0 is the caller and seats 1 and
3 are defending:

```
    Trick 1 -- seat 0 leads.
      seat 0 holds TH TS JS QC KS -- anything is legal
          TS                          +0.02
          KS                          +0.02
          QC                          +0.21
          TH                          +0.35
          JS                          +0.64  <-- taken
      seat 0 plays JS
      seat 1 holds JD 9C AD KC TD -- anything is legal
          AD                          -0.74
          KC                          -0.55
          9C                          -0.54
          JD                          -0.50
          TD                          -0.50  <-- taken
      seat 1 plays TD
      seat 2 holds TC JC 9H QD KD -- only JC is legal
      seat 2 plays JC
      seat 3 holds QH AC 9S KH QS -- must follow spades, so: 9S QS
          QS                          -1.19
          9S                          -1.11  <-- taken
      seat 3 plays 9S
      ---- trick 1 to seat 0 with JS     (team 0: 1, team 1: 0)

    Trick 2 -- seat 0 leads.
      seat 0 holds TH TS QC KS -- anything is legal
          TS                          +0.01
          QC                          +0.52
          TH                          +0.61
          KS                          +0.62  <-- taken
      seat 0 plays KS
      seat 1 holds JD 9C AD KC -- anything is legal
          AD                          -0.71
          KC                          -0.55
          9C                          -0.37
          JD                          -0.36  <-- taken
      seat 1 plays JD
      seat 2 holds TC 9H QD KD -- anything is legal
          QD                          +0.21
          KD                          +0.21
          TC                          +0.24
          9H                          +0.24  <-- taken
      seat 2 plays 9H
      seat 3 holds QH AC KH QS -- only QS is legal
      seat 3 plays QS
      ---- trick 2 to seat 0 with KS     (team 0: 2, team 1: 0)

    Trick 3 -- seat 0 leads.
      seat 0 holds TH TS QC -- anything is legal
          TS                          +0.37
          QC                          +0.67
          TH                          +0.73  <-- taken
      seat 0 plays TH
      seat 1 holds 9C AD KC -- anything is legal
          AD                          -0.44
          KC                          -0.33
          9C                          -0.25  <-- taken
      seat 1 plays 9C
      seat 2 holds TC QD KD -- anything is legal
          QD                          -0.60
          KD                          -0.60
          TC                          -0.55  <-- taken
      seat 2 plays TC
      seat 3 holds QH AC KH -- must follow hearts, so: QH KH
          QH                          -0.73  <-- taken
          KH                          -0.73
      seat 3 plays QH
      ---- trick 3 to seat 3 with QH     (team 0: 2, team 1: 1)

    Trick 4 -- seat 3 leads.
      seat 3 holds AC KH -- anything is legal
          AC                          -0.66
          KH                          -0.66  <-- taken
      seat 3 plays KH
      seat 0 holds TS QC -- anything is legal
          QC                          +0.41
          TS                          +1.00  <-- taken
      seat 0 plays TS
      seat 1 holds AD KC -- anything is legal
          AD                          -1.00  <-- taken
          KC                          -1.00
      seat 1 plays AD
      seat 2 holds QD KD -- anything is legal
          QD                          +1.00  <-- taken
          KD                          +1.00
      seat 2 plays QD
      ---- trick 4 to seat 0 with TS     (team 0: 3, team 1: 1)

    Trick 5 -- seat 0 leads.
      seat 0 holds QC -- only QC is legal
      seat 0 plays QC
      seat 1 holds KC -- only KC is legal
      seat 1 plays KC
      seat 2 holds KD -- only KD is legal
      seat 2 plays KD
      seat 3 holds AC -- only AC is legal
      seat 3 plays AC
      ---- trick 5 to seat 3 with AC     (team 0: 3, team 1: 2)
```

Seat 0 leads the right bower again, and this time it is leading its own
contract. It takes the first two tricks outright, gives up the third, and wins
the fourth with the ten of trump.

**Watch trick 4.** Seat 0's ten of trump is worth exactly `+1.00`. Seat 1's two
cards are both exactly `-1.00`. Seat 2's two are both exactly `+1.00`. Those are
not roundings. By that point the contract is settled in *every single imagined
deal*, all of them agreeing the calling side takes three tricks for one point,
so the averages collapse onto the whole number. When the engine's numbers stop
having decimals, it has stopped guessing.

```
  seat 0 ordered up spades (dealer pitched 9D)
  The calling team took 3 of 5 tricks -> +1 to them.
```

One valuation produces a dealer taking a contract it had already worked out was
a loser, and losing it. The other produces the eldest hand taking a contract
worth three quarters of a point, and making it.

Compare the two dealer discards, too. Both throw the nine of diamonds, but look
at the signs. Ordering up itself, the dealer's best discard is still `-0.45`:
it is choosing the least bad card for a contract it expects to fail. Ordered up
*by an opponent*, the same card scores `+0.17`, because now the dealer is
picking whatever hurts the caller most. Same card, opposite reasons, and the
engine works out both rather than assuming a helpful dealer.

Worth saying honestly: on *this* deal team 0 actually scored better under "play
it out", because they collected 2 points from the dealer's euchre. A euchre is
suffered by whoever calls. The point isn't that one side did better here; it is
that the dealer made a call it had already priced as bad.

### What it does to a whole table

One deal proves nothing. Here are forty, played twice over -- once by a blind
table valuing a pass by playing it out, once by a blind table valuing a pass at
nothing -- against the same forty deals played face up. Same cards, same seats,
same dealer rotation all three times.

|                              | pass = "play it out" | pass = "nothing" | face up |
| ---------------------------- | -------------------- | ---------------- | ------- |
| **euchred**                  | **17 of 40 (42.5%)** | **5 of 40 (12.5%)** | **3 of 40 (7.5%)** |
| contracts made               | 23 of 40             | 35 of 40         | 37 of 40 |
| marches                      | 6 (15.0%)            | 9 (22.5%)        | 11 (27.5%) |
| mean tricks to the caller    | 2.92                 | 3.62             | 3.58    |
| mean points to the caller    | +0.025               | +0.900           | +1.050  |
| ordered up in round one      | 39                   | 38               | 26      |
| named a suit in round two    | 1                    | 2                | 14      |
| called alone                 | 8 (20.0%)            | 5 (12.5%)        | 0       |
| passed a deal in             | 0                    | 0                | 0       |

**Valuing a pass by playing it out gets you euchred on 42.5% of your contracts.
Valuing it at nothing gets you euchred on 12.5%.** Face-up play manages 7.5%.
Forty deals is small and each of those rates carries ten to fifteen points of
uncertainty on its own, but the gap is about 30 points and survives that
comfortably.

The two rows near the bottom are the tell. The "play it out" table orders up the
turned card **39 times out of 40** and reaches the second round once. The
face-up table orders up 26 times and turns the card down 14 times. Turning a
card down and naming a better suit is ordinary good Euchre, and the "play it
out" bidder almost never does it.

### Why "play it out" gets the pass wrong

The trouble is that the imagined continuation is played face up by *everybody*.

When the engine asks "what happens if I pass?", the answer comes back from a
table where **your partner can see your hand**. So your partner picks the deal
up exactly when their cards complement yours, names the right suit, and makes
it. That is a wonderful partner. It is not the one you have. The partner you
actually have is guessing, and will pass plenty of hands the imagined partner
would have called.

The same fantasy runs on the other side of the table: your opponents can see
everything too, so they take the deal off you precisely when they should.

Both distortions inflate or deflate the price of a pass depending on where you
sit. An early seat sees a **rosy** pass, because three seats are still to speak
and one of them is its partner. The dealer sees a **grim** one, because there is
nobody left to hand it to.

You can watch both in the auction above. Seat 0 prices its pass at `+0.96` and
declines a `+0.75` call, counting on somebody behind it. The dealer prices its
pass at `-1.74` and takes a `-0.56` call, having run out of people to count on.
Three seats pass the deal along, it lands on the one seat that was most afraid
to decline, and that seat gets euchred.

That is the shape of the 42.5%.


### So why prefer "nothing"?

Not because it wins more. Because:

- **It behaves like a Euchre player.** It turns cards down, it reaches the
  second round, and it makes 35 contracts out of 40.
- **It is about four times faster**, since it never has to solve a whole extra
  auction inside every imagined deal.
- **It is exactly right where it matters most.** For the last seat to speak in
  the second round, a pass really does end the deal for nothing. There the two
  models agree, and "nothing" is not an approximation at all.

"Play it out" is the more principled model carrying a bias it cannot shake.
"Nothing" is the cruder model whose error happens to point the other way. They
score the same, and one of them produces an auction you would recognise.

---

## What one hand is worth

Everything so far has been about how the engine works. This is the question it
exists to answer.

Take an ordinary hand:

```
  you hold     JS  AS  9H  9D  TC
  turned       9S
  you are eldest, the dealer is your opponent
```

The right bower, the ace of trump, and three rags. Two of the top three trump.
At most tables this gets ordered up without a second thought.

So pin that decision and price it. Deal the other eighteen cards at random ten
thousand times, force yourself to order it up every time, and let four blind
players bid and play the rest out:

```
Euchre: what is this hand worth at a table that cannot see it?
  your hand                      JS AS 9H 9D TC
  up-card                        9S
  seating                        seat 0, dealer 3, you speak 1 of 4
  sweep                          10000 deals, pass model 'zero'
  assumption                     you order it up whenever the auction reaches you

  PIMC sim (nobody can see your hand)
    deals you ordered              10000 of 10000 (100.0%)
    EV given you ordered           -0.671 +/- 0.031 points per deal
    your team euchred              5759 of 10000 (57.6%)
    trump called                   spades 10000
```

**Ordering it up loses two thirds of a point a deal, and you are euchred on
57.6% of them.**

### Why price a hand like this

Most Euchre simulators fill the four seats with rules: lead trump with three,
duck the first round, save the off-suit ace, lead your partner's suit. This one
has no rule about what to play, ever. **Every card in every imagined deal is
chosen by solving that deal.** The only piece of judgment in the engine is
valuing a pass at zero, and it sits in the auction, not in the play.

Three things follow.

**A rule-driven sim measures its rules, not the hand.** If the seats follow
conventions, the number that comes out is the value of those conventions on this
hand. Change the leading rule and the answer moves. You cannot tell whether the
hand is worth `-0.671` or whether your discard rule is.

**Errors from rules accumulate rather than cancel.** A deal is twenty card
decisions. Random noise averages away across ten thousand deals; a rule that is
slightly wrong fires the same way every time it applies, in the same direction.
More deals make that bias sharper, not smaller.

**Rules are tuned on ordinary hands, and the hands worth asking about are not
ordinary.** The reason to consult a calculator is that a hand is awkward, which
is exactly where a convention is least reliable.

Good play is not supplied to the engine. It discovers it objectively, by
playing the situation out thousands of times and counting what scores best.

### What the run costs

Ten thousand deals across six processes: **41 minutes**, 0.249 seconds a deal,
about **34 million complete Euchre hands solved**.

Measured over thirty deals of this sweep, on one core:

```
  solved hands per deal   mean 3401   median 3245   min 2313   max 4531
  seconds per deal        1.020
```

The depth is the reason. Each deal runs an auction and five tricks; each
decision is settled by imagining 132 layouts for a card, 231 for a bid, 266 for
a discard; each imagined layout is a full Euchre hand solved to the end.

Only the deal count narrows the interval, which is roughly `4/sqrt(deals)`.
Halving `+/- 0.031` takes 40,000 deals, close to three hours. The per-decision
counts move the answer instead of narrowing it, and were measured as the point
where a decision stops changing its mind.

---
