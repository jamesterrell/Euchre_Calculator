# Four symmetries of a Euchre position, and why the solver may use them

`bitcore.py` searches far fewer nodes than `fast_search.py` does, and looks up
far more of what it has already searched, and none of that is an approximation.
It rests on four statements about the game, proved here. Each says that two
positions which look different are the *same game*, or that two moves which
look different lead to the same value -- so the search may try one and not the
other, or answer one out of what it learned about the other.

Everything here is about **God Mode** Euchre: perfect information, four
seats, minimax by parity. That is what the solver computes and it is the only
thing these theorems are about. Nothing here says anything about what a PIMC
player should believe.

## Setup and notation

A *position* `P` is: the cards each of the four seats still holds; the cards
already played to the trick now on the table, in order; which seat is to act;
how many tricks the calling team has taken; and, fixed for the whole hand, the
trump suit, whether the caller is alone, and which team called.

Write `esuit(c)` for a card's **effective** suit -- its printed suit, except
that the jack of the same colour as trump is a trump. Within one effective
suit the cards are totally ordered by `rank(c)`; the order across suits is
never consulted, because the rules only ever compare cards in the same suit
(see below). Two facts are all we use about the rules:

* **(R1) Legality.** A seat must play a card of `esuit(led)` if it holds one,
  and may otherwise play anything. The obligation depends on effective suits
  and on nothing else.
* **(R2) The trick.** The trick is won by the highest trump played to it, and
  if no trump was played, by the highest card of the led suit. So the winner is
  the maximum of the trick under a relation that compares two cards only when
  they share an effective suit.

Define the **live set** of `P`:

> `L(P)` = every card in any seat's hand, together with the single card that is
> currently *winning* the trick on the table, if a trick is in progress.

Every one of the twenty-four cards is in exactly one of three places: in a
seat's hand, winning the trick on the table, or **dead** -- buried in the
kitty, spent in a completed trick, or played to the trick on the table and
already beaten. "Dead" does not mean unaccounted for. A card played to this
trick is tracked, precisely, by being *absent from the hand it came from*, and
that absence is as much a part of `P` as the cards still there.

What is dropped is only the dead pile's *internal structure* -- which dead card
is which. The kitty and the completed tricks are uncontroversial there. The
beaten cards of the trick in progress are the case worth arguing, so:

**Lemma 0.** *The value of `P` depends on the cards played to the trick now on
the table only through the suit that was led, the card currently winning it,
the seat that played that card, and how many seats have played.*

*Proof.* Two things in the rules could tell one dead card from another, and
neither does.

**Legality (R1)** is a rule about the cards a seat *holds*. A beaten card is in
no hand, so no future legality test can name it. The seat that played it did
have to follow suit at the time -- but that obligation constrained *which card
left the hand*, and its effect is already recorded in the cards the hand has
left. The led *suit* does have to be remembered, because the seats still to
play are bound by it, and it is.

**The trick (R2)** takes the maximum of the cards played to it. A card that is
not the maximum now can never become one: later cards only raise it. So who
takes the trick is settled by the current winner and by the cards still to
come. And the current winner is a *complete* summary rather than an
approximation of one -- if any trump has been played then the winner is a
trump, so a beaten card can never be a trump the winner does not already beat.

A beaten card therefore enters no further legality test and no further
comparison. The value of `P` is the minimax value of the play still to come;
nothing in that play refers to such a card; so two positions that agree on the
four hands and on `(led, win_card, win_seat, n_in_trick)` have the same value,
whatever their beaten cards were. ∎

*What this is and is not.* It says the beaten cards need not be *named*. It
does not say they are gone from the bookkeeping -- they are exactly as gone as
the kitty, which is to say out of their owners' hands and out of `L(P)`, both
of which the state records. And it is a statement about **values**, not about
**reachability**: two positions with the same forward state can have different
pasts, and one described this way may be one no legal play could produce, since
a seat that followed suit and lost is from here on indistinguishable from one
that could not follow and discarded. That costs nothing. A minimax value is
defined by the tree below a position and never asks how the position was
arrived at, and `bitcore` is in any case only ever handed positions that a real
deal walked it into.

This is why `bitcore` carries a trick as `(led, win_card, win_seat,
n_in_trick)` and not as a list of cards -- the cards themselves are already
accounted for by their absence from the hands. It is a strict reduction of the
state, and it widens the equivalence classes in Theorem 1 by shrinking `L(P)`.

---

## Theorem 1 (equivalent cards in one hand)

> Let `a` and `b` be two cards in the hand of the seat to act, with
> `esuit(a) = esuit(b)`, and suppose no card of `L(P)` other than `a` and `b`
> lies strictly between them in rank. Then playing `a` and playing `b` lead to
> positions of equal value.

*Proof.* Write `P_a` and `P_b` for the positions after playing `a` and after
playing `b`. They differ in exactly one way: in `P_a` the card `a` is on the
table and `b` is in that seat's hand; in `P_b` it is the other way round.
Everything else -- every other hand, the tricks taken, whose turn it is -- is
identical.

Let `τ` be the map that exchanges the names `a` and `b`. Take any complete
play-out of `P_a`, and map it to a play-out of `P_b` by replacing the one
occasion on which that seat plays `b` with playing `a`, and leaving every other
card untouched. This is a bijection between play-outs of `P_a` and play-outs of
`P_b`, because the seat holds exactly one of the two in each position.

It carries legal play to legal play. The only card whose identity changed is
`a`/`b`, and `esuit(a) = esuit(b)`, so by (R1) it is playable exactly when its
counterpart was, and its presence in the hand blocks or permits exactly the
same discards.

It preserves every trick winner. Consider any trick `T` of the play-out. Note
first that `a` and `b` are **never both in `T`**: one of them was played at the
root and the other is played later by the same seat, and a seat plays one card
per trick. So `T` contains at most one of them, and every *other* card of `T`
is unchanged by the map. There are two cases.

* `T` contains neither. Then `T` is identical in both play-outs. Its winner is
  the same.
* `T` contains `a` in one play-out and `b` in the other, played by the same
  seat, alongside the same other cards `c_1, ..., c_k`, none of them `a` or
  `b`. Each `c_i` was live at the moment `T` was played, which is later than
  `P`; but the cards live later are a subset of those live at `P`, since cards
  leave hands and are never returned and a card winning a trick came out of a
  hand. So `c_i ∈ L(P)`, and the hypothesis applies to it: no card of `L(P)`
  other than `a` and `b` lies strictly between them, so for
  every `c_i` of the same effective suit, `rank(c_i) > rank(a) ⟺ rank(c_i) >
  rank(b)`; and cards of other suits compare to `a` exactly as they compare to
  `b`, since `esuit(a) = esuit(b)`. By (R2) the maximum of `T` is therefore
  attained by the same seat in both. The winner is the same.

Every trick has the same winner in corresponding play-outs, so the calling team
takes the same tricks and the payoff is identical. The two game trees are
isomorphic with equal payoffs at corresponding leaves, so their minimax values
agree. ∎

**What it does.** `bitcore._moves` returns one card per maximal run of
consecutive live cards inside the hand, instead of every legal card. A seat
holding `9-10` of a suit with nothing between them searches one move, not two;
a seat holding `9-10-Q` with the jack in the kitty searches one, not three.
`_moves_at`, which owes a value for *every* legal card, searches the
representatives and fills the rest in from them -- by the theorem those numbers
are not guesses.

**A run is more than the theorem says, and transitivity is what bridges the
gap.** The theorem is about a *pair* with nothing live between them; the code
collapses whole runs, and the ends of a run usually do have something between
them. Hearts trump, the queen buried, a hand holding `9H TH KH`: the theorem
applies to `9H` and `TH` (nothing between at all) and to `TH` and `KH` (only
the dead queen between), but *not* directly to `9H` and `KH`, because `TH` lies
between them and `TH` is live -- it is in the hand. What licenses collapsing
all three anyway is that the theorem asserts an equality of values, and
equality is transitive: `val(9H) = val(TH) = val(KH)`.

So the condition the code enforces is the right one, and it is two conditions
rather than one: the cards must be **consecutive in the live order**, and they
must **all be in this hand**. A live card in the gap that the hand does *not*
hold breaks the chain and there is nothing to be transitive through -- with the
queen and jack gone but the ten in an opponent's hand, a hand holding `9H KH`
searches both. `p & ~(p << 1)` over the rank-compressed hand computes exactly
those maximal runs, which is to say the transitive closure, in two table
lookups.

Measured, over 10,000 deals of `hand_ev.py`: the reduction strikes off 13% of
the candidate cards at a typical decision, and because that compounds down
twenty plies it is worth **40% of the nodes** -- 190,884 a deal with it against
267,247 without -- and about 18% of the wall clock.

**What it does not say.** The corresponding statement for two cards in
*different* hands is **false**. If `a` and `b` are held by different seats they
can meet in the same trick, and then exchanging them exchanges which seat wins
it. The proof above turns on the fact that a seat plays one card per trick, and
that is exactly the step that fails.

### Corollary 1a (equivalent cards to bury)

> Let the dealer hold `a` and `b`, let `esuit(a) = esuit(b)`, and suppose every
> card strictly between them is in the kitty. Then burying `a` and burying `b`
> leave positions of equal value.

*Proof.* Identical to Theorem 1, with "the card that stayed in hand" in place
of "the card that was not played at the root": after burying `a`, `b` is in
hand and `a` is out of play for good, and conversely. Only one of them is ever
played, so no trick contains both, and the rest of the argument is unchanged.
∎

This is `fastsim.discard_reps`. It applies to the dealer's five candidates in
`order_up_value` and to the six the PIMC dealer is choosing among, and which
cards it collapses depends on the sampled world, because the world says what is
in the kitty.

---

## Theorem 2 (rank compression)

> Let `φ` be any map that, within each effective suit, is a strictly increasing
> bijection from `L(P)` onto some other set of cards, and let `φ(P)` be the
> position with every live card replaced by its image. Then `P` and `φ(P)` have
> the same value.

*Proof.* `φ` preserves effective suits by construction, so by (R1) it carries
legal plays to legal plays and back. It is strictly increasing within each
suit, so it preserves every comparison the rules make -- and by (R2) the rules
make no others. Hence it carries play-outs to play-outs bijectively, trick
winner to trick winner, and payoff to payoff. The minimax values agree. ∎

Unlike Theorem 1 this is a plain isomorphism and needs no argument about tricks
at all: nothing is being *exchanged*, only relabelled monotonically.

**What it does.** The canonical form takes `φ` to be "renumber each suit's live
cards `0, 1, 2, ...` from the bottom", which is `bitcore.PEXT`. Two positions
that differ only in which dead cards lie between the live ones become one
entry. At the first trick boundary sixteen cards are live out of twenty-four,
so a great many distinct deals compress onto the same key -- which is why the
transposition table is worth sharing between different deals of a sweep, not
just between branches of one search.

---

## Theorem 3 (seat rotation)

> Let `ρ` renumber the seats by `s ↦ (s - k) mod 4` for any `k`, moving each
> hand, the seat to act and the seat winning the trick with it, and replacing
> the calling team `t` by `(t - k) mod 2`. Then `ρ(P)` has the same value as
> `P`.

*Proof.* The rules refer to seats only through the turn order, which `ρ`
preserves because it is a rotation of the cycle, and through the partition into
teams, which `ρ` preserves because teams are the parity classes and `ρ` shifts
parity uniformly -- two seats are partners before iff they are partners after.
The payoff depends on the seats only through which team took each trick, which
is therefore unchanged. ∎

**What it does.** The key is built after rotating so the seat on lead is seat
0, which folds four positions into one and removes the leader from the key
entirely. A loner's sitting partner survives the rotation as "the hand with no
cards", so nothing extra is needed to find it again.

---

## Theorem 4 (plain suits are interchangeable)

> Let `s` and `t` be two non-trump effective suits and let `σ` exchange them,
> carrying every card of `s` to the card of `t` of the same rank and back,
> including the led suit and the winning card. Then `σ(P)` has the same value as
> `P`.

*Proof.* Trump is fixed by `σ` and neither `s` nor `t` is trump. (R1) is
symmetric in the plain suits: a seat must follow whichever plain suit was led,
and holds the same number of them afterwards. (R2) compares cards within a
suit, and ranks trump above everything; a plain card of a suit other than the
led one can never win a trick, before or after. Since `σ` is rank-preserving
within each suit, it maps every comparison to the corresponding one. ∎

*Caveat.* After Theorem 2's compression the three plain suits are no longer
distinguished by their *size* in the deck -- the same-colour suit has five
cards to the others' six only because its jack is trump -- so all three are
interchangeable once only their live cards remain. That is what `_tt_key`
relies on when it sorts all three, rather than only the two off-colour ones.

**What it does.** The three plain fields are sorted into a canonical order by
their compressed contents before the key is assembled, folding up to six
positions into one.

---

## What none of this touches

The **forced-outcome cutoffs** (`caller_tricks + tricks_left < 3` and
`caller_tricks >= 3` with a trick already lost) are not symmetries; they are
arithmetic on the payoff, and they were in `fast_search` already.

The **alpha-beta window** is exact in the usual sense: a fail-soft search
returns the true value whenever that value lies strictly inside its window, and
otherwise returns a bound on the correct side. `fastsim.order_up_value` uses
that and nothing more -- it needs the dealer's *best* discard, not all five
values, so each candidate after the first is searched with the window open on
one side only, and a candidate that comes back at or beyond the incumbent's
bound is rejected without ever being valued.

The **transposition table** stores a value together with the kind of bound it
is, and re-uses it only in ways that bound-kind licenses. Its key is the
canonical form of everything the value depends on -- the four hands under
Theorems 2, 3 and 4, plus the tricks taken, whether the caller is alone, and
which side called -- and the whole key is compared on a hit, so there are no
hash collisions to reason about. An entry written by one deal is therefore as
true for the next, which is why the table is never cleared.

The one thing that is **not** proved here is the `epsilon` stopping rule in
`players._race`, which is a measured approximation and always was. It is
untouched: the compiled engine runs the same rule with the same constants, and
`--epsilon none` turns it off in both.

---

## Three things that were tried and are not here

All three were measured on the same run -- `TH AS AD KD JD`, `9H` up, seat 2,
dealer 0, `--assume order`, 10,000 deals at ten threads -- against a baseline
of 190,880 positions a deal and 21.6 s.

**Killer moves.** The standard trick of remembering, per ply, the card that
refuted a sibling and trying it first. It made the search *worse*: 197,636
positions and 24.2 s. There is already a good move order here -- the cards
that fight for the trick, cheapest of them first -- and a hand holds at most
five cards, so the killer mostly displaces a first move that was already the
right one. Recorded because it is the obvious next thing to reach for.

**Asking a part-played position two yes-or-no questions instead of one open
one.** A value in `{-2, 1, 2}` can be pinned by "is it at least 1?" and then
"at least 2?", each on a window one point wide. That *does* pay for a
whole-hand solve, where `fastsim.play_value` uses it: 190,883 positions
against 200,192. Inside `_moves_at` it pays nothing -- 191,354 positions and
21.9 s, which is inside the run-to-run spread -- because it doubles the number
of searches entered per candidate card and those positions are small enough
that entering one is most of the work. The split is where it is because both
sides were measured.

**A set-associative table.** Eight entries to a bucket and a bucket to a cache
line, so looking at all eight costs the one cache miss that looking at one
costs. While the search was still reaching the nodes that the `lo`/`hi` bounds
now cut off, this was worth 91 s against 47 s -- the single biggest change of
the day. With those bounds in it reversed: 186,190 positions against 190,880
direct-mapped, but 22.1 s against 21.6 s, and much worse when the table is
under pressure -- 388k positions against 294k at 2^22 slots, since eight ways
is also an eighth as many addresses. It is recorded here mainly as a caution:
it was re-measured only because a *different* bug forced a recheck, and it had
been true when it was first measured.

**And one thing that is here for the opposite reason.** The table always
replaces. Depth-preferred replacement -- keep the entry whose subtree was
bigger -- is the usual policy and costs **twice the nodes** here, 391k against
191k. A trick-one entry really does stand for a hundred times the subtree, but
there are far fewer of them than of the trick-two and trick-three entries they
then block out of that slot for the rest of the sweep, and the blocked ones
are the ones being asked for.
