# Roadmap

Where the project is going, in the user's own framing, with status as of
2026-09-13 (end of the PIMC sim work; the bidding work ended at
`85f7f0f` and loners at `c9de0e5`).

The motivating point: **God Mode is a fine baseline but not the answer.**
People need to know what they can expect to score against opponents playing
heuristics and gut instinct, not against opponents who can see all four hands.

## 1. Simulate the full game

- [x] dealing, up-card, kitty, dealer seat -- `game.Deal`
- [x] bidding: order/pass round one, name-a-suit round two, stick-the-dealer
      -- `bidding.solve_bidding`
- [x] pickup and discard -- `Deal.pick_up`, minimaxed in `bidding.order_up`
- [x] **loners** -- `alone=` in the solver, `allow_loners=` in the auction
      (opt-in). Defending alone is still not modelled.
- [x] **a hand played out decision by decision** -- `table.play_deal`: four
      independent player objects, asked one at a time, with the referee
      checking legality. Not the same thing as solving a hand, and the thing
      every non-omniscient player needs.
- [ ] **a game played out to 10 points** -- `Deal` is a single hand, there is no
      running score across hands

## 2. Heuristics per player

- [x] **the player abstraction** -- `players.py`. Three methods, `bid`,
      `discard` and `play`, each handed a turn object by `table.py`. Plain
      objects, no rule DSL, as planned.
- [x] **three players** -- `GodModePlayer` (the old baseline, re-expressed),
      `PIMCPlayer`, `RandomPlayer`.
- [ ] **actual heuristics** -- a rule-based bidder and a rule-based card
      player. The mechanism is shipped; the vocabulary still has to be written,
      and now there is something to measure it against that is neither
      omniscient nor random.

Held to: plain Python callables and objects, not a rule DSL or JSON schema.
Ship the mechanism, let the vocabulary grow out of rules actually written. The
DSL is a refactor to do later from knowledge, a guess if done now.

## 3. EV for a given hand

- [x] EV forced to order -- `bidding.order_up`, ~2.3 ms/deal (6 solves)
- [x] EV over the full auction -- `bidding.solve_bidding`, ~13 ms/deal (32 solves)
- [x] EV of going alone -- `order_up(..., alone=True)`, `first_bid_options`
- [x] **EV against opponents who cannot see your hand** -- `pimc_sweep.py`.
      Not heuristic opponents, but not God Mode ones either, which was the
      distortion that mattered most.
- [ ] **EV against heuristic opponents** -- waiting on the heuristics above.

Both existing ones are God Mode. They answer different questions and both have
a place: `order_up` asks "what do I score if I order", `solve_bidding` asks
"what happens to me holding this hand", which includes the deals where you pass
and somebody else calls.

## 4. Front end

- [ ] not started. Worth doing last: a UI mirrors an API that is about to change
      substantially once players and loners land.

## Two findings that bear on the ordering

**The God Mode auction essentially never passes out** -- 0 of 1600 solved
auctions. Somebody can nearly always find a call that is at worst harmless,
because every seat knows exactly what every other seat will bid. Real tables
throw hands in constantly. That is the sharpest evidence so far that God Mode
distorts the *bidding* harder than it distorts the *play*, which argues for
heuristic bidders ahead of heuristic card play.

**The PIMC sim was the unclaimed middle rung; it is claimed.** `players.PIMCPlayer`,
with `table.py` to drive it and `observation.py` to feed it. It cost one new
solver entry point (`fast_search.solve_position`, for hands that are already
part-played) and no change at all to the search itself.

It did not resolve the passed-out finding above, which is the interesting part.
A PIMC sim table passes out **0 of 60 deals**, exactly like the God Mode one,
and pricing a pass at literally zero does not change that. Eight seats bid in
turn and each round-two seat picks among three suits, so somebody almost always
finds a call that looks positive -- and the sim's estimates are optimistic, so
the bar is low. **Throwing a hand in needs a model of what the other seats will
do with it**, which neither a God Mode continuation nor a flat zero provides.
That is now the sharpest open question in the project, and it is a bidding
question rather than a search question.

What the PIMC sim did show, over the same deals:

- it **over-calls badly** -- 43% of its contracts are euchred against 10% for
  God Mode, and the average call is worth slightly less than nothing;
- that is **not sampling noise** -- 5, 10 and 30 samples give 48%, 45%, 48%;
- it is **mostly the pass model**. Pricing a pass by running the rest of the
  auction in God Mode makes declining look worse than it is, because God Mode
  always finds a call. `pass_model="zero"` cuts the euchre rate to 20-28% and
  is ~4x faster -- but head to head the two are indistinguishable (-1.26 vs
  -1.34 +/- 0.36). The per-call average flattered the quieter bidder because it
  is taken only over the deals it chose to call;
- **loners go from 1.7% to 10%**, the direction real tables go;
- **God Mode is worth 1.26 +/- 0.36 points a deal**, measured head to head with
  the teams swapped on every deal.

## A third finding, from the loner work

**Loners are nearly invisible to a God Mode auction** -- they change the result
on ~1% of deals (6 of 480), always by turning a made contract into a lone march.
The scoring explains it: going alone only pays when you can take all five
unaided, since 3-4 tricks is +1 either way and a euchre costs the same 2. At the
eldest seat over 32 deals, going alone was better on 0, worse on 8, equal on 24.

Real tables call loners far more often than 1%, and lose them. That is the same
gap the passed-out finding points at, from the other direction: God Mode
declines the speculative loner a human takes, and never throws in the hand a
human folds. Both are bidding distortions, not play distortions.

## Structural notes worth not rediscovering

- Loners were expected to be expensive because `_search` hardcodes four seats
  and a 4-card trick. Resolved by writing a second recursion, `_search_alone`,
  rather than parameterising the first: threading the trick width through the
  single function was measured at 1.7x-2.5x slower per node on the four-handed
  hot path, with identical node counts. The copy is guarded by running both
  against the same two independent oracles. If trick play ever becomes
  heuristic-driven, this goes away -- a variable seat count is ordinary code in
  a game loop.
- Bidding decisions need a hand evaluator, which is the thing this tool exists
  to produce. Expect that loop: bootstrap with crude hand-strength rules, feed
  measured EV back into them. The PIMC sim is now available as the bootstrap --
  a hand evaluator, just an expensive and over-optimistic one.
- Sampled worlds deliberately ignore the bidding: a seat that ordered up is not
  assumed to hold trump. Closing that needs a bidding model, which is the thing
  being measured, so the circularity is left open rather than guessed at. It is
  the main reason a PIMC sim player here is weaker than it could be.
- Swapping God Mode bidding for heuristics means replacing the decision at
  `bidding.py`'s `_round_one` with a rule that only looks at `deal.hands[seat]`
  and `deal.up_card`. Everything below it -- `play_value`, the contract, the
  scoring -- stays as is.
