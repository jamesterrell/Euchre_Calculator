# Roadmap

Where the project is going, in the user's own framing, with status as of
2026-09-08 (end of the bidding work, main at `85f7f0f`).

The motivating point: **perfect information is a fine baseline but not the
answer.** People need to know what they can expect to score against opponents
playing heuristics and gut instinct, not against opponents who can see all four
hands.

## 1. Simulate the full game

- [x] dealing, up-card, kitty, dealer seat -- `game.Deal`
- [x] bidding: order/pass round one, name-a-suit round two, stick-the-dealer
      -- `bidding.solve_bidding`
- [x] pickup and discard -- `Deal.pick_up`, minimaxed in `bidding.order_up`
- [ ] **loners** -- not modelled; a call is always four-handed
- [ ] **a game played out to 10 points** -- `Deal` is a single hand, there is no
      running score across hands

## 2. Heuristics per player

- [ ] not started. There is no player abstraction at all -- every decision is
      currently perfect-knowledge minimax.

When it happens: plain Python callables and a registry dict, not a rule DSL or
JSON schema. Ship the mechanism, let the vocabulary grow out of rules actually
written. The DSL is a refactor to do later from knowledge, a guess if done now.

## 3. EV for a given hand

- [x] EV forced to order -- `bidding.order_up`, ~2.3 ms/deal (6 solves)
- [x] EV over the full auction -- `bidding.solve_bidding`, ~13 ms/deal (32 solves)
- [ ] **EV against heuristic opponents** -- the version that actually matters

Both existing ones are perfect-information. They answer different questions and
both have a place: `order_up` asks "what do I score if I order", `solve_bidding`
asks "what happens to me holding this hand", which includes the deals where you
pass and somebody else calls.

## 4. Front end

- [ ] not started. Worth doing last: a UI mirrors an API that is about to change
      substantially once players and loners land.

## Two findings that bear on the ordering

**The double-dummy auction essentially never passes out** -- 0 of 1600 solved
auctions. Somebody can nearly always find a call that is at worst harmless,
because every seat knows exactly what every other seat will bid. Real tables
throw hands in constantly. That is the sharpest evidence so far that perfect
information distorts the *bidding* harder than it distorts the *play*, which
argues for heuristic bidders ahead of heuristic card play.

**PIMC is an unclaimed middle rung.** Perfect-Information Monte Carlo: at a
player's turn, sample N layouts of the unseen cards consistent with what that
player has observed, solve each double-dummy, play the card that scores best on
average. The expensive part already exists -- at ~0.4 ms a solve, 50 samples per
decision is ~20 ms. It yields a strong but genuinely non-omniscient opponent
that fails in realistic ways (it cannot signal, and it suffers strategy fusion).
Offered but not decided on either way.

## Structural notes worth not rediscovering

- Loners were originally going to be expensive because `_search` hardcodes four
  seats and a 4-card trick. That only matters if `fast_search` stays in the play
  path. If trick play becomes heuristic-driven, loners live in the game loop
  where a variable seat count is ordinary code, not surgery on a recursive njit
  hot loop.
- Bidding decisions need a hand evaluator, which is the thing this tool exists
  to produce. Expect that loop: bootstrap with crude hand-strength rules, feed
  measured EV back into them.
- Swapping perfect-knowledge bidding for heuristics means replacing the decision
  at `bidding.py`'s `_round_one` with a rule that only looks at
  `deal.hands[seat]` and `deal.up_card`. Everything below it -- `play_value`,
  the contract, the scoring -- stays as is.
