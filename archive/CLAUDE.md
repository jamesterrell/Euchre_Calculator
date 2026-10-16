<!-- Loads only when working under archive/. Moved out of the root CLAUDE.md
     on 2026-10-16; content is verbatim. -->

# Archived approaches

`archive/beta_approach/` holds the previous solver: `tree_search.py`,
`n_play_round.py`, `n_branches.py`. They import each other by bare name, so to
run them put that directory on the path rather than treating it as a package:

```python
import sys; sys.path.insert(0, "archive/beta_approach")
from tree_search import definitive_winner
```

**It disagrees with `fast_search` on roughly 20% of hands, in both directions,**
because it does not compute a minimax value. It scores a move by the **mean**
outcome over branches surviving its heuristic filters, then prunes players
greedily one at a time. On `test_hand.txt` it returns 1 where the true value is
2 -- the calling team can force all five tricks. Treat its numbers as historical.

How it worked, briefly: `n_play_round` built the full Cartesian product of every
player's remaining cards (`n**4` rows) and narrowed it with a filter chain --
`n_ap_filter` to pin the opener, then for each responder in seating order
`nfb_by_hand` (follow suit), `common_sense`, `smart_loss`, `trump_or_dump`. Only
the first is a legality rule; the other three are strategy heuristics, and
mixing them into the same chain is precisely why legal-but-optimal plays got
discarded. `tree_search` then expanded that tree once per candidate opening card
and again once per responder prune, for each of 5 tricks -- roughly 40
expansions of the same tree per hand.

Two hazards if you ever edit it: `next_round` writes each trick's winner at
column `game_round - 1` of an always-5-wide score array, so short simulations
need padded score arrays and shifted `game_round` values -- hand-unrolled and
duplicated across `n_trick_sim`, `player_best_prune`, and `definitive_winner`
with different offsets in each. And buffers are worst-case preallocated
(`round1` 625 branches, `next_round` `len(leads) * 4**4`), which is what drove
the ~533 MB resident footprint.

`archive/legacy_approach/` is older still, superseded by both. Do not extend it.
