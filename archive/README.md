# archive

Superseded implementations, kept for reference and comparison. Nothing here is
on the live path; the current solver is `fast_search.py` at the repo root.

## beta_approach/

The previous solver: `tree_search.py` driving `n_play_round.py` and
`n_branches.py`. Breadth-first -- it materialises every branch of a trick as a
dense array, then narrows it with a chain of legality and heuristic filters.

It does **not** compute a minimax value. It scores a move by the mean outcome
over the surviving branches and prunes players greedily, so it disagrees with
`fast_search` on roughly 20% of hands in both directions. On `test_hand.txt` it
returns 1 where the true value is 2.

These modules import each other by bare name, so add the directory to the path
rather than importing it as a package:

```python
import sys; sys.path.insert(0, "archive/beta_approach")
from tree_search import definitive_winner
```

Expect a ~197 s Numba warmup on the first call and ~0.8 s/hand after that.

## legacy_approach/

Older still, predating `beta_approach`. Kept only as a record of the earlier
bit-string and list-comprehension experiments.
