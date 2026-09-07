"""
Test suite for the Euchre calculator.

The live modules sit at the repo root and import each other by bare name, so
this package puts both the root and this directory on the path. That keeps
`import euchre_testkit` and `from fast_search import solve` working the same way
here as they do in a plain Python session at the root, and it means plain
`python -m unittest discover` needs no extra flags.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
