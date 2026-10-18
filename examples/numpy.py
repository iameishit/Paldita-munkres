"""numpy arrays and pandas DataFrames, with labels."""

import sys

# This file is named numpy.py; keep its own folder off the import path so that
# `import numpy` finds the real library instead of this script.
sys.path[:] = [
    p
    for p in sys.path
    if p not in ("", __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
]

import numpy as np

from munkres import linear_sum_assignment, solve

costs = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
rows, cols = linear_sum_assignment(costs)  # arrays in, arrays out (like SciPy)
print("total:", costs[rows, cols].sum())

try:
    import pandas as pd
except ImportError:
    print("(install pandas to see the labelled example)")
else:
    frame = pd.DataFrame(costs, index=["ana", "ben", "chen"], columns=["x", "y", "z"])
    print(solve(frame).labelled())
