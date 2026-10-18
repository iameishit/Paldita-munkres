# Copyright (c) 2026 Eishit Nigam. Licensed under the Apache License, Version 2.0.
"""Sparse bipartite matching with the SciPy-compatible `min_weight_full_bipartite_matching` signature."""

from __future__ import annotations

from typing import Any

from munkres._core import DISALLOWED, UnsolvableMatrix

__all__ = ["min_weight_full_bipartite_matching"]

_FLOAT_EXACT = 1 << 53


def _edges(matrix: Any) -> tuple[int, int, Any, Any, Any]:
    """(n, m, rows, cols, data) of the stored entries; duplicates are summed, like a CSR matrix."""
    import numpy as np

    if hasattr(matrix, "tocsr"):  # any scipy.sparse matrix/array, without importing scipy
        csr = matrix.tocsr()
        csr.sum_duplicates()
        coo = csr.tocoo()
        return (
            csr.shape[0],
            csr.shape[1],
            np.asarray(coo.row),
            np.asarray(coo.col),
            np.asarray(coo.data),
        )
    dense = np.asarray(matrix)
    if dense.ndim != 2:
        raise ValueError("biadjacency_matrix must be 2-dimensional")
    r, c = np.nonzero(dense)  # a dense array follows csr_matrix(dense): zeros are non-edges
    return dense.shape[0], dense.shape[1], r, c, dense[r, c]


def min_weight_full_bipartite_matching(
    biadjacency_matrix: Any, maximize: bool = False
) -> tuple[Any, Any]:
    """
    Minimum (or maximum) weight matching that covers every vertex of the smaller side of a bipartite graph,
    given as a sparse biadjacency matrix whose stored entries are the edges and their weights
    (same call as `scipy.sparse.csgraph.min_weight_full_bipartite_matching`).

    Returns `(row_ind, col_ind)` sorted by row. Raises `ValueError` (`UnsolvableMatrix`) if no full
    matching exists, and `ValueError` for non-finite weights. Explicit zeros are edges; absent entries are not.
    """
    import numpy as np

    n, m, r, c, w = _edges(biadjacency_matrix)
    if w.size and w.dtype.kind not in "biuf":
        raise ValueError("edge weights must be real numbers")
    if w.dtype.kind == "f" and not np.isfinite(w).all():
        raise ValueError("edge weights must be finite")
    if n == 0 or m == 0:
        e = np.empty(0, dtype=np.intp)
        return e, e.copy()
    if w.dtype.kind == "f" or (w.size and int(np.abs(w).max()) <= _FLOAT_EXACT) or not w.size:
        dense = np.full((n, m), np.inf)  # +inf = no edge; exact for floats and for ints up to 2**53
        wf = w.astype(np.float64)
        dense[r, c] = -wf if maximize else wf  # maximise = minimise the negated weights
        from munkres._api import linear_sum_assignment

        try:
            return linear_sum_assignment(dense)
        except UnsolvableMatrix:
            raise UnsolvableMatrix("no full matching exists") from None
    from munkres._api import solve  # huge integers: exact Python arithmetic

    rows = [[DISALLOWED] * m for _ in range(n)]
    for i, j, x in zip(r.tolist(), c.tolist(), w.tolist(), strict=False):
        rows[i][j] = x
    try:
        res = solve(rows, maximize=maximize)
    except UnsolvableMatrix:
        raise UnsolvableMatrix("no full matching exists") from None
    return np.array(list(res.rows), dtype=np.intp), np.array(list(res.cols), dtype=np.intp)
