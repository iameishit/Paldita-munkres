# Copyright (c) 2008-2020 Brian M. Clapper (original author)
# Copyright (c) 2026 Eishit Nigam (modifications for 2.x)
# Licensed under the Apache License, Version 2.0. See LICENSE.md and NOTICE.
"""
Munkres: the Hungarian (Kuhn-Munkres) algorithm for the assignment problem.

Given a cost for every (row, column) pairing, find the one-to-one assignment
with the lowest total cost::

    >>> from munkres import Munkres
    >>> cost = [[4, 1, 3],
    ...         [2, 0, 5],
    ...         [3, 2, 2]]
    >>> Munkres().compute(cost)
    [(0, 1), (1, 0), (2, 2)]

Forbid a pairing with `DISALLOWED`; turn a profit matrix into a cost matrix
with `make_cost_matrix`. If the forbidden cells make a complete assignment
impossible, `UnsolvableMatrix` is raised at once (it never hangs).

The package has no runtime dependencies. numpy arrays and pandas DataFrames
are accepted as input if you have them, and are never modified.
"""

from munkres._analysis import (
    Prices,
    bottleneck,
    counterfactual,
    k_best,
    shadow_prices,
    tolerance,
)
from munkres._api import (
    Assignment,
    Diagnosis,
    build_cost_matrix,
    diagnose,
    linear_sum_assignment,
    solve,
)
from munkres._core import (
    DISALLOWED,
    DISALLOWED_OBJ,
    DISALLOWED_PRINTVAL,
    AnyNum,
    Cell,
    Matrix,
    MatrixLike,
    Munkres,
    Number,
    UnsolvableMatrix,
    make_cost_matrix,
    print_matrix,
)
from munkres._extras import (
    SinkhornResult,
    Transport,
    sinkhorn,
    soft_assignment,
    stable_matching,
    transport,
)
from munkres._trace import Trace

__all__ = [
    "DISALLOWED",
    "DISALLOWED_OBJ",
    "DISALLOWED_PRINTVAL",
    "AnyNum",
    "Assignment",
    "Cell",
    "Diagnosis",
    "Matrix",
    "MatrixLike",
    "Munkres",
    "Number",
    "Prices",
    "SinkhornResult",
    "Trace",
    "Transport",
    "UnsolvableMatrix",
    "bottleneck",
    "build_cost_matrix",
    "counterfactual",
    "diagnose",
    "k_best",
    "linear_sum_assignment",
    "make_cost_matrix",
    "print_matrix",
    "shadow_prices",
    "sinkhorn",
    "soft_assignment",
    "solve",
    "stable_matching",
    "tolerance",
    "transport",
]

__version__ = "2.0.0"
__author__ = "Brian Clapper, bmc@clapper.org"
__maintainer__ = "Eishit Nigam"
__url__ = "https://github.com/iameishit/Paldita-munkres"
__copyright__ = "(c) 2008-2020 Brian M. Clapper; (c) 2026 Eishit Nigam"
__license__ = "Apache-2.0"
