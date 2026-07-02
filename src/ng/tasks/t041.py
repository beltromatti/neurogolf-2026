"""task041 (22168020) — fill V/arrow shapes up to their top row.

Generator: up to 6 "wedge" shapes (descending then ascending diagonal arm,
plus 2 center cells), distinct colors, on a 10x10 grid.  Output paints, for
every column of the shape, from the arm cell upward to the shape's top row.

Equivalent local rule (verified against generator): output(r,c)=color k iff
some k-cell exists at (r',c) with r'>=r AND row r is >= min-row of color k.
Bounding boxes of the shapes never overlap, so fills are disjoint.

Net: per channel: U = column-sum of cells below (incl self); T = prefix-sum
(from top) of "row contains color"; R = U*T.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    C9 = g.op("Slice", "input", i64([1, 0, 0]), i64([10, 10, 10]), i64([1, 2, 3]))
    kcol = g.init(np.ones((9, 1, 10, 1), dtype=np.float32))
    # U(r,c) = sum_{d>=0} X(r+d, c)
    U = g.op("Conv", C9, kcol, pads=[0, 0, 9, 0], group=9)
    rowhas = g.op("ReduceMax", C9, axes=[3], keepdims=1)  # [1,9,10,1]
    # T(r) = sum_{d>=0} rowhas(r-d)  (>0 iff r >= minrow)
    T = g.op("Conv", rowhas, kcol, pads=[9, 0, 0, 0], group=9)
    R = g.op("Mul", U, T)
    any_ = g.op("ReduceSum", R, axes=[1], keepdims=1)
    ch0 = g.op("Sub", g.init(np.array([1.0], dtype=np.float32)), any_)
    cat = g.op("Concat", ch0, R, axis=1)
    g.op("Pad", cat, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
