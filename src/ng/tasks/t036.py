"""task036 (1f85a75f) — crop the connected "celestial object" to its bbox.

Generator: 30x30; a 4-connected object of colors[0] within a 3..5 x 3..5 box
placed in rows/cols 5..24; noise pixels of the OTHER colors everywhere else
(never within the object's margin, and never of the object's color).
Output = bounding-box crop of the object-color pixels.

Object color identification: the unique present color whose ENTIRE pixel set
fits inside a single 5x5 window (noise colors have ~17 scattered pixels;
probability of a false positive is ~1e-6 per pair - accepted).

Net: per-color 5x5 window counts (depthwise conv on the 20x20 content crop)
vs full-grid totals -> fits flag -> object color value c*; object mask via
scalar equality; row/col compaction to origin via prefix-sum MatMul
permutation matrices; Slice to 5x5; OneHot; mask channel0 outside bbox; Pad.
4-connectivity guarantees contiguous row/col occupancy, so compaction=shift.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = g.init(np.array([1.0], dtype=np.float32))

    # full-grid per-color totals (channels 1..9)
    T = g.op("ReduceSum", "input", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    T9 = g.op("Slice", T, i64([1]), i64([10]), i64([1]))     # [1,9,1,1]

    # content crop (object always within rows/cols 5..24), channels 1..9
    M = g.op("Slice", "input", i64([1, 5, 5]), i64([10, 25, 25]), i64([1, 2, 3]))
    counts = g.op("Conv", M, g.init(np.ones((9, 1, 5, 5), dtype=np.float32)),
                  pads=[2, 2, 2, 2], group=9)                # [1,9,20,20]
    maxw = g.op("ReduceMax", counts, axes=[2, 3], keepdims=1)  # [1,9,1,1]
    # fits = 1[maxw >= T] ; present = 1[T > 0]
    fits = g.op("Sign", g.op("Relu", g.op("Add", g.op("Sub", maxw, T9),
                                           g.init(np.array([0.5], dtype=np.float32)))))
    present = g.op("Sign", T9)
    cvals = g.init(np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1))
    cstar = g.op("ReduceSum", g.op("Mul", g.op("Mul", fits, present), cvals),
                 axes=[1], keepdims=1)                       # [1,1,1,1]

    # scalar color grid + object mask (V == c*)
    V = g.op("Conv", M, g.init(np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)))
    O = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", V, cstar))))
    Vo = g.op("Mul", O, cstar)                               # [1,1,20,20]

    rowhas = g.op("ReduceMax", O, axes=[3], keepdims=1)      # [1,1,20,1]
    colhas = g.op("ReduceMax", O, axes=[2], keepdims=1)      # [1,1,1,20]
    Lst = g.init(np.tril(np.ones((20, 20), dtype=np.float32), -1))  # 1[j<i]
    iota_col = g.init(np.arange(20, dtype=np.float32).reshape(1, 1, 20, 1))
    iota_row = g.init(np.arange(20, dtype=np.float32).reshape(1, 1, 1, 20))

    # Rrow[a,i] = 1[a == dest_r[i]]  (zero rows of Vo make spurious 1s harmless)
    dst_r = g.op("MatMul", Lst, rowhas)                      # [1,1,20,1]
    dstT_r = g.op("Transpose", dst_r, perm=[0, 1, 3, 2])
    Rrow = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", iota_col, dstT_r))))
    # RcolT[i,b] = 1[b == dest_c[i]]
    dst_c = g.op("MatMul", Lst, g.op("Transpose", colhas, perm=[0, 1, 3, 2]))
    RcolT = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", iota_row, dst_c))))

    Y = g.op("MatMul", Rrow, Vo)
    Z = g.op("MatMul", Y, RcolT)                             # compacted scalar grid

    ones_r = g.op("Slice", g.op("MatMul", Rrow, rowhas), i64([0]), i64([5]), i64([2]))
    ones_c = g.op("Slice", g.op("MatMul", colhas, RcolT), i64([0]), i64([5]), i64([3]))
    inb5 = g.op("Mul", ones_r, ones_c)                       # [1,1,5,5]

    Z5 = g.op("Slice", Z, i64([0, 0]), i64([5, 5]), i64([2, 3]))
    oh = g.op("OneHot", g.op("Squeeze", Z5, axes=[1]),
              g.init(np.asarray(10.0, dtype=np.float32)),
              g.init(np.array([0.0, 1.0], dtype=np.float32)), axis=1)  # [1,10,5,5]
    out5 = g.op("Mul", oh, inb5)
    g.op("Pad", out5, out="output", pads=[0, 0, 0, 0, 0, 0, 25, 25])
    return g.model()
