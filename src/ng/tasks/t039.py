"""task039 (2013d3e2) — output the 3x3 top-left quadrant of the pinwheel.

Generator: 10x10 grid containing only a 4-fold-rotationally-symmetric 6x6
pattern at (row,col), row/col in 1..3.  The mandatory center pixels + the
mandatory extra corner pixel(s) make the fg bounding box exactly 6x6 with
every row/col occupied.  Output = grid[row..row+2][col..col+2] (3x3), black
cells included.

Net: minr/minc via prefix-sum of row/col occupancy; 3x10 row-selection and
10x3 col-selection matrices built as relu(1-|iota_diff - min|); two MatMuls
extract the 3x3 crop for all 10 channels at once.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = g.init(np.array([1.0], dtype=np.float32))

    Xc = g.op("Slice", "input", i64([0, 0]), i64([10, 10]), i64([2, 3]))
    wfg = np.zeros((1, 10, 1, 1), dtype=np.float32)
    wfg[0, 1:] = 1.0
    F = g.op("Conv", Xc, g.init(wfg))  # fg mask [1,1,10,10]

    rowhas = g.op("ReduceMax", F, axes=[3], keepdims=1)  # [1,1,10,1]
    colhas = g.op("Transpose", g.op("ReduceMax", F, axes=[2], keepdims=1),
                  perm=[0, 1, 3, 2])  # [1,1,10,1]
    Lle = g.init(np.tril(np.ones((10, 10), dtype=np.float32)))  # 1[j<=i]

    def first_index(has):  # min index with has>0, as [1,1,1,1]
        s = g.op("MatMul", Lle, has)  # prefix counts
        return g.op("ReduceSum", g.op("Sub", one, g.op("Sign", s)),
                    axes=[2], keepdims=1)

    minr = first_index(rowhas)
    minc = first_index(colhas)

    Dr = g.init((np.arange(10)[None, :] - np.arange(3)[:, None])
                .astype(np.float32).reshape(1, 1, 3, 10))
    Dc = g.init((np.arange(10)[:, None] - np.arange(3)[None, :])
                .astype(np.float32).reshape(1, 1, 10, 3))
    R3 = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", Dr, minr))))
    C3T = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", Dc, minc))))

    O = g.op("MatMul", R3, Xc)      # [1,10,3,10]
    O2 = g.op("MatMul", O, C3T)     # [1,10,3,3]
    g.op("Pad", O2, out="output", pads=[0, 0, 0, 0, 0, 0, 27, 27])
    return g.model()
