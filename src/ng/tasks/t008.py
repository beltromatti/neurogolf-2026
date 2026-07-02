"""task008 (05f2a901) — red magnet slides until adjacent to cyan 2x2.

Generator: red box (with nibbled edges; the cell facing the cyan is
guaranteed present) strictly separated from cyan along one axis
(after optional flip/transpose: 4 possible directions). Red translates
along that axis until its bounding box touches the cyan; cyan stays.

Compile (grids are <=16x16 -> work on a 16x16 crop):
  1. fv = share-a-column flag  -> vertical arrangement (else transpose).
  2. canonical frame R1/C1 = fv*X + (1-fv)*X^T.
  3. gap g = max(minCy-maxRed, minRed-maxCy) - 1 via ramp reductions,
     signed shift s = g * sign(red-above).
  4. shift matrix M[i,j] = relu(1-|i-j-s|); R4 = M @ R1 (shift down by s).
  5. un-transpose, reassemble channels 0/2/8, Pad back to 30x30.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G

N = 16


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))

    R = g.op("Slice", "input", i64([2, 0, 0]), i64([3, N, N]), i64([1, 2, 3]))
    C = g.op("Slice", "input", i64([8, 0, 0]), i64([9, N, N]), i64([1, 2, 3]))

    # --- axis flag: share a column => vertical ---------------------------
    csR = g.op("ReduceSum", R, axes=[2], keepdims=1)   # [1,1,1,16]
    csC = g.op("ReduceSum", C, axes=[2], keepdims=1)
    fv = g.op("Sign", g.op("ReduceMax", g.op("Mul", csR, csC), keepdims=1))

    # --- canonicalize (transpose if horizontal) --------------------------
    def blend_t(x):
        xt = g.op("Transpose", x, perm=[0, 1, 3, 2])
        return g.op("Add", xt, g.op("Mul", fv, g.op("Sub", x, xt)))

    R1 = blend_t(R)
    C1 = blend_t(C)

    # --- row extremes via ramps ------------------------------------------
    ramp = f32(np.arange(N, dtype=np.float32).reshape(1, 1, N, 1))
    rev = f32(np.arange(N - 1, -1.0, -1.0, dtype=np.float32).reshape(1, 1, N, 1))

    def rowprof(x):
        return g.op("Sign", g.op("ReduceSum", x, axes=[3], keepdims=1))

    rpR, rpC = rowprof(R1), rowprof(C1)
    A1 = g.op("ReduceMax", g.op("Mul", rpR, ramp), keepdims=1)   # max red row
    rmA2 = g.op("ReduceMax", g.op("Mul", rpR, rev), keepdims=1)  # 15-min red
    B1 = g.op("ReduceMax", g.op("Mul", rpC, ramp), keepdims=1)   # max cyan row
    rmB2 = g.op("ReduceMax", g.op("Mul", rpC, rev), keepdims=1)  # 15-min cyan

    s1 = g.op("Add", rmB2, A1)   # 15 - (minCy - maxRed)
    s2 = g.op("Add", rmA2, B1)   # 15 - (minRed - maxCy)
    gap = g.op("Sub", f32([14.0]), g.op("Min", s1, s2))
    sgn = g.op("Sign", g.op("Sub", f32([14.5]), s1))  # +1 red above, -1 below
    s = g.op("Mul", gap, sgn)                          # signed shift (scalar)

    # --- shift matrix and translation ------------------------------------
    rrd = f32(np.subtract.outer(np.arange(N), np.arange(N)).astype(np.float32))
    Msh = g.op("Relu", g.op("Sub", f32([1.0]),
                            g.op("Abs", g.op("Sub", rrd, s))))
    R4 = g.op("MatMul", Msh, R1)   # [1,1,16,16] shifted down by s

    # --- back to original frame ------------------------------------------
    R4t = g.op("Transpose", R4, perm=[0, 1, 3, 2])
    R5 = g.op("Add", R4t, g.op("Mul", fv, g.op("Sub", R4, R4t)))

    # --- assemble ----------------------------------------------------------
    wocc = np.ones((1, CH, 1, 1), dtype=np.float32)
    occ = g.op("Conv", "input", g.init(wocc))           # [1,1,30,30]
    occ16 = g.op("Slice", occ, i64([0, 0]), i64([N, N]), i64([2, 3]))
    ch0 = g.op("Sub", g.op("Sub", occ16, R5), C)
    Y = g.op("Concat", ch0, R5, C, axis=1)              # [1,3,16,16]
    wsc = np.zeros((CH, 3, 1, 1), dtype=np.float32)
    wsc[0, 0] = wsc[2, 1] = wsc[8, 2] = 1.0
    out10 = g.op("Conv", Y, g.init(wsc))                # [1,10,16,16]
    g.op("Pad", out10, out="output",
         pads=[0, 0, 0, 0, 0, 0, 30 - N, 30 - N])
    return g.model()
