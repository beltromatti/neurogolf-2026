"""task014 (0b148d64) — 4 quadrants of two colors (3x common A, 1x rare B);
output = grid content cropped to the bounding box of the rarest color B,
translated to origin. The B-quadrant bbox contains only B/black cells.

Net: rarest color = argmin of per-channel counts (masking absent + ch0);
B mask via scalar-value equality; bbox extents via masked min/max of index
vectors; translate with two data-dependent 0/1 shift matrices (MatMul);
keep-mask [a<=maxr-minr]x[b<=maxc-minc]; one-hot; mask.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    one = f32([1.0])
    half = f32([0.5])

    wv = np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1)
    v = g.op("Conv", "input", g.init(wv))                  # [1,1,30,30]

    # rarest foreground color
    cnt = g.op("ReduceSum", "input", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    present = g.op("Sign", cnt)
    pen = g.op("Mul", f32([10000.0]), g.op("Sub", one, present))
    ch0pen = np.zeros((1, CH, 1, 1), dtype=np.float32)
    ch0pen[0, 0] = 10000.0
    masked = g.op("Add", g.op("Add", cnt, pen), g.init(ch0pen))
    minc = g.op("ReduceMin", masked, axes=[1], keepdims=1)     # scalar
    rvec = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", masked, minc))))
    cch = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))
    rval = g.op("ReduceSum", g.op("Mul", rvec, cch), axes=[1], keepdims=1)

    # B mask = [v == rval]
    B = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", v, rval))))

    # bbox extents
    arow = f32(np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
    icol = f32(np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    rowhas = g.op("ReduceMax", B, axes=[3], keepdims=1)    # [1,1,30,1]
    colhas = g.op("ReduceMax", B, axes=[2], keepdims=1)    # [1,1,1,30]
    K = f32([100.0])

    def masked_minmax(has, idxv, axis):
        mi = g.op("Mul", idxv, has)
        mx = g.op("ReduceMax", mi, axes=[axis], keepdims=1)
        p = g.op("Mul", K, g.op("Sub", one, has))
        mn = g.op("ReduceMin", g.op("Add", mi, p), axes=[axis], keepdims=1)
        return mn, mx

    minr, maxr = masked_minmax(rowhas, arow, 2)
    mincl, maxcl = masked_minmax(colhas, icol, 3)

    # row shift matrix Rr[a,i] = [i == a + minr]
    Dr = g.op("Sub", g.op("Sub", icol, arow), minr)        # [1,1,30,30]
    Rr = g.op("Relu", g.op("Sub", one, g.op("Abs", Dr)))
    tmp = g.op("MatMul", Rr, v)                            # v[a+minr, j]
    # col shift matrix RcT[j,b] = [j == b + mincl]
    Dc = g.op("Sub", g.op("Sub", arow, icol), mincl)
    RcT = g.op("Relu", g.op("Sub", one, g.op("Abs", Dc)))
    out2 = g.op("MatMul", tmp, RcT)                        # v[a+minr, b+mincl]

    # keep mask
    hr = g.op("Sub", maxr, minr)
    wc = g.op("Sub", maxcl, mincl)
    ka = g.op("Relu", g.op("Sign", g.op("Add", g.op("Sub", hr, arow), half)))
    kb = g.op("Relu", g.op("Sign", g.op("Add", g.op("Sub", wc, icol), half)))
    keep = g.op("Mul", ka, kb)                             # [1,1,30,30]

    d = g.op("Sub", out2, cch)                             # [1,10,30,30]
    oh = g.op("Relu", g.op("Sub", one, g.op("Abs", d)))
    g.op("Mul", oh, keep, out="output")
    return g.model()
