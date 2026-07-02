"""task017 (0dfd9992) — 21x21 doubly-periodic quilt (period = length in 4..9)
with up to five black cutout rectangles (<=5x5); output restores the pattern.

Net: for each candidate period p in 4..9, gate_p = [grid consistent with a
p-shift on all visible overlapping pairs] via sum of v*shift(v)*|v-shift(v)|
(holes are 0 so invalid pairs vanish). Fill = max over the 4 p-shifted grids,
gated; two passes so 2-hop sources (2p / diagonal p,p) are covered. Shifts
are Convs with one-hot kernels + asymmetric pads (cheap params, one tensor).
Pattern colors are 1..9 so no in-grid black handling is needed.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G

N = 21


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = f32([1.0])

    wv = np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1)
    v30 = g.op("Conv", "input", g.init(wv))
    v = g.op("Slice", v30, i64([0, 0]), i64([N, N]), i64([2, 3]))

    def shifts(x, p):
        """4 grids shifted by +-p vertically/horizontally (zero border)."""
        kd = np.zeros((1, 1, p + 1, 1), np.float32); kd[0, 0, 0, 0] = 1.0
        ku = np.zeros((1, 1, p + 1, 1), np.float32); ku[0, 0, p, 0] = 1.0
        kr = np.zeros((1, 1, 1, p + 1), np.float32); kr[0, 0, 0, 0] = 1.0
        kl = np.zeros((1, 1, 1, p + 1), np.float32); kl[0, 0, 0, p] = 1.0
        sd = g.op("Conv", x, g.init(kd), pads=[p, 0, 0, 0])  # x[r-p,c]
        su = g.op("Conv", x, g.init(ku), pads=[0, 0, p, 0])  # x[r+p,c]
        sr = g.op("Conv", x, g.init(kr), pads=[0, p, 0, 0])  # x[r,c-p]
        sl = g.op("Conv", x, g.init(kl), pads=[0, 0, 0, p])  # x[r,c+p]
        return sd, su, sr, sl

    def mismatch(sx):
        d = g.op("Abs", g.op("Sub", v, sx))
        q = g.op("Mul", g.op("Mul", v, sx), d)
        return g.op("ReduceSum", q, axes=[2, 3], keepdims=1)

    gates = {}
    gated1 = []
    for p in range(4, 10):
        sd, su, sr, sl = shifts(v, p)
        S = g.op("Add", mismatch(sd), mismatch(sr))
        gates[p] = g.op("Relu", g.op("Sub", one, S))       # {0,1}
        fill = g.op("Max", sd, su, sr, sl)
        gated1.append(g.op("Mul", fill, gates[p]))
    f1 = g.op("Max", v, *gated1)

    gated2 = []
    for p in range(4, 10):
        sd, su, sr, sl = shifts(f1, p)
        fill = g.op("Max", sd, su, sr, sl)
        gated2.append(g.op("Mul", fill, gates[p]))
    f2 = g.op("Max", f1, *gated2)

    cch = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))
    d = g.op("Sub", f2, cch)                               # [1,10,21,21]
    oh = g.op("Relu", g.op("Sub", one, g.op("Abs", d)))
    g.op("Pad", oh, out="output", pads=[0, 0, 0, 0, 0, 0, 30 - N, 30 - N])
    return g.model()
