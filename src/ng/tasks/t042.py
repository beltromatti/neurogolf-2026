"""task042 (22233c11) — two green m-squares on a diagonal; add cyan squares.

Generator: magnify m in 1..3, 1-2 objects.  Object anchored at (r,col):
  flip=0: green squares (r..r+m-1, col..col+m-1) and (r+m..r+2m-1, col+m..col+2m-1)
          cyan (r-m..r-1, col+2m..col+3m-1) and (r+2m..r+3m-1, col-m..col-1)
  flip=1: greens mirrored horizontally within the 2m x 2m box;
          cyan (r-m..r-1, col-m..col-1) and (r+2m..r+3m-1, col+2m..col+3m-1)
Cyan is clipped at the 10x10 grid border (common.draw).

Net: for each (m, flip) template-match the exact green pattern (+1 on the
2m^2*2 green cells, -1 elsewhere in the surrounding (2m+2)^2 box, bias
-(2m^2-1)) -> anchor hits; then scatter cyan offsets with a second conv
(clipping falls out of conv padding on the 10x10 crop).
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    Gr = g.op("Slice", "input", i64([3, 0, 0]), i64([4, 10, 10]), i64([1, 2, 3]))

    parts = []
    for m in (1, 2, 3):
        s = 2 * m + 2
        kd = -np.ones((2, 1, s, s), dtype=np.float32)
        for f in (0, 1):
            # square tops: flip0 -> (0, 0) and (m, m); flip1 -> (0, m) and (m, 0)
            tops = [(0, 0), (m, m)] if f == 0 else [(0, m), (m, 0)]
            for (tr, tc) in tops:
                kd[f, 0, 1 + tr:1 + tr + m, 1 + tc:1 + tc + m] = 1.0
        bias = g.init(np.full((2,), -(2.0 * m * m - 1.0), dtype=np.float32))
        hits = g.op("Relu", g.op("Conv", Gr, g.init(kd), bias, pads=[1, 1, 2 * m, 2 * m]))
        # scatter cyan: kernel idx i = (3m-1) - dy, j = (3m-1) - dx
        ks = np.zeros((1, 2, 4 * m, 4 * m), dtype=np.float32)
        for f in (0, 1):
            # cyan block top-left offsets (dy, dx)
            blocks = [(-m, 2 * m), (2 * m, -m)] if f == 0 else [(-m, -m), (2 * m, 2 * m)]
            for (by, bx) in blocks:
                for dy in range(by, by + m):
                    for dx in range(bx, bx + m):
                        ks[0, f, 3 * m - 1 - dy, 3 * m - 1 - dx] = 1.0
        c = g.op("Conv", hits, g.init(ks), pads=[3 * m - 1, 3 * m - 1, m, m])
        parts.append(c)
    cyan = g.op("Add", g.op("Add", parts[0], parts[1]), parts[2])  # [1,1,10,10]

    Xc = g.op("Slice", "input", i64([0, 0]), i64([10, 10]), i64([2, 3]))
    w = np.zeros((10, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = -1.0
    w[8, 0, 0, 0] = 1.0
    lift = g.op("Conv", cyan, g.init(w))
    out10 = g.op("Add", Xc, lift)
    g.op("Pad", out10, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
