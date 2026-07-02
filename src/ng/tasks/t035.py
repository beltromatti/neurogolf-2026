"""task035 (1f642eb9) — project border pixels onto the cyan pool edge.

Generator: 10x10; cyan pool rows 3..2+h, cols 6-w..5.  Colored pixels sit on
the grid border (row0/row9 at pool columns, col0/col9 at pool rows); each is
copied onto the pool border cell obtained by clamping its coordinates to the
pool (i.e. the nearest pool edge cell in its row/column).

Net (scalar domain): V = color grid; edge masks = topmost/bottommost/left-
most/rightmost cyan cell per column/row; border values broadcast via Gather
with repeated indices; contrib = sum(edge_mask * broadcast); overwrite where
contrib>0; OneHot expand; Pad.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = g.init(np.array([1.0], dtype=np.float32))

    Xc = g.op("Slice", "input", i64([0, 0]), i64([10, 10]), i64([2, 3]))
    V = g.op("Conv", Xc, g.init(np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    C = g.op("Gather", Xc, i64([8]), axis=1)  # cyan mask [1,1,10,10]

    # neighbor reads via 2-tap convs
    kv = np.zeros((1, 1, 2, 1), dtype=np.float32)
    kh = np.zeros((1, 1, 1, 2), dtype=np.float32)
    kv0 = kv.copy(); kv0[0, 0, 0, 0] = 1.0  # with pads[1,.,0,.]: reads (r-1)
    kv1 = kv.copy(); kv1[0, 0, 1, 0] = 1.0  # with pads[0,.,1,.]: reads (r+1)
    kh0 = kh.copy(); kh0[0, 0, 0, 0] = 1.0
    kh1 = kh.copy(); kh1[0, 0, 0, 1] = 1.0
    up = g.op("Conv", C, g.init(kv0), pads=[1, 0, 0, 0])
    dn = g.op("Conv", C, g.init(kv1), pads=[0, 0, 1, 0])
    lf = g.op("Conv", C, g.init(kh0), pads=[0, 1, 0, 0])
    rt = g.op("Conv", C, g.init(kh1), pads=[0, 0, 0, 1])
    Tm = g.op("Relu", g.op("Sub", C, up))  # topmost cyan per column
    Bm = g.op("Relu", g.op("Sub", C, dn))  # bottommost
    Lm = g.op("Relu", g.op("Sub", C, lf))  # leftmost per row
    Rm = g.op("Relu", g.op("Sub", C, rt))  # rightmost

    z10 = i64([0] * 10)
    n10 = i64([9] * 10)
    R0 = g.op("Gather", V, z10, axis=2)  # row 0 broadcast to all rows
    R9 = g.op("Gather", V, n10, axis=2)
    C0 = g.op("Gather", V, z10, axis=3)  # col 0 broadcast to all cols
    C9 = g.op("Gather", V, n10, axis=3)

    contrib = g.op("Add",
                   g.op("Add", g.op("Mul", R0, Tm), g.op("Mul", R9, Bm)),
                   g.op("Add", g.op("Mul", C0, Lm), g.op("Mul", C9, Rm)))
    pm = g.op("Sign", contrib)
    Vout = g.op("Add", g.op("Mul", V, g.op("Sub", one, pm)), contrib)

    idx = g.op("Squeeze", Vout, axes=[1])
    oh = g.op("OneHot", idx, g.init(np.asarray(10.0, dtype=np.float32)),
              g.init(np.array([0.0, 1.0], dtype=np.float32)), axis=1)
    g.op("Pad", oh, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
