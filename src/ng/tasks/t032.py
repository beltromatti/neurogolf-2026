"""task032 (1e0a9b12) — gravity: each column's colored cells fall to bottom.

Grid is size x size (4..6); column c holds colors[c] at some rows.  Output:
same count of that color packed at the bottom of the column.  Order within a
column is irrelevant (single color per column).

colored(i,c) = [i >= size_c - n_c] & occupied(i,c) with n_c = #colored cells,
size_c = column height; per-column color from channel presence.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    wfg = np.zeros((1, 10, 1, 1), dtype=np.float32)
    wfg[0, 1:] = 1.0
    fg = g.op("Conv", "input", g.init(wfg))                    # [1,1,30,30]
    occ = g.op("Conv", "input", g.init(np.ones((1, 10, 1, 1), dtype=np.float32)))

    n = g.op("ReduceSum", fg, axes=[2], keepdims=1)            # [1,1,1,30]
    size = g.op("ReduceSum", occ, axes=[2], keepdims=1)        # [1,1,1,30]
    ar = g.init((np.arange(30, dtype=np.float32) + 0.5).reshape(1, 1, 30, 1))
    t = g.op("Add", g.op("Sub", ar, size), n)                  # i - size + n + .5
    A = g.op("Sign", g.op("Relu", t))                          # [1,1,30,30]
    colored = g.op("Mul", A, occ)

    P = g.op("Sign", g.op("ReduceSum", "input", axes=[2], keepdims=1))  # [1,10,1,30]
    P9 = g.op("Gather", P, g.init(np.arange(1, 10)), axis=1)   # [1,9,1,30]

    sub0 = g.op("Sub", occ, colored)                           # black channel
    cp = g.op("Mul", colored, P9)                              # [1,9,30,30]
    g.op("Concat", sub0, cp, axis=1, out="output")
    return g.model()
