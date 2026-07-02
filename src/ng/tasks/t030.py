"""task030 (1caeab9d) — align the red(2)/yellow(4) shapes to the blue(1) row.

Generator: identical mini-shape stamped in colors 1,2,4 at distinct column
bands and random row offsets; output aligns all three vertically to color 1's
row offset.  Columns never overlap.

Compile: per color ch in {1,2,4}, top row one-hot f_ch; shift-up-by-t_ch
matrix M_up[r,s]=1[s-r=t_ch] is a Toeplitz built by Gather of the zero-padded
f_ch with index (s-r) mod 60; then shift down by blue's t_1 the same way.
out = M_down @ (M_up @ X).  Channel 0 = occupied - union(shifted colors).
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = g.init(np.array([1.0], dtype=np.float32))

    Xc = g.op("Gather", "input", i64([1, 2, 4]), axis=1)      # [1,3,30,30]
    R = g.op("ReduceMax", Xc, axes=[3], keepdims=1)           # [1,3,30,1]
    L = g.init(np.tril(np.ones((30, 30), dtype=np.float32), -1))
    above = g.op("Sign", g.op("MatMul", L, R))
    first = g.op("Mul", R, g.op("Sub", one, above))           # one-hot top rows
    fpad = g.op("Pad", first, pads=[0, 0, 0, 0, 0, 0, 30, 0])  # [1,3,60,1]

    idx_up = np.array([[(s - r) % 60 for s in range(30)] for r in range(30)])
    up5 = g.op("Gather", fpad, i64(idx_up), axis=2)           # [1,3,30,30,1]
    Mup = g.op("Reshape", up5, i64([1, 3, 30, 30]))

    fb = g.op("Gather", fpad, i64([0]), axis=1)               # [1,1,60,1] blue
    idx_dn = np.array([[(r - s) % 60 for s in range(30)] for r in range(30)])
    dn5 = g.op("Gather", fb, i64(idx_dn), axis=2)             # [1,1,30,30,1]
    Mdn = g.op("Reshape", dn5, i64([1, 1, 30, 30]))

    mm1 = g.op("MatMul", Mup, Xc)                             # shifted to top
    mm2 = g.op("MatMul", Mdn, mm1)                            # down to blue row

    occ = g.op("Conv", "input", g.init(np.ones((1, 10, 1, 1), dtype=np.float32)))
    union = g.op("Conv", mm2, g.init(np.ones((1, 3, 1, 1), dtype=np.float32)))
    ch0 = g.op("Sub", occ, union)

    cat = g.op("Concat", ch0, mm2, axis=1)                    # [1,4,30,30]
    w = np.zeros((10, 4, 1, 1), dtype=np.float32)
    w[0, 0] = w[1, 1] = w[2, 2] = w[4, 3] = 1.0
    g.op("Conv", cat, g.init(w), out="output")
    return g.model()
