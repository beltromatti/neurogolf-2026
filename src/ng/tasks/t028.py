"""task028 (1bfc4729) — two pixels at rows 2 and 7 expand to a fixed pattern.

size=10 always. Input: pixel color0 at (2,*), color1 at (7,*).
Output (independent of columns!): rows 0,2 full color0; rows 7,9 full color1;
cols 0,9 rows 0-4 color0, rows 5-9 color1; rest black.

Compile: p[ch] = presence of ch in row 2, q[ch] = presence in row 7,
e0 = onehot(black). out[ch] = p[ch]*TOP + q[ch]*BOT + e0[ch]*BLACK
= MatMul(coeff[10,3], M[3,100]) reshaped to 10x10 and padded to 30x30.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # patterns on the 10x10 grid
    top = np.zeros((10, 10), dtype=np.float32)
    bot = np.zeros((10, 10), dtype=np.float32)
    top[0, :] = top[2, :] = 1.0
    bot[7, :] = bot[9, :] = 1.0
    top[0:5, 0] = top[0:5, 9] = 1.0
    bot[5:10, 0] = bot[5:10, 9] = 1.0
    blk = 1.0 - top - bot
    M = np.stack([top.ravel(), bot.ravel(), blk.ravel()])  # [3,100]

    # coeffs: rows 2 and 7 presence per channel + black indicator
    rows = g.op("Gather", "input", i64([2, 7]), axis=2)        # [1,10,2,30]
    pq = g.op("ReduceMax", rows, axes=[3], keepdims=1)         # [1,10,2,1]
    chmask = np.ones((1, 10, 1, 1), dtype=np.float32)
    chmask[0, 0, 0, 0] = 0.0                                   # drop black
    pq = g.op("Mul", pq, g.init(chmask))
    e0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    e0[0, 0, 0, 0] = 1.0
    coeff = g.op("Concat", pq, g.init(e0), axis=2)             # [1,10,3,1]
    coeff = g.op("Reshape", coeff, i64([10, 3]))               # [10,3]

    out = g.op("MatMul", coeff, g.init(M))                     # [10,100]
    out = g.op("Reshape", out, i64([1, 10, 10, 10]))
    g.op("Pad", out, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
