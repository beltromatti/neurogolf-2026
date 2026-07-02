"""task001 (007bbfb7) — fractal self-product.

Generator: input always 3x3 with 2..8 pixels of one color c (1..9).
Output 9x9: out[rr*3+r][cc*3+c] = color iff pixel at (rr,cc) AND (r,c).

Compile: F[R,C] = fg[R//3,C//3] * fg[R%3,C%3] on a 9x9 canvas;
channel0 = 1-F (all 9x9 cells are inside the grid); colored channels =
F * per-channel spatial max (one-hot color).
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # content 3x3 (all 10 channels)
    s3 = g.op("Slice", "input", i64([0, 0]), i64([3, 3]), i64([2, 3]))
    # fg = any color 1..9  -> [1,1,3,3]
    wfg = np.zeros((1, CH, 1, 1), dtype=np.float32)
    wfg[0, 1:, 0, 0] = 1.0
    fg = g.op("Conv", s3, g.init(wfg))
    # per-channel color presence  -> [1,10,1,1] -> keep 1..9
    cv = g.op("ReduceMax", s3, axes=[2, 3], keepdims=1)
    cv9 = g.op("Slice", cv, i64([1]), i64([10]), i64([1]))

    idxdiv = i64([0, 0, 0, 1, 1, 1, 2, 2, 2])
    idxmod = i64([0, 1, 2, 0, 1, 2, 0, 1, 2])
    a = g.op("Gather", fg, idxdiv, axis=2)
    a2 = g.op("Gather", a, idxdiv, axis=3)
    b = g.op("Gather", fg, idxmod, axis=2)
    b2 = g.op("Gather", b, idxmod, axis=3)
    F = g.op("Mul", a2, b2)  # [1,1,9,9] in {0,1}

    one = g.init(np.array([1.0], dtype=np.float32))
    b0 = g.op("Sub", one, F)              # black channel
    colored = g.op("Mul", F, cv9)         # [1,9,9,9]
    full = g.op("Concat", b0, colored, axis=1)  # [1,10,9,9]
    g.op("Pad", full, out="output", pads=[0, 0, 0, 0, 0, 0, 21, 21])
    return g.model()
