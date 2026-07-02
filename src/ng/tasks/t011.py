"""task011 (09629e4f) — 11x11 hollywood squares (gray lines rows/cols 3,7).
Each 3x3 mini-grid holds 5 rainbow pixels except one with 4; that special
mini-grid's 3x3 pattern is upscaled: local pixel (mr,mc) of color X paints
the whole mini-grid (mr,mc) of the output with X. Gray lines preserved.

Net (scalar color-value domain):
  v9 = 9x9 cell grid (gathered rows/cols skipping lines) as color values
  count per block via 3x3 stride-3 Conv on Sign(v9); S = 5 - count (one-hot)
  pattern p[mr,mc] = Conv(v9 * upsample(S), 7x7 kernel with 1s at 3a,3b)
  scatter p to 11x11 blocks via Gathers with a zero row; add gray lines;
  one-hot compare; pad to 30x30.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # scalar color values
    wv = np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1)
    v = g.op("Conv", "input", g.init(wv))                  # [1,1,30,30]
    cell = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    v9r = g.op("Gather", v, i64(cell), axis=2)             # [1,1,9,30]
    v9 = g.op("Gather", v9r, i64(cell), axis=3)            # [1,1,9,9]

    fg = g.op("Sign", v9)
    k1 = np.ones((1, 1, 3, 3), dtype=np.float32)
    count = g.op("Conv", fg, g.init(k1), strides=[3, 3])   # [1,1,3,3] in {4,5}
    S = g.op("Sub", f32([5.0]), count)                     # 1 at special block
    up = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    Sr = g.op("Gather", S, i64(up), axis=2)                # [1,1,9,3]
    S9 = g.op("Gather", Sr, i64(up), axis=3)               # [1,1,9,9]
    Y = g.op("Mul", v9, S9)                                # only special block

    kp = np.zeros((1, 1, 7, 7), dtype=np.float32)
    for a in range(3):
        for b in range(3):
            kp[0, 0, 3 * a, 3 * b] = 1.0
    p = g.op("Conv", Y, g.init(kp))                        # [1,1,3,3] pattern

    # scatter pattern to 11x11: pad to 4x4 (zero row/col 3), gather with sep=3
    p4 = g.op("Pad", p, pads=[0, 0, 0, 0, 0, 0, 1, 1])     # [1,1,4,4]
    sc = [0, 0, 0, 3, 1, 1, 1, 3, 2, 2, 2]
    br = g.op("Gather", p4, i64(sc), axis=2)               # [1,1,11,4]
    blocks = g.op("Gather", br, i64(sc), axis=3)           # [1,1,11,11]

    # gray separator lines (value 5 at rows/cols 3,7)
    rv = np.zeros((1, 1, 11, 1), dtype=np.float32)
    rv[0, 0, [3, 7], 0] = 5.0
    cv = np.zeros((1, 1, 1, 11), dtype=np.float32)
    cv[0, 0, 0, [3, 7]] = 5.0
    sep = g.op("Min", g.op("Add", g.init(rv), g.init(cv)), f32([5.0]))
    val = g.op("Add", blocks, sep)                         # [1,1,11,11]

    # one-hot
    cch = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))
    d = g.op("Sub", val, cch)                              # [1,10,11,11]
    oh = g.op("Relu", g.op("Sub", f32([1.0]), g.op("Abs", d)))
    g.op("Pad", oh, out="output", pads=[0, 0, 0, 0, 0, 0, 19, 19])
    return g.model()
