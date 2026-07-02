"""task027 (1b60fb0c) — 4-fold rotational symmetry completion.

Generator: creature pixels (r,c); blue copies at
  A=(5-r+off,5+c), B=(4-c+off,5-r+off), C=(5+c,4+r); red copy D=(4+r,4-c+off)
drawn in output only (blue overwrites red on overlap).
The rotation rho_o(i,j) = (j, 9+o-i) cycles A->C->D->B->A.  Hence the closure
blue|rho|rho^2|rho^3 = A|B|C|D and red = closure \\ blue, for the correct
offset o.  Offset chosen as the one whose closure is smaller.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build(fp16: bool = True):
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    blue = g.op("Gather", "input", i64([1]), axis=1)  # [1,1,30,30]
    if fp16:
        blue = g.op("Cast", blue, to=10)  # FLOAT16
    T = g.op("Transpose", blue, perm=[0, 1, 3, 2])

    reds, scores = [], []
    for o in (0, 1):
        idx = i64([(9 + o - b) % 30 for b in range(30)])
        r1 = g.op("Gather", T, idx, axis=3)     # rho(blue)
        r3 = g.op("Gather", T, idx, axis=2)     # rho^3(blue)
        r2 = g.op("Gather", g.op("Gather", blue, idx, axis=2), idx, axis=3)
        clo = g.op("Sign", g.op("Sum", blue, r1, r2, r3))
        reds.append(g.op("Sub", clo, blue))
        scores.append(g.op("ReduceSum", clo, keepdims=0))

    # ind0 = 1 if score0 < score1 else 0
    d = g.op("Sub", scores[1], scores[0])
    ind0 = g.op("Relu", g.op("Sign", d))
    diff = g.op("Sub", reds[0], reds[1])
    red = g.op("Add", reds[1], g.op("Mul", ind0, diff))
    if fp16:
        red = g.op("Cast", red, to=1)

    # paint red(2): input already black there (ch0=1) -> add -1 to ch0, +1 to ch2
    w = np.zeros((10, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = -1.0
    w[2, 0, 0, 0] = 1.0
    lifted = g.op("Conv", red, g.init(w))
    g.op("Add", "input", lifted, out="output")
    return g.model()
