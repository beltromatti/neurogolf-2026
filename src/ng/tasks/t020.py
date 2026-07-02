"""task020 (11852cab) — EXACT ARC-GEN generator semantics.

10x10 grid; pattern around a center: ring1 = 4 diagonals dist 1, ring2 =
(+-2,0),(0,+-2), ring3 = (+-2,+-2); each ring one color from {1,2,3,4,8};
one ring has 3 of 4 cells missing in the input. Output completes them.

Net (10x10 crop, everything in-grid): value+1 coding V; center C found by
counting pattern hits (center weight 4 + 12 ring offsets, threshold 13);
per-ring color = sum(V at ring)/count via masked reduce + Div; missing
cells = ring offsets from center that are empty; paint; decode via int
Equal against [1..10].
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def _rings():
    r1 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    r2 = [(-2, 0), (0, 2), (2, 0), (0, -2)]
    r3 = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    return [r1, r2, r3]


def build():
    S = 10
    g = G(opset=9)
    x = g.op("Slice", "input", axes=[2, 3], starts=[0, 0], ends=[S, S])
    wv = np.arange(1, CH + 1, dtype=np.float32).reshape(1, CH, 1, 1)
    V = g.op("Conv", x, g.init(wv))  # value+1 coding [1,1,10,10]
    wf = np.ones((1, CH, 1, 1), dtype=np.float32)
    wf[0, 0] = 0.0
    fg = g.op("Conv", x, g.init(wf))  # colored mask

    # center detection: 4*self + 1 per pattern offset, need >= 13
    K = np.zeros((1, 1, 5, 5), dtype=np.float32)
    K[0, 0, 2, 2] = 4.0
    for ring in _rings():
        for dr, dc in ring:
            K[0, 0, 2 + dr, 2 + dc] = 1.0
    C = g.op("Relu", g.op("Conv", fg, g.init(K),
                          g.init(np.array([-12.0], dtype=np.float32)),
                          pads=[2, 2, 2, 2]))  # one-hot center

    # ring kernels [3,1,5,5]
    W3 = np.zeros((3, 1, 5, 5), dtype=np.float32)
    for k, ring in enumerate(_rings()):
        for dr, dc in ring:
            W3[k, 0, 2 + dr, 2 + dc] = 1.0
    W3n = g.init(W3)
    one = g.init(np.array([1.0], dtype=np.float32))
    V0 = g.op("Sub", V, one)  # true color value (black=0; all in-grid)
    RV = g.op("Conv", V0, W3n, pads=[2, 2, 2, 2])   # [1,3,10,10]
    RF = g.op("Conv", fg, W3n, pads=[2, 2, 2, 2])
    n = g.op("ReduceSum", g.op("Mul", RV, C), axes=[2, 3], keepdims=1)  # [1,3,1,1]
    d = g.op("ReduceSum", g.op("Mul", RF, C), axes=[2, 3], keepdims=1)
    vm = g.op("Div", n, d)  # ring color value (true color)

    blk = g.op("Sub", one, fg)  # empty cells (all in-grid)
    missing = g.op("Mul", g.op("Conv", C, W3n, pads=[2, 2, 2, 2]), blk)  # [1,3,10,10]
    addsum = g.op("ReduceSum", g.op("Mul", missing, vm), axes=[1], keepdims=1)
    Vout = g.op("Add", V, addsum)

    Vi = g.op("Cast", Vout, to=6)  # int32
    cvec = g.init(np.arange(1, CH + 1, dtype=np.int32).reshape(1, CH, 1, 1))
    eq = g.op("Equal", Vi, cvec)  # bool [1,10,10,10]
    f = g.op("Cast", eq, to=1)
    g.op("Pad", f, out="output", pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S])
    return g.model()
