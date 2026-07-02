"""task019 (10fcaaa3) — EXACT ARC-GEN generator semantics.

Input h x w (2..6 each) with colored pixels (color != cyan). Output 2h x 2w =
input tiled 2x2; cyan(8) at every diagonal neighbor (in-bounds, no wrap) of
any tiled colored cell, then colors overwrite cyan.

Net (12x12 work region): h,w counted from occupancy; tile via
MatMul(Arow, X) and MatMul(X, Acol) where Arow = I + 1[a-b==h] built from a
constant difference matrix D[a,b]=a-b:  Arow = relu(1-|D-h|) + I,
Acol = relu(1-|D+w|) + I.  Cyan = X-dilate(fg) on black-in-grid cells;
lift +ch8/-ch0; Pad back.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    S = 12
    g = G(opset=9)
    x = g.op("Slice", "input", axes=[2, 3], starts=[0, 0], ends=[S, S])  # [1,10,12,12]
    ones = g.init(np.ones((1, CH, 1, 1), dtype=np.float32))
    occ = g.op("Conv", x, ones)  # [1,1,12,12]
    h = g.op("ReduceSum", g.op("Slice", occ, axes=[3], starts=[0], ends=[1]),
             axes=[2, 3], keepdims=1)  # [1,1,1,1]
    w = g.op("ReduceSum", g.op("Slice", occ, axes=[2], starts=[0], ends=[1]),
             axes=[2, 3], keepdims=1)

    a = np.arange(S, dtype=np.float32)
    D = g.init((a[:, None] - a[None, :]).reshape(1, 1, S, S))
    I = g.init(np.eye(S, dtype=np.float32).reshape(1, 1, S, S))
    one = g.init(np.array([1.0], dtype=np.float32))

    def shift_plus_id(diff):  # relu(1 - |diff|) + I
        return g.op("Add", g.op("Relu", g.op("Sub", one, g.op("Abs", diff))), I)

    Arow = shift_plus_id(g.op("Sub", D, h))   # 1[a-b in {0,h}]
    Acol = shift_plus_id(g.op("Add", D, w))   # 1[b-a in {0,w}]
    T = g.op("MatMul", Arow, x)               # [1,10,12,12]
    T2 = g.op("MatMul", T, Acol)              # tiled 2x2

    occ2 = g.op("Conv", T2, ones)
    wfg = np.zeros((1, CH, 1, 1), dtype=np.float32)
    wfg[0, 1:] = 1.0
    fg = g.op("Conv", T2, g.init(wfg))
    kx = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kx[0, 0, 0, 0] = kx[0, 0, 0, 2] = kx[0, 0, 2, 0] = kx[0, 0, 2, 2] = 1.0
    dil = g.op("Conv", fg, g.init(kx), pads=[1, 1, 1, 1])
    black = g.op("Sub", occ2, fg)
    cyan = g.op("Mul", dil, black)  # >0 where cyan should go
    wl = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    wl[8, 0, 0, 0] = 1.0
    wl[0, 0, 0, 0] = -1.0
    lift = g.op("Conv", cyan, g.init(wl))
    R = g.op("Add", T2, lift)
    g.op("Pad", R, out="output", pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S])
    return g.model()
