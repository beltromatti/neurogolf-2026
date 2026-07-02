"""task012 (0962bcdd) — 12x12; two "flowers": center color0 with 4-plus of
color1. Output: center + diagonal rays (dist 1,2) in color0, orthogonal rays
(dist 1,2) in color1. apply_gravity is a grid symmetry and the star pattern
is symmetric, so it's irrelevant.

Net: scalar values; centers = fg cells with 4 fg ortho-neighbors; center
color c0 = v at center; arm color c1 = v of up-neighbor (always an arm);
rays via fixed 5x5 convs from center; one-hot compare; pad.
Flowers never overlap each other or the border (r in {2,8}, c in 3..9,
symmetries thereof).
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    wv = np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1)
    v30 = g.op("Conv", "input", g.init(wv))                # [1,1,30,30]
    v = g.op("Slice", v30, i64([0, 0]), i64([12, 12]), i64([2, 3]))
    fg = g.op("Sign", v)                                   # [1,1,12,12]

    kplus = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kplus[0, 0, [0, 2, 1, 1], [1, 1, 0, 2]] = 1.0
    nb = g.op("Conv", fg, g.init(kplus), f32([-3.0]), pads=[1, 1, 1, 1])
    M = g.op("Relu", nb)                                   # centers {0,1}

    c0 = g.op("Mul", v, M)                                 # c0 values at centers
    A = g.op("Mul", v, g.op("Sub", fg, M))                 # arm values
    kup = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kup[0, 0, 0, 1] = 1.0                                  # reads (r-1,c)
    c1 = g.op("Mul", g.op("Conv", A, g.init(kup), pads=[1, 1, 1, 1]), M)

    kdiag = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for s in (1, 2):
        kdiag[0, 0, [2 - s, 2 - s, 2 + s, 2 + s], [2 - s, 2 + s, 2 - s, 2 + s]] = 1.0
    diag = g.op("Conv", c0, g.init(kdiag), pads=[2, 2, 2, 2])
    korth = np.zeros((1, 1, 5, 5), dtype=np.float32)
    for s in (1, 2):
        korth[0, 0, [2 - s, 2 + s, 2, 2], [2, 2, 2 - s, 2 + s]] = 1.0
    orth = g.op("Conv", c1, g.init(korth), pads=[2, 2, 2, 2])

    outv = g.op("Add", g.op("Add", c0, diag), orth)        # disjoint positions

    cch = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))
    d = g.op("Sub", outv, cch)                             # [1,10,12,12]
    oh = g.op("Relu", g.op("Sub", f32([1.0]), g.op("Abs", d)))
    g.op("Pad", oh, out="output", pads=[0, 0, 0, 0, 0, 0, 18, 18])
    return g.model()
