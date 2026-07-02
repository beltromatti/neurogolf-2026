"""task024 (178fcbfb) — EXACT ARC-GEN generator semantics.

Pixels of colors 2 (red), 1 (blue), 3 (green); rows of pixels all DISTINCT.
Output: full column red through each red pixel; full row blue/green through
each blue/green pixel. Write order red,blue,green => green row > blue row >
red col > black. Rows distinct => green/blue rows disjoint.

Net (cropped to 15x15, grids are 6..15):
  Gmask = rowhas3*occ ; Bmask = rowhas1*occ ;
  Rmask = colhas2*(1-rowhas1-rowhas3)*occ ; ch0 = occ-G-B-R
  Concat -> Conv 1x1 scatter to channels -> Pad.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    S = 15
    g = G(opset=9)
    x = g.op("Slice", "input", axes=[2, 3], starts=[0, 0], ends=[S, S])  # [1,10,15,15]
    occ = g.op("Conv", x, g.init(np.ones((1, CH, 1, 1), dtype=np.float32)))
    ch = lambda c: g.op("Slice", x, axes=[1], starts=[c], ends=[c + 1])
    h1 = g.op("ReduceMax", ch(1), axes=[3], keepdims=1)  # [1,1,15,1]
    h3 = g.op("ReduceMax", ch(3), axes=[3], keepdims=1)
    v2 = g.op("ReduceMax", ch(2), axes=[2], keepdims=1)  # [1,1,1,15]
    Gm = g.op("Mul", h3, occ)
    Bm = g.op("Mul", h1, occ)
    one = g.init(np.array([1.0], dtype=np.float32))
    nh = g.op("Sub", one, g.op("Add", h1, h3))  # [1,1,15,1]
    Rm = g.op("Mul", g.op("Mul", v2, nh), occ)
    ch0 = g.op("Sub", g.op("Sub", g.op("Sub", occ, Gm), Bm), Rm)
    cat = g.op("Concat", ch0, Bm, Rm, Gm, axis=1)  # [1,4,15,15]
    w = np.zeros((CH, 4, 1, 1), dtype=np.float32)
    w[0, 0] = w[1, 1] = w[2, 2] = w[3, 3] = 1.0
    out = g.op("Conv", cat, g.init(w))  # [1,10,15,15]
    g.op("Pad", out, out="output", pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S])
    return g.model()
