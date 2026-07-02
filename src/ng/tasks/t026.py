"""task026 (1b2d62fb) — EXACT ARC-GEN generator semantics.

Generator: grid_intersect(width=3, height=5, ...): input ALWAYS 5x7,
col 3 = blue separator, halves cells are maroon(9) or black(0).
Output 5x3: cyan(8) where BOTH left cell (r,c) and right cell (r,c+4)
are black; else black(0).

Compile: A = ch0[0:5,0:3], B = ch0[0:5,4:7]; and = A*B;
Conv 1->10 (w8=+1, w0=-1, bias ch0=+1) ; Pad to 30x30.
Uses opset 9 so Slice/Pad take attributes (free params).
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G(opset=9)
    # channel 0, left half rows 0..4 cols 0..2 / right half cols 4..6
    a = g.op("Slice", "input", axes=[1, 2, 3], starts=[0, 0, 0], ends=[1, 5, 3])
    b = g.op("Slice", "input", axes=[1, 2, 3], starts=[0, 0, 4], ends=[1, 5, 7])
    m = g.op("Mul", a, b)  # [1,1,5,3] both-black
    w = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w[8, 0, 0, 0] = 1.0
    w[0, 0, 0, 0] = -1.0
    bias = np.zeros(CH, dtype=np.float32)
    bias[0] = 1.0
    c = g.op("Conv", m, g.init(w), g.init(bias))  # [1,10,5,3]
    g.op("Pad", c, out="output", pads=[0, 0, 0, 0, 0, 0, 25, 27])
    return g.model()
