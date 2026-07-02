"""task006 (0520fde7) — intersect halves.

Generator (grid_intersect): input 3x7, blue pixels in left (cols 0..2) and
right (cols 4..6) halves, gray divider col 3. Output 3x3: red(2) where
left AND right are blue, black elsewhere.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # blue channel, rows 0..2, cols 0..6 -> [1,1,3,7]
    b = g.op("Slice", "input", i64([1, 0, 0]), i64([2, 3, 7]), i64([1, 2, 3]))
    l = g.op("Slice", b, i64([0]), i64([3]), i64([3]))
    r = g.op("Slice", b, i64([4]), i64([7]), i64([3]))
    both = g.op("Mul", l, r)  # [1,1,3,3] in {0,1}

    w = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = -1.0
    w[2, 0, 0, 0] = 1.0
    bias = np.zeros(CH, dtype=np.float32)
    bias[0] = 1.0
    out10 = g.op("Conv", both, g.init(w), g.init(bias))  # [1,10,3,3]
    g.op("Pad", out10, out="output", pads=[0, 0, 0, 0, 0, 0, 27, 27])
    return g.model()
