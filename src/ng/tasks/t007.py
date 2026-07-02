"""task007 (05269061) — complete diagonal stripes, 7x7.

Generator: 3 colors, cell color = colors[(r+c) % 3]; input reveals each
color on exactly one anti-diagonal (one diag per residue class).
Output: full stripes.

Compile: class-sums via MatMul with D[49,3] (one color per class ->
sum works, no max needed), then scatter back with a 2D Gather of
indices (r+c)%3.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # channels 1..9, 7x7 -> [1,9,7,7]
    x = g.op("Slice", "input", i64([1, 0, 0]), i64([10, 7, 7]), i64([1, 2, 3]))
    xr = g.op("Reshape", x, i64([1, 9, 49]))

    D = np.zeros((49, 3), dtype=np.float32)
    for i in range(49):
        D[i, (i // 7 + i % 7) % 3] = 1.0
    cv = g.op("MatMul", xr, g.init(D))  # [1,9,3] per-channel class counts

    idx = np.array([[(r + c) % 3 for c in range(7)] for r in range(7)],
                   dtype=np.int64)
    out9 = g.op("Gather", cv, g.init(idx), axis=2)  # [1,9,7,7]
    g.op("Pad", out9, out="output", pads=[0, 1, 0, 0, 0, 0, 23, 23])
    return g.model()
