"""task038 (1fad071e) — count 2x2 blue boxes, output 1x5 bar of blue.

Generator: 9x9 grid with 2x2 boxes (red/blue) that never touch (spacing 1)
plus singletons that never touch same color (8-neighborhood).  Output is a
1x5 row: first `big_blue` cells blue(1), rest black(0).

Net: count 2x2 all-blue windows (each big blue box yields exactly one),
then build the row analytically: ch1[j] = n - j - 0.5, ch0[j] = -(ch1[j]).
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # blue channel, cropped to the 9x9 grid
    B = g.op("Slice", "input", i64([1, 0, 0]), i64([2, 9, 9]), i64([1, 2, 3]))
    # 2x2 all-blue detector
    k = np.ones((1, 1, 2, 2), dtype=np.float32)
    hits = g.op("Relu", g.op("Conv", B, g.init(k), g.init(np.array([-3.0], dtype=np.float32))))
    n = g.op("ReduceSum", hits, keepdims=1)  # [1,1,1,1]
    # ch1[j] = n - (j + 0.5)  (>0 iff j < n) ; ch0 = -ch1
    iota = g.init(np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32).reshape(1, 1, 1, 5))
    ch1 = g.op("Sub", n, iota)
    ch0 = g.op("Neg", ch1)
    cat = g.op("Concat", ch0, ch1, axis=1)  # [1,2,1,5]
    g.op("Pad", cat, out="output", pads=[0, 0, 0, 0, 0, 8, 29, 25])
    return g.model()
