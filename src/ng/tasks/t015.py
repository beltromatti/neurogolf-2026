"""task015 (0ca9ddb6) — 9x9 stars; blue(1) gets orange(7) plus-halo,
red(2) gets yellow(4) diagonal-halo. Generator guarantees halos stay
in-bounds (twinklers scooched off edges) and never overlap stars/halos.

Net: crop 9x9, gather ch1&ch2, one grouped 3x3 Conv producing both halos,
1x1 Conv lifting them into ch7/ch4 and clearing ch0, add, pad back.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    crop = g.op("Slice", "input", i64([0, 0]), i64([9, 9]), i64([2, 3]))
    g12 = g.op("Gather", crop, i64([1, 2]), axis=1)        # [1,2,9,9]
    # grouped conv: ch1 -> plus neighbors, ch2 -> diagonal neighbors
    k = np.zeros((2, 1, 3, 3), dtype=np.float32)
    k[0, 0, 0, 1] = k[0, 0, 2, 1] = k[0, 0, 1, 0] = k[0, 0, 1, 2] = 1.0
    k[1, 0, 0, 0] = k[1, 0, 0, 2] = k[1, 0, 2, 0] = k[1, 0, 2, 2] = 1.0
    halos = g.op("Conv", g12, g.init(k), group=2, pads=[1, 1, 1, 1])
    # lift: halo7 -> ch7, halo4 -> ch4, both clear ch0
    w = np.zeros((CH, 2, 1, 1), dtype=np.float32)
    w[7, 0] = w[4, 1] = 1.0
    w[0, 0] = w[0, 1] = -1.0
    lifted = g.op("Conv", halos, g.init(w))                # [1,10,9,9]
    res = g.op("Add", crop, lifted)
    g.op("Pad", res, out="output", pads=[0, 0, 0, 0, 0, 0, 21, 21])
    return g.model()
