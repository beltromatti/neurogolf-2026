"""task037 (1f876c06) — connect same-color diagonal endpoints.

Generator: 3-6 diagonal segments (length 3..7, direction dr=+1, dc=+/-1) on a
10x10 grid; distinct colors; input shows only the two endpoints, output the
full segment.  Segments never overlap/cross (bitmap check).

Net: per color channel, a cell is on the segment iff it sees a same-color
pixel up-left AND down-right along the main diagonal (or up-right AND
down-left along the anti-diagonal), within 5 steps (interior cells of a
length-7 segment are <=5 from each endpoint; endpoints mark themselves via
the 0-offset).  4 depthwise convs with diagonal kernels + 2 Mul + Add.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    K = 6  # offsets 0..5

    # channels 1..9 cropped to 10x10
    C9 = g.op("Slice", "input", i64([1, 0, 0]), i64([10, 10, 10]), i64([1, 2, 3]))

    main = np.zeros((9, 1, K, K), dtype=np.float32)
    anti = np.zeros((9, 1, K, K), dtype=np.float32)
    for d in range(K):
        main[:, 0, d, d] = 1.0
        anti[:, 0, d, K - 1 - d] = 1.0
    km = g.init(main)
    ka = g.init(anti)
    e = K - 1
    # P1: sum over up-left cells (offset -d,-d), P2: down-right (+d,+d)
    P1 = g.op("Conv", C9, km, pads=[e, e, 0, 0], group=9)
    P2 = g.op("Conv", C9, km, pads=[0, 0, e, e], group=9)
    # P3: up-right (-d,+d), P4: down-left (+d,-d)
    P3 = g.op("Conv", C9, ka, pads=[e, 0, 0, e], group=9)
    P4 = g.op("Conv", C9, ka, pads=[0, e, e, 0], group=9)

    R = g.op("Add", g.op("Mul", P1, P2), g.op("Mul", P3, P4))  # >0 on segments
    any_ = g.op("ReduceSum", R, axes=[1], keepdims=1)
    ch0 = g.op("Sub", g.init(np.array([1.0], dtype=np.float32)), any_)
    cat = g.op("Concat", ch0, R, axis=1)  # [1,10,10,10]
    g.op("Pad", cat, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
