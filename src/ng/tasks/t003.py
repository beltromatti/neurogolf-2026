"""task003 (017c7c7b) — extend periodic pattern 6x3 -> 9x3, blue->red.

Generator: pattern period p in {2,3,4} (steps 2/3, optional flip for
steps=2 giving p=4). Rows 6..8 of the output continue the pattern:
  if rows0..2 == rows3..5 (true iff p==3): rows6..8 = rows3..5
  else (p in {2,4}):                       rows6..8 = rows2..4
(verified against generator semantics for every case, incl. flip).
Recolor 1 -> 2.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # blue channel, rows 0..5, cols 0..2 -> [1,1,6,3]
    b = g.op("Slice", "input", i64([1, 0, 0]), i64([2, 6, 3]), i64([1, 2, 3]))
    top = g.op("Slice", b, i64([0]), i64([3]), i64([2]))
    bot = g.op("Slice", b, i64([3]), i64([6]), i64([2]))
    d = g.op("Abs", g.op("Sub", top, bot))
    s = g.op("ReduceSum", d, keepdims=1)      # [1,1,1,1]
    g1 = g.op("Sign", s)                       # 1 -> p!=3 -> use rows2..4

    b234 = g.op("Slice", b, i64([2]), i64([5]), i64([2]))
    diff = g.op("Sub", b234, bot)
    tail = g.op("Add", bot, g.op("Mul", g1, diff))  # bot + g1*(b234-bot)
    pat = g.op("Concat", b, tail, axis=2)      # [1,1,9,3]

    w = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w[0, 0, 0, 0] = -1.0
    w[2, 0, 0, 0] = 1.0
    bias = np.zeros(CH, dtype=np.float32)
    bias[0] = 1.0
    out10 = g.op("Conv", pat, g.init(w), g.init(bias))  # [1,10,9,3]
    g.op("Pad", out10, out="output", pads=[0, 0, 0, 0, 0, 0, 21, 27])
    return g.model()
