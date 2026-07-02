"""task040 (2204b7a8) — recolor green pixels to the nearer border-line color.

Generator: 10x10; col 0 line colors[0], col 9 line colors[1] (or rows, if
transposed).  Green(3) pixels (never on the lines) become colors[0] if their
coordinate < 5, else colors[1].

Key facts: V(0,0)=colors[0]=:a and V(9,9)=colors[1]=:b in BOTH orientations;
V(0,9)==a iff the lines are horizontal (colors distinct, so exact test).

Net: scalar color grid V; half-plane mask H = horiz?rows<5:cols<5;
Vout = V*(1-G) + (a*H + b*(1-H))*G; expand via Cast+OneHot; Pad.
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    one = g.init(np.array([1.0], dtype=np.float32))

    Xc = g.op("Slice", "input", i64([0, 0]), i64([10, 10]), i64([2, 3]))
    wv = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    V = g.op("Conv", Xc, g.init(wv))  # [1,1,10,10] scalar colors
    Gm = g.op("Gather", Xc, i64([3]), axis=1)  # green mask

    a = g.op("Slice", V, i64([0, 0]), i64([1, 1]), i64([2, 3]))
    b = g.op("Slice", V, i64([9, 9]), i64([10, 10]), i64([2, 3]))
    v09 = g.op("Slice", V, i64([0, 9]), i64([1, 10]), i64([2, 3]))
    horiz = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", v09, a))))

    rowhalf = g.init((np.arange(10) < 5).astype(np.float32).reshape(1, 1, 10, 1))
    colhalf = g.init((np.arange(10) < 5).astype(np.float32).reshape(1, 1, 1, 10))
    nh = g.op("Sub", one, horiz)
    H = g.op("Add", g.op("Mul", horiz, rowhalf), g.op("Mul", nh, colhalf))  # [1,1,10,10]

    newV = g.op("Add", g.op("Mul", a, H), g.op("Mul", b, g.op("Sub", one, H)))
    keep = g.op("Mul", V, g.op("Sub", one, Gm))
    Vout = g.op("Add", keep, g.op("Mul", newV, Gm))

    idx = g.op("Squeeze", Vout, axes=[1])  # float [1,10,10] (exact integers)
    oh = g.op("OneHot", idx, g.init(np.asarray(10.0, dtype=np.float32)),
              g.init(np.array([0.0, 1.0], dtype=np.float32)), axis=1)
    g.op("Pad", oh, out="output", pads=[0, 0, 0, 0, 0, 0, 20, 20])
    return g.model()
