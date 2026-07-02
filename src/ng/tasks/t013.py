"""task013 (0a938d79) — two pixels on the short-axis borders define stripes:
full lines along the short axis at positions start, start+g, start+2g, ...
(g = distance between the two pixels), alternating the two pixel colors,
extending only in the increasing direction. Orientation: stripes run along
the SHORT axis (w in 20..30, h in 6..12, possibly transposed => h>w decides).

Net: blend input with its transpose using t=[h>w] to canonicalize (vertical
stripes); project colors to columns; c0/c1 = min/max pixel columns; period
g=c1-c0; stripe membership via E[k,c]=[c==c0+k*g] matrix; parity gives the
color; expand down rows with the in-grid mask; un-transpose; one-hot.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    one = f32([1.0])

    wv = np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1)
    v = g.op("Conv", "input", g.init(wv))                  # [1,1,30,30]
    occ = g.op("Conv", "input", g.init(np.ones((1, CH, 1, 1), np.float32)))
    vT = g.op("Transpose", v, perm=[0, 1, 3, 2])
    occT = g.op("Transpose", occ, perm=[0, 1, 3, 2])

    # t = [h > w]
    rowocc = g.op("ReduceMax", occ, axes=[3], keepdims=1)  # [1,1,30,1]
    h = g.op("ReduceSum", rowocc, axes=[2], keepdims=1)
    colocc = g.op("ReduceMax", occ, axes=[2], keepdims=1)  # [1,1,1,30]
    w = g.op("ReduceSum", colocc, axes=[3], keepdims=1)
    t = g.op("Relu", g.op("Sign", g.op("Sub", h, w)))      # scalar {0,1}
    u = g.op("Sub", one, t)

    vc = g.op("Add", g.op("Mul", v, u), g.op("Mul", vT, t))
    occC = g.op("Add", g.op("Mul", occ, u), g.op("Mul", occT, t))

    colv = g.op("ReduceSum", vc, axes=[2], keepdims=1)     # [1,1,1,30]
    a = g.op("Sign", colv)
    idx = f32(np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    ia = g.op("Mul", idx, a)
    c1 = g.op("ReduceMax", ia, axes=[3], keepdims=1)
    pen = g.op("Mul", f32([100.0]), g.op("Sub", one, a))
    c0 = g.op("ReduceMin", g.op("Add", ia, pen), axes=[3], keepdims=1)
    gp = g.op("Sub", c1, c0)                               # period, >=2

    # colors at c0 / c1
    e0 = g.op("Relu", g.op("Sub", one, g.op("Abs", g.op("Sub", idx, c0))))
    col0 = g.op("ReduceSum", g.op("Mul", colv, e0), axes=[3], keepdims=1)
    col1 = g.op("Sub", g.op("ReduceSum", colv, axes=[3], keepdims=1), col0)

    # E[k,c] = [c == c0 + k*g]
    kvec = f32(np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
    kpar = f32((np.arange(30) % 2).astype(np.float32).reshape(1, 1, 30, 1))
    base = g.op("Sub", idx, c0)                            # [1,1,1,30]
    D = g.op("Sub", base, g.op("Mul", kvec, gp))           # [1,1,30,30]
    E = g.op("Relu", g.op("Sub", one, g.op("Abs", D)))
    odd = g.op("ReduceSum", g.op("Mul", E, kpar), axes=[2], keepdims=1)
    allm = g.op("ReduceSum", E, axes=[2], keepdims=1)
    even = g.op("Sub", allm, odd)
    outcol = g.op("Add", g.op("Mul", even, col0), g.op("Mul", odd, col1))

    outvC = g.op("Mul", outcol, occC)                      # [1,1,30,30]
    outvT = g.op("Transpose", outvC, perm=[0, 1, 3, 2])
    outv = g.op("Add", g.op("Mul", outvC, u), g.op("Mul", outvT, t))

    cch = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))
    d = g.op("Sub", outv, cch)                             # [1,10,30,30]
    oh = g.op("Relu", g.op("Sub", one, g.op("Abs", d)))
    g.op("Mul", oh, occ, out="output")
    return g.model()
