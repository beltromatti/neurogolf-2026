"""task004 (025d127b) — slanted-box cells shift right, except bottom edge
and the right-edge clamp cell.

From the generator: shapes occupy disjoint row bands separated by an
empty row. Per colored cell:
  STAY iff (the grid row below is entirely empty)          # bottom edge
        or (cell below AND cell below-left are colored)    # clamp cell
  else move right by 1.
Vacated cells become black; grids are <=16x16 -> 16x16 crop.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G

N = 16


def build():
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))

    X = g.op("Slice", "input", i64([0, 0, 0]), i64([CH, N, N]), i64([1, 2, 3]))

    wfg = np.zeros((1, CH, 1, 1), dtype=np.float32)
    wfg[0, 1:, 0, 0] = 1.0
    f = g.op("Conv", X, g.init(wfg))                 # colored mask
    occ = g.op("Conv", X, g.init(np.ones((1, CH, 1, 1), dtype=np.float32)))

    # bn[r] = 1 iff row r+1 has any colored cell
    rowsum = g.op("ReduceSum", f, axes=[3], keepdims=1)     # [1,1,16,1]
    kup = np.zeros((1, 1, 3, 1), dtype=np.float32)
    kup[0, 0, 2, 0] = 1.0                                    # reads r+1
    bn = g.op("Sign", g.op("Conv", rowsum, g.init(kup), pads=[1, 0, 1, 0]))

    # below / below-left colored
    D = g.op("Conv", f, g.init(kup), pads=[1, 0, 1, 0])      # f[r+1,c]
    kdl = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kdl[0, 0, 2, 0] = 1.0                                    # f[r+1,c-1]
    DL = g.op("Conv", f, g.init(kdl), pads=[1, 1, 1, 1])
    clamp = g.op("Mul", g.op("Mul", f, D), DL)

    one = f32([1.0])
    staybot = g.op("Mul", f, g.op("Sub", one, bn))
    stay = g.op("Add", staybot, clamp)                       # disjoint -> {0,1}
    move = g.op("Sub", f, stay)

    # move colors right by one; keep stayers
    Xm = g.op("Mul", X, move)
    kr = np.zeros((CH, 1, 1, 3), dtype=np.float32)
    kr[:, 0, 0, 0] = 1.0                                     # out[c] = x[c-1]
    XmR = g.op("Conv", Xm, g.init(kr), pads=[0, 1, 0, 1], group=CH)
    Xs = g.op("Mul", X, stay)
    colored = g.op("Add", XmR, Xs)

    # black channel: occupied and not colored-after
    kr1 = np.zeros((1, 1, 1, 3), dtype=np.float32)
    kr1[0, 0, 0, 0] = 1.0
    mR = g.op("Conv", move, g.init(kr1), pads=[0, 1, 0, 1])
    ofg = g.op("Add", mR, stay)
    ch0 = g.op("Sub", occ, ofg)
    w0 = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w0[0, 0, 0, 0] = 1.0
    lift = g.op("Conv", ch0, g.init(w0))                     # [1,10,16,16]
    out10 = g.op("Add", colored, lift)
    g.op("Pad", out10, out="output",
         pads=[0, 0, 0, 0, 0, 0, 30 - N, 30 - N])
    return g.model()
