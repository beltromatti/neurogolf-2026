"""task202 (855e0971) — EXACT ARC-GEN generator semantics.

Generator: grid = horizontal strata (bands of rows), each a DISTINCT color
(common.random_colors). Black pixels: each in a distinct row AND distinct
column (cols sampled globally without replacement; rows disjoint via strata
floors). Output: each black pixel's column is blacked across its whole
stratum. xpose=1 transposes everything (vertical strata, rows filled).

Compilation (all tensors small; cond via outer Equal + Where):
  rowval(r)  = 10 + stratum color of row r     (ReduceMax over W + 1x1 Conv)
  colvalC(c) = 10 + stratum color of column c  (transposed analogue)
  posC(c) = 1 + row of black pixel in col c, 0 if none
            (single Conv, kernel [1,10,30,1] with (r+1) on the black channel)
  posR(r) = 1 + col of black pixel in row r, 0 if none
  colval(c)  = [sentinel, rowval][posC(c)]   (runtime Gather, sentinel=-5)
  rowvalB(r) = [sentinel, colvalC][posR(r)]
  x = [some row has >=2 colors]  (rows are monochromatic iff xpose=0)
  rowfac = Where(x, rowvalB, rowval); colfac = Where(x, colvalC, colval)
  cond(r,c) = rowfac(r) == colfac(c)  (int32 Equal, outer broadcast -> bool)
  output = Where(cond, e_black, input)
cost ~6000 (memory ~5350 + params ~645)
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G, H, W


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # per-row / per-column color presence
    rowhas = g.op("ReduceMax", "input", axes=[3], keepdims=1)  # [1,10,30,1]
    colhas = g.op("ReduceMax", "input", axes=[2], keepdims=1)  # [1,10,1,30]

    # stratum color value (+10 so 'outside grid' rows/cols never collide)
    wk = f32(np.arange(CH, dtype=np.float32).reshape(1, CH, 1, 1))  # [0,1..9]
    b10 = f32([10.0])
    rowval = g.op("Conv", rowhas, wk, b10)    # [1,1,30,1]
    colvalC = g.op("Conv", colhas, wk, b10)   # [1,1,1,30]

    # orientation: x=1 iff some row has >=2 (non-black) colors  => xpose=1
    w01 = np.ones((1, CH, 1, 1), dtype=np.float32)
    w01[0, 0] = 0.0
    nc = g.op("Conv", rowhas, f32(w01))                       # [1,1,30,1]
    xb = g.op("Greater", g.op("ReduceMax", nc, keepdims=1), f32([1.5]))  # bool

    # black-pixel positions (1-based; 0 = none), via channel+space Conv
    k1 = np.zeros((1, CH, H, 1), dtype=np.float32)
    k1[0, 0, :, 0] = np.arange(1, H + 1)
    posC = g.op("Conv", "input", f32(k1))                     # [1,1,1,30]
    k2 = np.zeros((1, CH, 1, W), dtype=np.float32)
    k2[0, 0, 0, :] = np.arange(1, W + 1)
    posR = g.op("Conv", "input", f32(k2))                     # [1,1,30,1]

    sent = f32(np.full((1, 1, 1, 1), -5.0))
    to_i = lambda t: g.op("Cast", t, to=6)  # int32

    # colval(c) = rowval of black pixel's row in column c (or sentinel)
    padR = g.op("Concat", sent, rowval, axis=2)               # [1,1,31,1]
    gC = g.op("Gather", padR, to_i(posC), axis=2)             # rank-7, 30 elems
    colval = g.op("Reshape", gC, i64([1, 1, 1, W]))           # [1,1,1,30]
    # rowvalB(r) = colvalC of black pixel's column in row r (or sentinel)
    padC = g.op("Concat", sent, colvalC, axis=3)              # [1,1,1,31]
    gR = g.op("Gather", padC, to_i(posR), axis=3)             # rank-7, 30 elems
    rowvalB = g.op("Reshape", gR, i64([1, 1, H, 1]))          # [1,1,30,1]

    # branch select on tiny tensors
    rowfac = g.op("Where", xb, rowvalB, rowval)               # [1,1,30,1]
    colfac = g.op("Where", xb, colvalC, colval)               # [1,1,1,30]

    # outer equality -> bool [1,1,30,30]; paint black there
    cond = g.op("Equal", to_i(rowfac), to_i(colfac))
    e0 = np.zeros((1, CH, 1, 1), dtype=np.float32)
    e0[0, 0] = 1.0
    g.op("Where", cond, f32(e0), "input", out="output")
    return g.model()
