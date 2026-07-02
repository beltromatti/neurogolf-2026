"""task010 (08ed6ac7) — 9x9 grid, four gray(5) bars at odd columns.

Generator: 4 distinct heights (1..9), bar `bar` gets color bar+1 where bars
are indexed in descending height order => color = 1 + #(bars strictly taller).

Net: column heights via ReduceSum, pairwise-greater counts via broadcasted
Sub/Sign, one-hot of (count+1) via |cnt - (c-1)| compare, paint on gray mask.
Grid is always 9x9 -> crop with Slice, Pad back.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    g = G()
    f32 = lambda v: g.init(np.asarray(v, dtype=np.float32))
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # crop to 9x9 (Slice-10: starts/ends/axes are inputs)
    crop = g.op("Slice", "input", i64([0, 0]), i64([9, 9]), i64([2, 3]))
    ch5 = g.op("Gather", crop, i64([5]), axis=1)          # [1,1,9,9] gray mask
    h = g.op("ReduceSum", ch5, axes=[2], keepdims=1)       # [1,1,1,9] heights
    hT = g.op("Transpose", h, perm=[0, 1, 3, 2])           # [1,1,9,1]
    d = g.op("Sub", hT, h)                                 # d[k,j] = h_k - h_j
    gt = g.op("Relu", g.op("Sign", d))                     # 1 iff h_k > h_j
    cnt = g.op("ReduceSum", gt, axes=[2], keepdims=1)      # [1,1,1,9]
    # one-hot: channel c fires iff cnt == c-1  (rank = cnt+1)
    cvec = f32(np.arange(-1, 9, dtype=np.float32).reshape(1, CH, 1, 1))
    e = g.op("Sub", cnt, cvec)                             # [1,10,1,9]
    oh = g.op("Relu", g.op("Sub", f32([1.0]), g.op("Abs", e)))
    bars = g.op("Mul", oh, ch5)                            # [1,10,9,9]
    # background channel 0 = in-grid & not bar
    occ = g.op("ReduceSum", crop, axes=[1], keepdims=1)    # [1,1,9,9]
    bg = g.op("Sub", occ, ch5)
    bg10 = g.op("Pad", bg, pads=[0, 0, 0, 0, 0, 9, 0, 0])  # lift into ch0
    res = g.op("Add", bars, bg10)
    g.op("Pad", res, out="output", pads=[0, 0, 0, 0, 0, 0, 21, 21])
    return g.model()
