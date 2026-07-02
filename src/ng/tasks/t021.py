"""task021 (1190e5a7) — EXACT ARC-GEN generator semantics.

Input: bg color grid cut by full horizontal/vertical lines of a 2nd color.
Row 0 / col 0 never contain a line-row/line-col (thicknesses >= 1), so:
  bg = color at (0,0)
  #hlines = number of in-grid rows r where cell (r,0) != bg
  #vlines = number of in-grid cols c where cell (0,c) != bg
Output = canvas(bg) of shape (#hlines+1, #vlines+1) at top-left.

Net: count non-bg cells along col 0 / row 0; build row/col masks via
Less(coord-1, count); output = rowmask*colmask*bgvec.  Opset 9 (free Slice).
"""
from __future__ import annotations

import numpy as np

from ..builder import G, H, W


def build():
    g = G(opset=9)
    bg = g.op("Slice", "input", axes=[2, 3], starts=[0, 0], ends=[1, 1])  # [1,10,1,1]
    col0 = g.op("Slice", "input", axes=[3], starts=[0], ends=[1])  # [1,10,30,1]
    row0 = g.op("Slice", "input", axes=[2], starts=[0], ends=[1])  # [1,10,1,30]

    def count_lines(vec):
        # lines = sum over positions of (occupied - is_bg)
        isbg = g.op("ReduceSum", g.op("Mul", vec, bg), axes=[1], keepdims=1)
        occ = g.op("ReduceSum", vec, axes=[1], keepdims=1)
        d = g.op("Sub", occ, isbg)
        return g.op("ReduceSum", d, axes=[2, 3], keepdims=1)  # [1,1,1,1]

    hl = count_lines(col0)
    wl = count_lines(row0)
    # rowmask[r] = 1[r < hl+1] = Less(r-1, hl)
    ar = g.init(np.arange(-1, H - 1, dtype=np.float32).reshape(1, 1, H, 1))
    ac = g.init(np.arange(-1, W - 1, dtype=np.float32).reshape(1, 1, 1, W))
    rm = g.op("Cast", g.op("Less", ar, hl), to=1)  # float [1,1,30,1]
    cm = g.op("Cast", g.op("Less", ac, wl), to=1)  # float [1,1,1,30]
    mask = g.op("Mul", rm, cm)  # [1,1,30,30]
    g.op("Mul", mask, bg, out="output")
    return g.model()
