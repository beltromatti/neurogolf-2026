"""task031 (1cf80156) — crop to the bounding box of the single colored blob.

Blob is 4-connected, so rows/cols containing color are exactly the bbox
rows/cols: rowkeep = any-color-in-row.  Crop = compaction MatMuls; interior
black cells travel through channel 0.
"""
from __future__ import annotations

import numpy as np

from ..builder import G
from ._compact import compact_cols_matrix, compact_rows_matrix, tri


def build():
    g = G()
    w = np.zeros((1, 10, 1, 1), dtype=np.float32)
    w[0, 1:] = 1.0
    fg = g.op("Conv", "input", g.init(w))                     # [1,1,30,30]
    rowkeep = g.op("ReduceMax", fg, axes=[3], keepdims=1)     # [1,1,30,1]
    colkeep = g.op("ReduceMax", fg, axes=[2], keepdims=1)     # [1,1,1,30]

    Rr = compact_rows_matrix(g, rowkeep, tri(g, True))
    Qc = compact_cols_matrix(g, colkeep, tri(g, False))
    y1 = g.op("MatMul", Rr, "input")
    g.op("MatMul", y1, Qc, out="output")
    return g.model()
