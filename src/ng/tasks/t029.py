"""task029 (1c786137) — zoom: crop the interior of the zoom_color rectangle.

Generator: random static of 3-4 colors; a rectangle outline of zoom_color
(a color not used by the static) around a zoom window; output = window
content (static + black).

Ring-color detection: a present color whose cells NEVER lie strictly inside
its own bounding box (static colors virtually always do).  Zoom window =
rows/cols strictly inside the ring bbox.  Crop = compaction MatMuls; black
cells travel through channel 0 automatically.
"""
from __future__ import annotations

import numpy as np

from ..builder import G
from ._compact import compact_cols_matrix, compact_rows_matrix, tri


def build():
    g = G()
    L = tri(g, True)   # L[i,j]=1 iff j<i
    U = tri(g, False)  # U[i,j]=1 iff i<j

    R = g.op("ReduceMax", "input", axes=[3], keepdims=1)   # [1,10,30,1] rows with c
    C = g.op("ReduceMax", "input", axes=[2], keepdims=1)   # [1,10,1,30] cols with c

    aboveR = g.op("Sign", g.op("MatMul", L, R))            # strict prefix-OR down
    belowR = g.op("Sign", g.op("MatMul", U, R))
    rowin = g.op("Mul", aboveR, belowR)                    # rows strictly inside bbox
    aboveC = g.op("Sign", g.op("MatMul", C, U))
    belowC = g.op("Sign", g.op("MatMul", C, L))
    colin = g.op("Mul", aboveC, belowC)                    # [1,10,1,30]

    # count of c-cells strictly inside own bbox: rowin^T @ X @ colin (contracted)
    rowinT = g.op("Transpose", rowin, perm=[0, 1, 3, 2])   # [1,10,1,30]
    mid = g.op("MatMul", rowinT, "input")                  # [1,10,1,30]
    inside = g.op("ReduceSum", g.op("Mul", mid, colin), axes=[3], keepdims=1)

    present = g.op("ReduceMax", R, axes=[2], keepdims=1)   # [1,10,1,1]
    one = g.init(np.array([1.0], dtype=np.float32))
    valid = g.op("Mul", present, g.op("Sub", one, g.op("Sign", inside)))

    rowkeep = g.op("ReduceSum", g.op("Mul", valid, rowin), axes=[1], keepdims=1)
    colkeep = g.op("ReduceSum", g.op("Mul", valid, colin), axes=[1], keepdims=1)

    Rr = compact_rows_matrix(g, rowkeep, L)
    Qc = compact_cols_matrix(g, colkeep, U)
    y1 = g.op("MatMul", Rr, "input")
    g.op("MatMul", y1, Qc, out="output")
    return g.model()
