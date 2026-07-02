"""task025 (1a07d186) — EXACT ARC-GEN generator semantics.

Full lines (vertical, or horizontal if transposed) of distinct colors;
scattered pixels. Output = lines only + each pixel whose color matches a
line moved to the adjacent column of THAT line on the pixel's side; pixels
of non-line colors vanish. At most one pixel per (color,row); pixels never
on/adjacent to a line column.

Net: same vertical-line branch applied to X and X^T (the wrong-orientation
branch finds no lines and contributes nothing):
  line[c,col]  = column is full of color c  (count==height, c>=1)
  rowpix[c,r]  = pixel present (row-dot with not-line-cols mask)
  colpos, m    = pixel / line column indices via MatMul with [0..29]
  left/right   = side; Moved = [left,right] @ [line shifted +-1]
  out = lines*occ + Moved (+ ch0 = occ - fg).
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G, H, W


def build():
    g = G(opset=9)
    ones = g.init(np.ones((1, CH, 1, 1), dtype=np.float32))
    one = g.init(np.array([1.0], dtype=np.float32))
    cmask = np.ones((1, CH, 1, 1), dtype=np.float32)
    cmask[0, 0] = 0.0
    cmaskn = g.init(cmask)
    colc = g.init(np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
    colv = g.init(np.arange(W, dtype=np.float32).reshape(W, 1))
    idxL = g.init(np.array(list(range(1, W)) + [0], dtype=np.int64))
    idxR = g.init(np.array([0] + list(range(W - 1)), dtype=np.int64))
    sh2 = g.init(np.array([W, 1], dtype=np.int64))

    def vbranch(X):
        occ = g.op("Conv", X, ones)                        # [1,1,30,30]
        hgt = g.op("ReduceSum", occ, axes=[2], keepdims=1)  # [1,1,1,30]
        cnt = g.op("ReduceSum", X, axes=[2], keepdims=1)    # [1,10,1,30]
        l0 = g.op("Relu", g.op("Add", g.op("Sub", cnt, hgt), one))
        line = g.op("Mul", g.op("Mul", l0, cmaskn), g.op("Sign", hgt))
        L = g.op("Mul", line, occ)                          # [1,10,30,30]
        notline = g.op("Sub", one, g.op("ReduceSum", line, axes=[1], keepdims=1))
        nl = g.op("Reshape", notline, sh2)                  # [30,1]
        cw = g.op("Reshape", g.op("Mul", notline, colc), sh2)
        rowpix = g.op("MatMul", X, nl)                      # [1,10,30,1]
        colpos = g.op("MatMul", X, cw)                      # [1,10,30,1]
        m = g.op("MatMul", line, colv)                      # [1,10,1,1]
        left = g.op("Mul", g.op("Relu", g.op("Sign", g.op("Sub", m, colpos))), rowpix)
        right = g.op("Sub", rowpix, left)
        lineL = g.op("Gather", line, idxL, axis=3)          # target m-1
        lineR = g.op("Gather", line, idxR, axis=3)          # target m+1
        LR = g.op("Concat", left, right, axis=3)            # [1,10,30,2]
        L2 = g.op("Concat", lineL, lineR, axis=2)           # [1,10,2,30]
        moved = g.op("MatMul", LR, L2)                      # [1,10,30,30]
        return g.op("Add", L, moved), occ

    b1, occ = vbranch("input")
    xt = g.op("Transpose", "input", perm=[0, 1, 3, 2])
    b2, _ = vbranch(xt)
    b2t = g.op("Transpose", b2, perm=[0, 1, 3, 2])
    fg = g.op("Add", b1, b2t)
    anyfg = g.op("Conv", fg, ones)
    ch0 = g.op("Sub", occ, anyfg)
    w0 = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w0[0, 0, 0, 0] = 1.0
    lift = g.op("Conv", ch0, g.init(w0))
    g.op("Add", fg, lift, out="output")
    return g.model()
