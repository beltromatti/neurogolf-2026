"""Shared helpers: row/col compaction (crop-to-top-left) via MatMul.

All tensors [1,*,30,*]-ish; keep vectors select rows/cols; kept rows/cols are
stacked to the top-left with static shape (TECHNIQUES.md #2).
Note: selection matrices carry factor 0.5 -- output values are 0.25 at kept
cells, which is fine since judging is (output > 0).
"""
from __future__ import annotations

import numpy as np

from ..builder import G


def tri(g: G, strict_lower: bool = True) -> str:
    """[30,30] strictly triangular ones initializer."""
    m = np.tril(np.ones((30, 30), dtype=np.float32), -1)
    if not strict_lower:
        m = m.T
    return g.init(m)


def compact_rows_matrix(g: G, rowkeep: str, L: str) -> str:
    """rowkeep [1,1,30,1] in {0,1} -> R [1,1,30,30], R[a,i]=0.5*[a=dest_i]*keep_i.
    Left-multiply: MatMul(R, X) stacks kept rows at top."""
    dest = g.op("MatMul", L, rowkeep)                 # [1,1,30,1] dest[i]
    destT = g.op("Transpose", dest, perm=[0, 1, 3, 2])  # [1,1,1,30]
    ar = g.init(np.arange(30, dtype=np.float32).reshape(30, 1))
    diff = g.op("Abs", g.op("Sub", ar, destT))        # [1,1,30,30] |a-dest_i|
    thr = g.op("Relu", g.op("Sub", g.init(np.array([0.5], dtype=np.float32)), diff))
    keepT = g.op("Transpose", rowkeep, perm=[0, 1, 3, 2])
    return g.op("Mul", thr, keepT)

def compact_cols_matrix(g: G, colkeep: str, U: str) -> str:
    """colkeep [1,1,1,30] -> Q [1,1,30,30], Q[i,a]=0.5*[a=dest_i]*keep_i.
    Right-multiply: MatMul(X, Q) stacks kept cols at left."""
    dest = g.op("MatMul", colkeep, U)                 # [1,1,1,30]
    destC = g.op("Transpose", dest, perm=[0, 1, 3, 2])  # [1,1,30,1]
    ar = g.init(np.arange(30, dtype=np.float32).reshape(1, 30))
    diff = g.op("Abs", g.op("Sub", destC, ar))        # [1,1,30,30] rows i cols a
    thr = g.op("Relu", g.op("Sub", g.init(np.array([0.5], dtype=np.float32)), diff))
    keepC = g.op("Transpose", colkeep, perm=[0, 1, 3, 2])
    return g.op("Mul", thr, keepC)
