"""task022 (137eaa0f) — EXACT ARC-GEN generator semantics.

11x11 input; four 3x3 patterns, each with a gray(5) center (invisible if its
color is 0); the 8 border cells of the output 3x3 are split among the four
patterns. Output 3x3: center gray, border cell (r,c) = color of the owning
pattern (0 if that color is 0). Pattern boxes never overlap.

Net: M[c,k] = sum_p X[c,p] * G5[p+off_k]  (correlation at the 9 offsets)
computed as MatMul( X[10,121], shifted-G5 stack [121,9] ). Channel 0 is then
rebuilt as "no colored channel present". Pad to 30x30.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G


def build():
    S = 11
    g = G(opset=9)
    x = g.op("Slice", "input", axes=[2, 3], starts=[0, 0], ends=[S, S])  # [1,10,11,11]
    g5 = g.op("Slice", x, axes=[1], starts=[5], ends=[6])  # [1,1,11,11]

    K9 = np.zeros((9, 1, 3, 3), dtype=np.float32)
    for k in range(9):
        # M[c,k] = sum_p X[c,p]*G5[p+off]; we need X at gray+off => off_k
        # must be read NEGATED: kernel position (1-dr, 1-dc).
        K9[k, 0, 2 - k // 3, 2 - k % 3] = 1.0
    B = g.op("Conv", g5, g.init(K9), pads=[1, 1, 1, 1])  # [1,9,11,11]
    Bt = g.op("Transpose", B, perm=[0, 2, 3, 1])  # [1,11,11,9]
    B2 = g.op("Reshape", Bt, g.init(np.array([S * S, 9], dtype=np.int64)))
    A = g.op("Reshape", x, g.init(np.array([CH, S * S], dtype=np.int64)))
    M = g.op("MatMul", A, B2)  # [10,9]
    Mr = g.op("Reshape", M, g.init(np.array([1, CH, 3, 3], dtype=np.int64)))

    wsum = np.zeros((1, CH, 1, 1), dtype=np.float32)
    wsum[0, 1:] = 1.0
    Ssum = g.op("Conv", Mr, g.init(wsum))  # [1,1,3,3] any colored
    one = g.init(np.array([1.0], dtype=np.float32))
    Z = g.op("Sub", one, g.op("Sign", Ssum))  # 1 where black
    kvec = np.ones((1, CH, 1, 1), dtype=np.float32)
    kvec[0, 0] = 0.0
    Mk = g.op("Mul", Mr, g.init(kvec))
    w0 = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w0[0, 0, 0, 0] = 1.0
    L = g.op("Conv", Z, g.init(w0))  # lift black into ch0
    R = g.op("Add", Mk, L)
    g.op("Pad", R, out="output", pads=[0, 0, 0, 0, 0, 0, 27, 27])
    return g.model()
