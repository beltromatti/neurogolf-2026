"""task002 (00d62c1b) — EXACT ARC-GEN generator semantics.

Generator (task_00d62c1b.py):
1. Pots: green rectangle rings (wide,tall in 3..8) with black interior ->
   interior filled yellow(4).
2. Raster cascade: black cell -> yellow if all 4 neighbors nonblack, where
   up/left neighbors see the *updated* grid (pots already filled), right/down
   see original-after-pot-fill. In-place raster == least fixpoint of
   Y <- B2 & U(N0|Y) & L(N0|Y), with B2 = black & R(N0) & D(N0).

Tensors: N0 = fg(1..9) | potInterior; cascade iterated `casc` times.
"""
from __future__ import annotations

import numpy as np

from ..builder import CH, G, H, W


def _shift_kernel(dr: int, dc: int) -> np.ndarray:
    """3x3 kernel reading neighbor at (dr,dc)."""
    k = np.zeros((1, 1, 3, 3), dtype=np.float32)
    k[0, 0, 1 + dr, 1 + dc] = 1.0
    return k


def build(casc: int = 45):
    g = G()
    i64 = lambda v: g.init(np.asarray(v, dtype=np.int64))

    # --- base masks ------------------------------------------------------
    # fg = any of ch1..9 ; ch0 = black-inside-grid ; both [1,1,30,30]
    wfg = np.zeros((1, CH, 1, 1), dtype=np.float32)
    wfg[0, 1:, 0, 0] = 1.0
    fg = g.op("Conv", "input", g.init(wfg))
    ch0 = g.op("Gather", "input", i64([0]), axis=1)
    # green channel (color 3)
    ch3 = g.op("Gather", "input", i64([3]), axis=1)
    # gb = 2-channel [green, black] for template matching
    gb = g.op("Concat", ch3, ch0, axis=1)

    # --- pot interiors via exact ring templates --------------------------
    # interior sizes 1..6 x 1..6 (tall,wide in 3..8); skip 1x1 (cascade fills).
    # NOTE: rings between adjacent pots create ~2.7%/pair irreducible
    # ambiguity (identical local pattern, different label) — accepted risk.
    pot_parts = []
    for ih in range(1, 7):
        for iw in range(1, 7):
            if ih == 1 and iw == 1:
                continue
            kh, kw = ih + 2, iw + 2
            k = np.zeros((1, 2, kh, kw), dtype=np.float32)
            # ring edges on green channel; corners are DON'T CARE (generator
            # draws pot rings without corner cells)
            k[0, 0, 0, 1:-1] = k[0, 0, -1, 1:-1] = 1.0
            k[0, 0, 1:-1, 0] = k[0, 0, 1:-1, -1] = 1.0
            # interior on black channel
            k[0, 1, 1:-1, 1:-1] = 1.0
            need = float(k.sum())
            # conv valid-anchor at top-left of ring
            c = g.op("Conv", gb, g.init(k), pads=[0, 0, kh - 1, kw - 1])
            # match: c == need  ->  relu(c - (need-1)) in {0,1}
            hit = g.op("Relu", g.op("Sub", c, g.init(np.array([need - 1.0], dtype=np.float32))))
            # scatter to interior cells: transpose-conv style via Conv over
            # flipped offsets: interior cell (r,c) is covered if an anchor at
            # (r-dr, c-dc) hit, dr in 1..ih, dc in 1..iw
            ks = np.zeros((1, 1, 2 * ih + 1, 2 * iw + 1), dtype=np.float32)
            ks[0, 0, :ih + 1, :iw + 1] = 1.0  # reads anchors up-left of cell
            ks[0, 0, ih, iw] = 1.0
            ks[0, 0, ih, iw] = 1.0
            # anchor at (r-dr,c-dc) for dr in 1..ih: kernel offset (-dr,-dc)
            ks = np.zeros((1, 1, 2 * ih + 1, 2 * iw + 1), dtype=np.float32)
            for dr in range(1, ih + 1):
                for dc in range(1, iw + 1):
                    ks[0, 0, ih - dr, iw - dc] = 1.0
            sc = g.op("Conv", hit, g.init(ks), pads=[ih, iw, ih, iw])
            pot_parts.append(sc)
    # P = sign(sum of parts) restricted to black
    s = pot_parts[0]
    for p_ in pot_parts[1:]:
        s = g.op("Add", s, p_)
    P = g.op("Mul", g.op("Sign", s), ch0)

    # --- cascade ----------------------------------------------------------
    one = g.init(np.array([1.0], dtype=np.float32))
    N0 = g.op("Add", fg, P)  # disjoint -> {0,1}
    # B2 = black & ~P & R(N0) & D(N0)
    rn = g.op("Conv", N0, g.init(_shift_kernel(0, 1)), pads=[1, 1, 1, 1])
    dn = g.op("Conv", N0, g.init(_shift_kernel(1, 0)), pads=[1, 1, 1, 1])
    notP = g.op("Sub", one, P)
    b2 = g.op("Mul", g.op("Mul", ch0, notP), g.op("Mul", rn, dn))
    a2 = g.op("Mul", b2, g.init(np.array([2.0], dtype=np.float32)))  # x2 folded

    # kernel (U+L)/2 with bias -0.5 : one Conv per iter
    kUL = 0.5 * (_shift_kernel(-1, 0) + _shift_kernel(0, -1))
    kULn = g.init(kUL)
    bias = g.init(np.array([-0.5], dtype=np.float32))
    Z = N0
    Y = None
    for _ in range(casc):
        c = g.op("Conv", Z, kULn, bias, pads=[1, 1, 1, 1])
        y = g.op("Mul", a2, g.op("Relu", c))  # {0,1}: 2*relu((U+L)/2-0.5)
        Z = g.op("Add", N0, y)  # y supersedes previous Y (monotone growth)
        Y = y

    # --- paint yellow(4) at P | Y ----------------------------------------
    fill = g.op("Sign", g.op("Add", P, Y))
    inv = g.op("Sub", one, fill)
    cleared = g.op("Mul", "input", inv)
    wl = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    wl[4, 0, 0, 0] = 1.0
    lifted = g.op("Conv", fill, g.init(wl))
    g.op("Add", cleared, lifted, out="output")
    return g.model()
