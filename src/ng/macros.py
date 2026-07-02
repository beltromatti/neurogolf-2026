"""Composable macro-blocks for common ARC sub-computations.

All operate on a G builder and tensor names; they return output tensor names.
Design rule: minimize count & size of intermediate tensors (each costs bytes).
Bool tensors cost 1 byte/elem, float32 4, float16 2, int64 8 (avoid).
"""
from __future__ import annotations

import numpy as np

from .builder import CH, G, H, W


def fg_mask(g: G, inp: str = "input") -> str:
    """[1,1,30,30] float: 1 where any non-background(!=0 color) cell.
    Channels 1..9 summed => nonzero-color mask. (channel 0 = color black.)"""
    w = np.zeros((1, CH, 1, 1), dtype=np.float32)
    w[0, 1:, 0, 0] = 1.0
    wname = g.init(w)
    return g.op("Conv", inp, wname)


def occupied_mask(g: G, inp: str = "input") -> str:
    """[1,1,30,30] float: 1 where the cell is inside the original grid
    (any channel set, including color 0)."""
    w = np.ones((1, CH, 1, 1), dtype=np.float32)
    return g.op("Conv", inp, g.init(w))


def plus_dilate(g: G, x: str, mask: str, iters: int, use_fp16: bool = False) -> str:
    """4-connected reachability: iterate x = dilate+(x) * mask.
    x, mask: [1,1,30,30] float. Cost: 2 tensors per iteration."""
    k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32).reshape(1, 1, 3, 3)
    kname = g.init(k)
    for _ in range(iters):
        d = g.op("Conv", x, kname, pads=[1, 1, 1, 1])
        m = g.op("Mul", d, mask)
        # keep values in {0,>0}; no clip needed since mask multiplication
        x = m
    return x


def binarize(g: G, x: str) -> str:
    """x>0 as float via Sign (values are >=0 in our flows)."""
    return g.op("Sign", x)


def const_scalar(g: G, v: float) -> str:
    return g.init(np.array([v], dtype=np.float32))


def select_channel(g: G, inp: str, c: int) -> str:
    """[1,1,30,30] one channel via Gather axis=1 (1 param)."""
    idx = g.init(np.array([c], dtype=np.int64))
    return g.op("Gather", inp, idx, axis=1)


def paint(g: G, base: str, mask01: str, color: int, out: str | None = None) -> str:
    """Recolor cells where mask01==1 to `color`, keeping others.
    base [1,10,30,30], mask01 [1,1,30,30] in {0,1}.
    output = base*(1-mask) + onehot(color)*mask, done via Conv trick:
    scatter mask into channel `color`, zero base under mask."""
    # zero out all channels under mask: base * (1-mask)  (broadcast over CH)
    one = const_scalar(g, 1.0)
    inv = g.op("Sub", one, mask01)
    cleared = g.op("Mul", base, inv)
    # lift mask into channel c: Conv 1->10 with single weight at out_ch=c
    w = np.zeros((CH, 1, 1, 1), dtype=np.float32)
    w[color, 0, 0, 0] = 1.0
    lifted = g.op("Conv", mask01, g.init(w))
    return g.op("Add", cleared, lifted, out=out)
