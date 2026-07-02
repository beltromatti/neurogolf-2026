"""Tiny ONNX graph builder for NeuroGolf networks.

Every network maps input[1,10,30,30] float32 -> output[1,10,30,30] float32,
judged as (output > 0) == one-hot(expected).

Cost model (mirrors official judge):
  cost = params + memory
  params = total elements across initializers + Constant-node tensors
  memory = sum over intermediate tensors of bytes (max of declared/runtime shape)
           -- tensors named exactly "input"/"output" are FREE.
Score = max(1, 25 - ln(cost)); cost 0 => 25.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

CH, H, W = 10, 30, 30
GRID = [1, CH, H, W]
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

_NP2ONNX = {
    np.dtype(np.float32): F32,
    np.dtype(np.int64): I64,
    np.dtype(np.bool_): BOOL,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.float16): TensorProto.FLOAT16,
}


class G:
    """Incremental ONNX graph builder.

    Usage:
        g = G()
        t = g.op("Transpose", "input", perm=[0,1,3,2])
        g.out(t)                  # or name the last op's output "output" directly
        model = g.model()
    """

    def __init__(self, opset: int = 10):
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.opset = opset
        self._n = 0

    def name(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def init(self, arr, name: str | None = None) -> str:
        """Add an initializer (counts as params = arr.size)."""
        arr = np.asarray(arr)
        name = name or self.name("w")
        dt = _NP2ONNX[arr.dtype]
        self.inits.append(helper.make_tensor(name, dt, list(arr.shape), arr.flatten().tolist()))
        return name

    def op(self, op_type: str, *inputs: str, out: str | None = None, n_out: int = 1, **attrs) -> str:
        """Append a node; returns output tensor name (or list if n_out>1)."""
        outs = [out or self.name()] if n_out == 1 else [self.name() for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, list(inputs), outs, **attrs))
        return outs[0] if n_out == 1 else outs

    # -- finalize ---------------------------------------------------------
    def model(self) -> onnx.ModelProto:
        inp = helper.make_tensor_value_info("input", F32, GRID)
        outp = helper.make_tensor_value_info("output", F32, GRID)
        graph = helper.make_graph(self.nodes, "g", [inp], [outp], self.inits)
        m = helper.make_model(
            graph, ir_version=10, opset_imports=[helper.make_opsetid("", self.opset)]
        )
        # embed inferred value_info so shape checks are deterministic
        m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
        onnx.checker.check_model(m, full_check=True)
        return m


# ---------------------------------------------------------------------------
# macro-primitives (single ops or short chains); all keep tensors small.
# ---------------------------------------------------------------------------

def identity() -> onnx.ModelProto:
    g = G()
    g.op("Identity", "input", out="output")
    return g.model()


def transpose_hw() -> onnx.ModelProto:
    g = G()
    g.op("Transpose", "input", out="output", perm=[0, 1, 3, 2])
    return g.model()


def gather_axis(idx: list[int], axis: int) -> onnx.ModelProto:
    """Single Gather on one spatial axis (flip/shift/reorder rows or cols)."""
    g = G()
    w = g.init(np.asarray(idx, dtype=np.int64))
    g.op("Gather", "input", w, out="output", axis=axis)
    return g.model()


def hflip() -> onnx.ModelProto:
    return gather_axis(list(range(W - 1, -1, -1)), 3)


def vflip() -> onnx.ModelProto:
    return gather_axis(list(range(H - 1, -1, -1)), 2)


def color_gather(idx: list[int]) -> onnx.ModelProto:
    """output[c] = input[idx[c]] -- color permutation, 10 params, 0 mem."""
    return gather_axis(idx, 1)


def conv1x1(wmat: np.ndarray) -> onnx.ModelProto:
    """Channel mixing conv: wmat[out_ch, in_ch]. 100 params, 0 mem."""
    g = G()
    w = g.init(wmat.reshape(CH, CH, 1, 1).astype(np.float32))
    g.op("Conv", "input", w, out="output")
    return g.model()


def score_of(cost: float) -> float:
    import math

    if cost <= 0:
        return 25.0
    return max(1.0, 25.0 - math.log(cost))
