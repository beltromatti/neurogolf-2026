"""Validation: official-mirror scoring + robustness on freshly generated pairs.

check(model, task_num, extra=100) runs:
  1. toolkit validator (official cost + all train/test/arc-gen pairs)
  2. `extra` brand-new pairs from the ARC-GEN generator (same code that
     produced the judge's private benchmark) -- passing these makes private
     failures very unlikely.
"""
from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toolkit"))

from src.tasks import grid_to_tensor, load_task  # noqa: E402  (toolkit package)
from src.validator import validate  # noqa: E402

ARC_GEN = ROOT / "data" / "ARC-GEN"
GEN_CACHE = ROOT / "data" / "gen_cache"
GEN_CACHE.mkdir(exist_ok=True)
SOLUTIONS = ROOT / "solutions"
SOLUTIONS.mkdir(exist_ok=True)


@lru_cache(maxsize=None)
def hash_order() -> list[str]:
    return json.load(open(ROOT / "data" / "hash_order.json"))


def task_hash(num: int) -> str:
    return hash_order()[num - 1]


def gen_pairs(num: int, n: int = 100) -> list[dict]:
    """Fresh pairs from ARC-GEN (cached). Mimics the private benchmark."""
    cache = GEN_CACHE / f"task{num:03d}_{n}.json"
    if cache.exists():
        return json.load(open(cache))
    h = task_hash(num)
    out = subprocess.run(
        [sys.executable, "arc_gen.py", "generate", h, str(n)],
        cwd=ARC_GEN, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"arc_gen failed for {h}: {out.stderr[-500:]}")
    pairs = eval(out.stdout)  # trusted local generator output (python literal)
    json.dump(pairs, open(cache, "w"))
    return pairs


def run_pairs(model_bytes: bytes, pairs: list[dict]) -> tuple[int, int, dict | None]:
    sess = onnxruntime.InferenceSession(model_bytes)
    ok = bad = 0
    first_fail = None
    for p in pairs:
        gi, go = p["input"], p["output"]
        if max(len(gi), len(gi[0]), len(go), len(go[0])) > 30:
            continue
        x = grid_to_tensor(gi)
        y = grid_to_tensor(go)
        out = sess.run(["output"], {"input": x})[0]
        if np.array_equal((out > 0).astype(np.float32), y):
            ok += 1
        else:
            bad += 1
            if first_fail is None:
                first_fail = p
    return ok, bad, first_fail


def check(model: onnx.ModelProto, num: int, extra: int = 100, save_if_ok: bool = True):
    """Full check. Returns dict with points/cost/robustness; saves winning nets."""
    tmp = SOLUTIONS / f"_cand_task{num:03d}.onnx"
    onnx.save(model, str(tmp))
    task = load_task(num)
    res = validate(tmp, task)
    result = {
        "task": num, "correct": res.correct, "points": res.points,
        "params": res.params, "memory": res.memory, "error": res.error,
        "passes": res.passes, "fails": res.fails, "robust_ok": None, "robust_bad": None,
    }
    if res.correct and extra:
        ok, bad, ff = run_pairs(model.SerializeToString(), gen_pairs(num, extra))
        result["robust_ok"], result["robust_bad"] = ok, bad
        if bad:
            result["correct"] = False
            result["points"] = 0.0
            result["robust_fail_pair"] = ff
    final = SOLUTIONS / f"task{num:03d}.onnx"
    if result["correct"] and save_if_ok:
        prev = best_points(num)
        if result["points"] > prev:
            onnx.save(model, str(final))
    tmp.unlink(missing_ok=True)
    return result


def best_points(num: int) -> float:
    """Points of the currently saved solution for a task (0 if none)."""
    final = SOLUTIONS / f"task{num:03d}.onnx"
    if not final.exists():
        return 0.0
    task = load_task(num)
    res = validate(final, task)
    return res.points if res.correct else 0.0
