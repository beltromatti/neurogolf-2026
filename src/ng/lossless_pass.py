"""Lossless optimization sweep over solutions/: onnxoptimizer + onnxsim,
keep only if points strictly improve (verified, not assumed).

Usage: python3 -m ng.lossless_pass [workers]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .check import ROOT, SOLUTIONS


def optimize_one(num: int) -> dict:
    import onnx
    import onnxoptimizer

    from .check import check

    path = SOLUTIONS / f"task{num:03d}.onnx"
    if not path.exists():
        return {"task": num, "skip": True}
    model = onnx.load(str(path))
    base = check(model, num, extra=30, save_if_ok=False)
    if not base["correct"]:
        return {"task": num, "skip": True, "note": "base not robust"}
    best_pts = base["points"]
    out = {"task": num, "old": round(best_pts, 3), "new": round(best_pts, 3)}

    candidates = []
    try:
        candidates.append(onnxoptimizer.optimize(model))
    except Exception:
        pass
    try:
        from onnxsim import simplify

        sim, ok = simplify(model)
        if ok:
            candidates.append(sim)
    except Exception:
        pass
    for cand in candidates:
        try:
            r = check(cand, num, extra=100, save_if_ok=False)
        except Exception:
            continue
        if r["correct"] and r["points"] > best_pts + 1e-9:
            best_pts = r["points"]
            onnx.save(cand, str(path))
            out["new"] = round(best_pts, 3)
    return out


def main(workers: int = 6) -> None:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(optimize_one, n) for n in range(1, 401)]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if not r.get("skip") and r["new"] > r["old"]:
                print(f"task{r['task']:03d}: {r['old']} -> {r['new']}", flush=True)
    gain = sum(r.get("new", 0) - r.get("old", 0) for r in results if not r.get("skip"))
    print(f"\nLOSSLESS GAIN: +{gain:.2f} points")
    json.dump(results, open(SOLUTIONS / "lossless_report.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
