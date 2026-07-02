"""Merge public bundle networks with our solutions: best-per-task, robustly.

Usage: python3 -m ng.merge_bundles [extra_pairs] [workers]
Extracts each data/bundles/*/submission.zip, then for every task validates all
candidate .onnx (bundles + current solutions/) and keeps the highest-scoring
robust one in solutions/. Results in solutions/merge_report.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .check import ROOT, SOLUTIONS

BUNDLES = ROOT / "data" / "bundles"


def extract_all() -> list[Path]:
    dirs = []
    for sub in sorted(BUNDLES.iterdir()):
        z = sub / "submission.zip"
        if not z.exists():
            continue
        out = sub / "nets"
        if not out.exists():
            out.mkdir()
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        # some zips nest a directory; flatten
        dirs.append(out)
    return dirs


def candidates_for(num: int, dirs: list[Path]) -> list[Path]:
    name = f"task{num:03d}.onnx"
    cands = []
    for d in dirs:
        for p in d.rglob(name):
            cands.append(p)
    cur = SOLUTIONS / name
    if cur.exists():
        cands.append(cur)
    return cands


def process_task(num: int, dirs: list[str], extra: int) -> dict:
    # imports inside: runs in worker process
    import onnx

    from .check import check

    best = {"task": num, "points": 0.0, "source": None}
    for p in candidates_for(num, [Path(d) for d in dirs]):
        try:
            r = check(onnx.load(str(p)), num, extra=extra, save_if_ok=False)
        except Exception:
            continue
        if r["correct"] and r["points"] > best["points"]:
            best.update(points=r["points"], source=str(p),
                        params=r["params"], memory=r["memory"])
    if best["source"] and not best["source"].endswith(f"solutions/task{num:03d}.onnx"):
        shutil.copy(best["source"], SOLUTIONS / f"task{num:03d}.onnx")
    return best


def main(extra: int = 100, workers: int = 8) -> None:
    dirs = [str(d) for d in extract_all()]
    print("bundle dirs:", len(dirs))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_task, n, dirs, extra): n for n in range(1, 401)}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            done += 1
            if r["points"] > 0 and done % 20 == 0:
                print(f"[{done}/400] latest: task{r['task']:03d} {r['points']:.2f}", flush=True)
    total = sum(r["points"] for r in results)
    solved = sum(1 for r in results if r["points"] > 0)
    print(f"\nMERGED TOTAL: {solved}/400 solved, {total:.2f} points")
    json.dump(sorted(results, key=lambda r: r["task"]),
              open(SOLUTIONS / "merge_report.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         int(sys.argv[2]) if len(sys.argv) > 2 else 8)
