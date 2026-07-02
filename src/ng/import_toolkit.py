"""Import toolkit-generated solutions into solutions/ with robust re-checking.

Usage: python3 -m ng.import_toolkit [extra_pairs]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx

from .check import ROOT, check

TK_SOLUTIONS = ROOT / "toolkit" / "solutions"


def main(extra: int = 200) -> None:
    report = {}
    for p in sorted(TK_SOLUTIONS.glob("task*.onnx")):
        num = int(p.stem[4:])
        try:
            r = check(onnx.load(str(p)), num, extra=extra)
        except Exception as exc:  # generator quirks etc.
            report[p.stem] = {"error": str(exc)[:200]}
            continue
        report[p.stem] = {
            "correct": r["correct"], "points": round(r["points"], 3),
            "robust": f"{r['robust_ok']}/{(r['robust_ok'] or 0) + (r['robust_bad'] or 0)}",
        }
        print(p.stem, report[p.stem], flush=True)
    json.dump(report, open(ROOT / "solutions" / "import_report.json", "w"), indent=1)
    total = sum(v.get("points", 0) for v in report.values() if v.get("correct"))
    print(f"imported: {sum(1 for v in report.values() if v.get('correct'))} tasks, {total:.2f} pts")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
