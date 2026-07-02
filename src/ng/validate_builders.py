"""Re-validate every src/ng/tasks/tNNN.py builder and keep improvements.

Usage: python3 -m ng.validate_builders [extra]
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

from .check import ROOT, best_points, check

TASKS_DIR = Path(__file__).parent / "tasks"


def main(extra: int = 300) -> None:
    report = {}
    for p in sorted(TASKS_DIR.glob("t[0-9][0-9][0-9].py")):
        num = int(p.stem[1:])
        try:
            mod = importlib.import_module(f"ng.tasks.{p.stem}")
            model = mod.build()
            prev = best_points(num)
            r = check(model, num, extra=extra)  # auto-saves if better
            report[p.stem] = {
                "correct": r["correct"], "points": round(r["points"], 2),
                "prev": round(prev, 2),
                "improved": bool(r["correct"] and r["points"] > prev),
                "robust_bad": r["robust_bad"],
            }
        except Exception:
            report[p.stem] = {"error": traceback.format_exc(limit=1)[-200:]}
        print(p.stem, report[p.stem], flush=True)
    json.dump(report, open(ROOT / "solutions" / "builder_report.json", "w"), indent=1)
    imp = sum(1 for v in report.values() if v.get("improved"))
    print(f"\n{imp} improvements out of {len(report)} builders")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
