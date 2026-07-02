"""Package solutions/ into submission.zip and optionally submit via Kaggle CLI.

Usage:
  python3 -m ng.package            # build submission.zip + score report
  python3 -m ng.package --submit "message"
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

from .check import ROOT, SOLUTIONS

OUT = ROOT / "submission.zip"


def main() -> None:
    nets = sorted(SOLUTIONS.glob("task[0-9][0-9][0-9].onnx"))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in nets:
            zf.write(p, p.name)
    # score summary from merge/import reports if present
    report = SOLUTIONS / "merge_report.json"
    total = None
    if report.exists():
        rs = json.load(open(report))
        total = sum(r["points"] for r in rs)
    print(f"submission.zip: {len(nets)} networks, {OUT.stat().st_size/1e6:.2f} MB"
          + (f", expected LB ≈ {total:.2f}" if total else ""))

    if len(sys.argv) > 2 and sys.argv[1] == "--submit":
        msg = sys.argv[2]
        r = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", "neurogolf-2026",
             "-f", str(OUT), "-m", msg],
            capture_output=True, text=True,
        )
        print(r.stdout or r.stderr)


if __name__ == "__main__":
    main()
