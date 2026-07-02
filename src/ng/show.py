"""CLI: inspect a task. `python3 -m ng.show 123 [n_pairs]`

Prints example grids, size relations, and the readable re-arc verifier source
(the exact specification of the transformation).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def grids(task_num: int, n: int = 3) -> None:
    data = json.load(open(ROOT / "data" / "tasks" / f"task{task_num:03d}.json"))
    h = json.load(open(ROOT / "data" / "hash_order.json"))[task_num - 1]
    print(f"=== task{task_num:03d} (ARC {h}) ===")
    counts = {k: len(v) for k, v in data.items()}
    print("pairs:", counts)
    shown = 0
    for section in ("train", "test"):
        for p in data[section]:
            if shown >= n:
                break
            gi, go = p["input"], p["output"]
            print(f"--- {section} pair (in {len(gi)}x{len(gi[0])} -> out {len(go)}x{len(go[0])}) ---")
            for row in gi:
                print(" ", "".join(str(c) for c in row))
            print("  =>")
            for row in go:
                print(" ", "".join(str(c) for c in row))
            shown += 1
    # size relation summary over all pairs
    rels = set()
    for section in ("train", "test", "arc-gen"):
        for p in data[section]:
            gi, go = p["input"], p["output"]
            rels.add((len(go) == len(gi), len(go[0]) == len(gi[0]),
                      len(go), len(go[0])) if len(go) != len(gi) or len(go[0]) != len(gi[0]) else "same")
    print("size relations:", "same-size only" if rels == {"same"} else rels)
    # verifier source
    src = open(ROOT / "data" / "ref_re-arc" / "verifiers.py").read()
    m = re.search(rf"def verify_{h}\(I: Grid\) -> Grid:\n(.*?)\n\n", src, re.S)
    print("--- re-arc verifier (spec) ---")
    print(f"def verify_{h}(I):")
    print(m.group(1) if m else "  (not found)")


if __name__ == "__main__":
    grids(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3)
