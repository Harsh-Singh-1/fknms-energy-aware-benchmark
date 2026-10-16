"""
09_block_audit.py -- per-scene spatial-block counts (Round 5, R2 comment 11).

Reads benchmark_block_metrics.csv (the per-block confusion export of
03_benchmark.py that feeds the spatial bootstrap) and reports, per scene and
protocol, the number of distinct test blocks. This documents why the
geographic-holdout total is 98 rather than a multiple of nine: block counts
are determined per scene AFTER cloud/shadow masking and the 300 m boundary
buffer, so they vary by scene. On the reference outputs: 11 blocks for eight
scenes and 10 for wet_2024 (the one scene above 5 percent in-area cloud),
totaling 98; the cross-validation counts vary from 14 to 20 per scene and
their total of 144 is only coincidentally divisible by nine.

Run (instant):
    python 09_block_audit.py                       # uses reference_outputs
    python 09_block_audit.py --metrics outputs/benchmark_block_metrics.csv

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Audit per-scene test-block counts.")
    ap.add_argument("--metrics",
                    default=str(HERE / "reference_outputs" / "benchmark_block_metrics.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    blocks = defaultdict(set)   # (protocol, scene) -> block ids
    models = defaultdict(set)
    with open(a.metrics, newline="") as f:
        for r in csv.DictReader(f):
            blocks[(r["protocol"], r["scene"])].add(r["block"])
            models[(r["protocol"], r["scene"])].add(r["model"])

    rows = []
    for proto in sorted({p for p, _ in blocks}):
        total = 0
        print(f"\n{proto}:")
        for (p, scene), bs in sorted(blocks.items()):
            if p != proto:
                continue
            total += len(bs)
            rows.append(dict(protocol=p, scene=scene, n_blocks=len(bs),
                             n_models=len(models[(p, scene)])))
            print(f"  {scene:14s} {len(bs):3d} blocks")
        print(f"  {'TOTAL':14s} {total:3d}")

    p = out / "block_counts.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "scene", "n_blocks", "n_models"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
