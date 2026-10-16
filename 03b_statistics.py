"""
03b_statistics.py -- spatially aware significance, from the per-block export.

Reads benchmark_block_metrics.csv (written by 03_benchmark.py) and produces the
paper's statistics without refitting any model:

  1. Block-level spatial bootstrap, stratified by scene. The resampling unit is
     a spatial block, whose size exceeds the habitat autocorrelation range, so
     blocks are treated as approximately independent. In each of --iters
     iterations (default 5,000, random seed --seed) blocks are resampled with
     replacement separately within each scene and then pooled; the same
     resampled block set is shared by every model, so paired differences are
     evaluated on identical data. This yields a 95 percent confidence interval
     per model (stats_block_ci.csv) and, from the shared draws, a confidence
     interval and a two-sided bootstrap p-value on every pairwise accuracy
     difference, Holm-corrected (stats_pairwise.csv). Passing --strata combined
     resamples from a single pool of all scene-block identifiers instead, for
     comparison.

  2. Scene-level test. Treating the nine scenes as independent replicates, a
     Wilcoxon signed-rank test on paired per-scene overall accuracy is run for
     every model pair, Holm-corrected (stats_scene_level.csv). This is the
     conservative test emphasized in the text.

Overall accuracy is pooled over the three benthic classes, matching the
benchmark tables. Run after 03_benchmark.py:
    python 03b_statistics.py

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CLASSES = [2, 3, 4]                       # Sand/Bare, Seagrass, Coral/Hard-bottom
CELLS = [f"c{i}_{j}" for i in CLASSES for j in CLASSES]
DIAG = [CELLS.index(f"c{c}_{c}") for c in CLASSES]


def read_metrics(path):
    """protocol -> scene -> model -> ndarray(n_blocks, 9) in file order."""
    prot = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            vec = np.array([int(float(r[c])) for c in CELLS], dtype=np.int64)
            prot.setdefault(r["protocol"], {}).setdefault(r["scene"], {}) \
                .setdefault(r["model"], []).append(vec)
    for p in prot:
        for s in prot[p]:
            for m in prot[p][s]:
                prot[p][s][m] = np.array(prot[p][s][m], dtype=np.int64)
    return prot


def oa_from_pool(pool):
    tot = pool.sum(axis=0)
    n = tot.sum()
    return float(tot[DIAG].sum() / n) if n > 0 else float("nan")


def bootstrap_protocol(prot_scene, strata, nboot, seed):
    """Point OA per model plus nboot resampled OAs, with the block draw shared
    across models within each iteration."""
    scenes = sorted(prot_scene.keys())
    models = sorted({m for s in scenes for m in prot_scene[s]})
    nblocks = {s: prot_scene[s][next(iter(prot_scene[s]))].shape[0] for s in scenes}

    point, nblk = {}, {}
    for m in models:
        pool = np.vstack([prot_scene[s][m] for s in scenes if m in prot_scene[s]])
        point[m] = oa_from_pool(pool)
        nblk[m] = pool.shape[0]

    rng = np.random.default_rng(seed)
    boot = {m: np.empty(nboot) for m in models}
    if strata == "combined":
        pooled = {m: np.vstack([prot_scene[s][m] for s in scenes if m in prot_scene[s]])
                  for m in models}
        N = sum(nblocks.values())
        for it in range(nboot):
            take = rng.integers(0, N, N)
            for m in models:
                boot[m][it] = oa_from_pool(pooled[m][take])
    else:
        for it in range(nboot):
            draws = {s: rng.integers(0, nblocks[s], nblocks[s]) for s in scenes}
            for m in models:
                parts = [prot_scene[s][m][draws[s]] for s in scenes if m in prot_scene[s]]
                boot[m][it] = oa_from_pool(np.vstack(parts))
    return point, boot, nblk


def holm(rows, key="p_raw"):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    m = len(rows)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = max(prev, min(1.0, (m - rank) * rows[i][key]))
        prev = adj
        rows[i]["p_holm"] = round(adj, 4)
        rows[i]["significant"] = adj < 0.05
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Spatial statistics from the per-block export.")
    ap.add_argument("--metrics", default=str(HERE / "outputs" / "benchmark_block_metrics.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20240517,
                    help="bootstrap random seed (the value used for the reported tables)")
    ap.add_argument("--strata", choices=["scene", "combined"], default="scene",
                    help="scene = resample blocks within each scene (the reported method)")
    a = ap.parse_args()

    prot = read_metrics(a.metrics)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"strata={a.strata}  iters={a.iters}  seed={a.seed}  protocols={sorted(prot)}")

    ci_rows, pair_rows, scene_rows = [], [], []
    for p in sorted(prot):
        point, boot, nblk = bootstrap_protocol(prot[p], a.strata, a.iters, a.seed)
        models = sorted(point)
        for m in models:
            lo, hi = np.percentile(boot[m], 2.5), np.percentile(boot[m], 97.5)
            ci_rows.append(dict(protocol=p, model=m, OA=round(point[m], 4),
                                CI_lo=round(float(lo), 4), CI_hi=round(float(hi), 4),
                                n_blocks=nblk[m]))

        raw = []
        for ma, mb in itertools.combinations(models, 2):
            d = boot[ma] - boot[mb]
            fr_le = float(np.mean(d <= 0.0))
            fr_ge = float(np.mean(d >= 0.0))
            pr = min(max(2.0 * min(fr_le, fr_ge), 1.0 / a.iters), 1.0)
            raw.append(dict(protocol=p, model_a=ma, model_b=mb,
                            dOA=round(point[ma] - point[mb], 4),
                            dCI_lo=round(float(np.percentile(d, 2.5)), 4),
                            dCI_hi=round(float(np.percentile(d, 97.5)), 4),
                            p_raw=round(pr, 4)))
        holm(raw)
        pair_rows.extend(raw)

        # scene-level Wilcoxon on paired per-scene pooled OA
        scenes = sorted(prot[p].keys())
        s_oa = {m: {s: oa_from_pool(prot[p][s][m]) for s in scenes if m in prot[p][s]}
                for m in models}
        raw_s = []
        for ma, mb in itertools.combinations(models, 2):
            common = [s for s in scenes if s in s_oa[ma] and s in s_oa[mb]]
            xi = np.array([s_oa[ma][s] for s in common])
            xj = np.array([s_oa[mb][s] for s in common])
            try:
                pv = float(wilcoxon(xi, xj, zero_method="wilcox").pvalue)
            except ValueError:
                pv = 1.0
            raw_s.append(dict(protocol=p, model_a=ma, model_b=mb,
                              mean_dOA=round(float(np.mean(xi - xj)), 4),
                              n_scenes=len(common), p_raw=round(pv, 4)))
        holm(raw_s)
        for r in raw_s:
            del r["p_raw"]
        scene_rows.extend(raw_s)

        sig = sum(r["significant"] for r in raw if r["protocol"] == p)
        print(f"  {p}: {len(models)} models, {sig}/{len(raw)} bootstrap pairs significant (Holm)")

    write(out / "stats_block_ci.csv", ci_rows)
    write(out / "stats_pairwise.csv", pair_rows)
    write(out / "stats_scene_level.csv", scene_rows)
    sig = sum(r["significant"] for r in scene_rows)
    print(f"\nScene-level significant pairs (Holm): {sig} of {len(scene_rows)} "
          f"(across both protocols) -- the conservative test.")


if __name__ == "__main__":
    main()
