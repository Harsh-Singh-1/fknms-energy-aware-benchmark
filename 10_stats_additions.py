"""
10_stats_additions.py -- Friedman omnibus test and kappa extraction (Round 5,
R2 comments 1 and 5).

Comment 1: the scene-level Wilcoxon signed-rank test with Holm correction over
36 pairs cannot reach significance at n=9 (minimum exact two-sided p = 2/512 =
0.0039 > 0.05/36 = 0.0014). This script adds the attainable scene-level
omnibus: a Friedman test across the nine classifiers over the nine paired
scenes, per protocol, with Kendall's W as effect size and the mean rank of
every model. The block bootstrap (03b) remains the pairwise procedure.

Comment 5: the kappa range quoted in the abstract/conclusion (0.12-0.39) is in
no table. This script extracts per-scene Cohen's kappa under the geographic
holdout for every model, reports the full range (traceable), and writes the
kappa columns for Table 14's three models.

Inputs: benchmark_summary.csv (default: reference_outputs, i.e. the published
run; pass --summary outputs/benchmark_summary.csv after any re-run).

Outputs (in --out):
    friedman_omnibus.csv   protocol, chi2, df, p, kendalls_W, n_scenes, n_models
    model_mean_ranks.csv   protocol, model, mean_rank
    table14_kappa.csv      scene x {LogReg, SVM, 2D_CNN} per-scene kappa (holdout)
    kappa_range.txt        min/max per-scene kappa across all models (holdout)

Run (instant):
    python 10_stats_additions.py

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
T14_MODELS = ["LogReg", "SVM", "2D_CNN"]


def main():
    ap = argparse.ArgumentParser(description="Friedman omnibus + kappa extraction.")
    ap.add_argument("--summary",
                    default=str(HERE / "reference_outputs" / "benchmark_summary.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    oa = defaultdict(dict)      # (protocol, scene) -> {model: OA_mean}
    kap = defaultdict(dict)     # scene -> {model: kappa}  (geo_holdout)
    with open(a.summary, newline="") as f:
        for r in csv.DictReader(f):
            oa[(r["protocol"], r["scene"])][r["model"]] = float(r["OA_mean"])
            if r["protocol"] == "geo_holdout":
                kap[r["scene"]][r["model"]] = float(r["kappa_mean"])

    # ---- Friedman omnibus per protocol ----
    fr_rows, rank_rows = [], []
    for proto in sorted({p for p, _ in oa}):
        scenes = sorted(s for p, s in oa if p == proto)
        models = sorted(set.intersection(*(set(oa[(proto, s)]) for s in scenes)))
        mat = np.array([[oa[(proto, s)][m] for m in models] for s in scenes])
        n, k = mat.shape
        chi2, p = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
        W = chi2 / (n * (k - 1))                      # Kendall's W from Friedman chi2
        fr_rows.append(dict(protocol=proto, n_scenes=n, n_models=k,
                            chi2=round(chi2, 3), df=k - 1, p=round(p, 6),
                            kendalls_W=round(W, 3)))
        ranks = stats.rankdata(-mat, axis=1)          # rank 1 = best OA per scene
        for j, m in enumerate(models):
            rank_rows.append(dict(protocol=proto, model=m,
                                  mean_rank=round(float(ranks[:, j].mean()), 2)))
        print(f"[{proto}] Friedman chi2({k-1}) = {chi2:.3f}, p = {p:.6f}, "
              f"Kendall's W = {W:.3f}  ({n} scenes x {k} models)")
        order = sorted((r for r in rank_rows if r["protocol"] == proto),
                       key=lambda r: r["mean_rank"])
        print("  mean ranks: " + ", ".join(f"{r['model']} {r['mean_rank']}" for r in order))

    with open(out / "friedman_omnibus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fr_rows[0].keys()))
        w.writeheader(); w.writerows(fr_rows)
    with open(out / "model_mean_ranks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "model", "mean_rank"])
        w.writeheader(); w.writerows(rank_rows)

    # ---- kappa extraction (geographic holdout) ----
    all_k = [v for s in kap for v in kap[s].values()]
    lo, hi = min(all_k), max(all_k)
    which_lo = [(s, m) for s in kap for m, v in kap[s].items() if v == lo]
    which_hi = [(s, m) for s in kap for m, v in kap[s].items() if v == hi]
    print(f"\nPer-scene kappa under the geographic holdout, all models: "
          f"{lo:.3f} to {hi:.3f}")
    print(f"  min at {which_lo}, max at {which_hi}")
    with open(out / "kappa_range.txt", "w") as f:
        f.write(f"Per-scene Cohen's kappa, geographic holdout, all nine models:\n"
                f"min = {lo:.4f} at {which_lo}\nmax = {hi:.4f} at {which_hi}\n"
                f"Quote this computed range in the abstract/conclusion in place "
                f"of any untraceable figure.\n")

    t14 = []
    for s in sorted(kap):
        row = dict(scene=s)
        for m in T14_MODELS:
            row[f"kappa_{m}"] = round(kap[s].get(m, float("nan")), 4)
        t14.append(row)
    with open(out / "table14_kappa.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(t14[0].keys()))
        w.writeheader(); w.writerows(t14)
    print(f"Saved {out / 'friedman_omnibus.csv'}, model_mean_ranks.csv, "
          f"table14_kappa.csv, kappa_range.txt")


if __name__ == "__main__":
    main()
