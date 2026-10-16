"""
06_figures.py -- regenerate the manuscript figures from the pipeline outputs.

Reads the CSVs written by 02-05 and writes PNGs to ./figures:

  benchmark_summary.csv     -> fig3 (accuracy + train/inference energy),
                               fig4 (per-class F1), fig6 (accuracy vs energy)
  energy_per_stage.csv      -> fig3, fig6
  benchmark_confusion.csv   -> fig5 (LogReg vs 2D CNN confusion)
  ablation_learning_curve.csv -> fig9
  ablation_bathymetry.csv     -> fig10

Any figure whose input is missing is skipped with a note, so this can run after a
partial pipeline. The feature-importance and pseudo-label DII panels (paper
Figures 7-8) are diagnostics of the pseudo-label baseline and are produced by
that baseline, not from these CSVs.

Run after the analysis steps:
    python 06_figures.py

Author: Harsh Deep Singh Narula
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["RF", "ExtraTrees", "XGBoost", "LightGBM", "SVM", "LogReg", "kNN", "MLP", "2D_CNN"]
LBL = {"RF": "RF", "ExtraTrees": "Extra-Trees", "XGBoost": "XGBoost", "LightGBM": "LightGBM",
       "SVM": "SVM", "LogReg": "LogReg", "kNN": "kNN", "MLP": "MLP", "2D_CNN": "2D CNN"}
CLR = {"sand": "#DDCC77", "sea": "#117733", "coral": "#CC6677"}
CLASSES = ["Sand/Bare", "Seagrass", "Coral/Hard-bottom"]


def _order(models):
    return [m for m in ORDER if m in set(models)]


def energy_by_model(fe):
    """train, inference, and operational uWh per model (mean over scenes)."""
    g = fe.groupby(["model", "stage"]).energy_uWh_mean.mean().unstack("stage")
    tr = g.get("train"); inf = g.get("fullmap_infer")
    out = pd.DataFrame({"train": tr, "infer": inf})
    out["operational"] = out["train"].fillna(0) + out["infer"].fillna(0)
    return out


def fig_accuracy_energy(summ, fe, figdir):
    geo = summ[summ.protocol == "geo_holdout"].groupby("model").OA_mean.mean()
    order = _order(geo.index)
    e = energy_by_model(fe).reindex(order)
    x = np.arange(len(order))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.bar(x, geo.reindex(order), color="#4C72B0")
    a1.axhline(0.5642, ls="--", color="gray", lw=1,
               label="holdout test-pixel majority (0.564)")
    a1.set_xticks(x); a1.set_xticklabels([LBL[m] for m in order], rotation=45, ha="right")
    a1.set_ylabel("Overall accuracy (geo-holdout)"); a1.set_ylim(0.45, 0.60)
    a1.legend(fontsize=8); a1.set_title("(a) Accuracy")
    w = 0.4
    a2.bar(x - w / 2, e["train"], w, label="train", color="#DD8452")
    a2.bar(x + w / 2, e["infer"], w, label="full-scene inference", color="#55A868")
    a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels([LBL[m] for m in order], rotation=45, ha="right")
    a2.set_ylabel("Energy (uWh, log)"); a2.legend(fontsize=8); a2.set_title("(b) Energy by stage")
    plt.tight_layout(); plt.savefig(figdir / "fig3_accuracy_energy.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig3_accuracy_energy.png")


def fig_perclass(summ, figdir):
    g = summ[summ.protocol == "geo_holdout"].groupby("model")[["F1_Sand", "F1_Seagrass", "F1_Coral"]].mean()
    order = _order(g.index); g = g.reindex(order); x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w, g.F1_Sand, w, label="Sand/Bare", color=CLR["sand"])
    ax.bar(x, g.F1_Seagrass, w, label="Seagrass", color=CLR["sea"])
    ax.bar(x + w, g.F1_Coral, w, label="Coral/Hard-bottom", color=CLR["coral"])
    ax.set_xticks(x); ax.set_xticklabels([LBL[m] for m in order], rotation=45, ha="right")
    ax.set_ylabel("Per-class F1 (geo-holdout)"); ax.set_ylim(0, 0.72); ax.legend(fontsize=9, ncol=3)
    plt.tight_layout(); plt.savefig(figdir / "fig4_perclass_f1.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig4_perclass_f1.png")


def fig_confusion(conf, figdir):
    if "protocol" in conf.columns:
        conf = conf[conf.protocol == "geo_holdout"]
    short = ["Sand", "Seagrass", "Coral"]

    def rownorm(model):
        sub = conf[conf.model == model]; M = np.zeros((3, 3))
        for _, r in sub.iterrows():
            if r["true"] in CLASSES and r["pred"] in CLASSES:
                M[CLASSES.index(r["true"]), CLASSES.index(r["pred"])] += r["count"]
        s = M.sum(1, keepdims=True); return np.divide(M, s, out=np.zeros_like(M), where=s > 0)

    have = [m for m in ["LogReg", "2D_CNN"] if m in set(conf.model)]
    if not have:
        print("  fig5 skipped (LogReg/2D_CNN not in confusion)"); return
    fig, axs = plt.subplots(1, len(have), figsize=(5.5 * len(have), 4.6), squeeze=False)
    for ax, m in zip(axs[0], have):
        rn = rownorm(m); im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(short); ax.set_yticks(range(3)); ax.set_yticklabels(short)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Reference (URM)"); ax.set_title(LBL[m])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center",
                        color="white" if rn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axs[0].tolist(), fraction=0.046, pad=0.04, label="proportion of reference class")
    plt.savefig(figdir / "fig5_confusion.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig5_confusion.png")


def fig_pareto(summ, fe, figdir):
    geo = summ[summ.protocol == "geo_holdout"].groupby("model").OA_mean.mean()
    e = energy_by_model(fe); order = _order(geo.index)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in order:
        if m not in e.index or not np.isfinite(e.loc[m, "operational"]):
            continue
        ax.scatter(e.loc[m, "operational"], geo[m], s=40, color="#4C72B0")
        ax.annotate(LBL[m], (e.loc[m, "operational"], geo[m]), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Operational energy: train + full-scene inference (uWh, log)")
    ax.set_ylabel("Overall accuracy (geo-holdout)"); ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(figdir / "fig6_accuracy_vs_energy.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig6_accuracy_vs_energy.png")


def fig_learning_curve(lc, figdir):
    sub = lc[lc.protocol == "spatial_cv"].pivot_table(index="nsamp", columns="model",
                                                      values="OA_mean", aggfunc="mean")
    col = {"LogReg": "#CC6677", "RF": "#4C72B0", "MLP": "#117733", "2D_CNN": "#9467bd"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in [c for c in ["LogReg", "RF", "MLP", "2D_CNN"] if c in sub.columns]:
        ax.plot(sub.index, sub[m], "o-", label=LBL[m], color=col.get(m))
    ax.set_xlabel("Training samples per class"); ax.set_ylabel("Overall accuracy (spatial CV)")
    ax.legend(); ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(figdir / "fig9_learning_curve.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig9_learning_curve.png")


def fig_bathymetry(bath, figdir):
    bg = bath[bath.protocol == "geo_holdout"].groupby("feature_set")[
        ["F1_Sand", "F1_Seagrass", "F1_Coral"]].mean()
    if not {"spectral", "spectral_plus_depth"}.issubset(bg.index):
        print("  fig10 skipped (bathymetry feature sets missing)"); return
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w / 2, bg.loc["spectral"].values, w, label="spectral", color="#4C72B0")
    ax.bar(x + w / 2, bg.loc["spectral_plus_depth"].values, w, label="spectral + depth", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Per-class F1 (geo-holdout, mean over models)"); ax.set_ylim(0, 0.72); ax.legend()
    plt.tight_layout(); plt.savefig(figdir / "fig10_bathymetry.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  fig10_bathymetry.png")


def main():
    ap = argparse.ArgumentParser(description="Regenerate manuscript figures from outputs.")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--figdir", default=str(HERE / "figures"))
    a = ap.parse_args()
    O = Path(a.out); figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)

    def load(name):
        p = O / name
        return pd.read_csv(p) if p.exists() else None

    summ = load("benchmark_summary.csv"); fe = load("energy_per_stage.csv")
    conf = load("benchmark_confusion.csv"); lc = load("ablation_learning_curve.csv")
    bath = load("ablation_bathymetry.csv")

    print("Writing figures:")
    if summ is not None and fe is not None:
        fig_accuracy_energy(summ, fe, figdir); fig_pareto(summ, fe, figdir)
    else:
        print("  fig3/fig6 skipped (need benchmark_summary.csv + energy_per_stage.csv)")
    if summ is not None:
        fig_perclass(summ, figdir)
    else:
        print("  fig4 skipped (need benchmark_summary.csv)")
    if conf is not None:
        fig_confusion(conf, figdir)
    else:
        print("  fig5 skipped (need benchmark_confusion.csv)")
    if lc is not None:
        fig_learning_curve(lc, figdir)
    else:
        print("  fig9 skipped (need ablation_learning_curve.csv)")
    if bath is not None:
        fig_bathymetry(bath, figdir)
    else:
        print("  fig10 skipped (need ablation_bathymetry.csv)")
    print(f"Done -> {figdir}")


if __name__ == "__main__":
    main()
