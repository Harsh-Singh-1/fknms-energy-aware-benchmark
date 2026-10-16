"""
02_pseudo_vs_urm.py

Two comparisons that together demonstrate the circularity of the spectral
pseudo-label protocol on identical imagery:

  A. Internal accuracy of the pseudo-label pipeline (Table 5). A classifier is
     trained and scored on the same adaptive spectral pseudo-labels, using a
     held-out split. Because the labels are generated from the same spectral
     features the classifier reads, accuracy is very high (about 0.98 to 0.99).
     This measures self-consistency with the labeling rule, not agreement with
     the seabed.

  B. Agreement with the field reference (Table 6). The same pipeline is compared
     against the field-validated Unified Reef Map over the mapped area, with a
     3x3 confusion matrix and per-scene, per-class agreement. The pipeline agrees
     only about 25 percent of the time and collapses toward a single class.

The distance between A (about 99 percent) and B (about 25 percent) on the same
scenes is the paper's central caution, and reporting both sides is how that gap
is shown rather than asserted.

Method, per scene:
  1. Generate the spectral pseudo-labels (fknms_core.make_labels, 4 classes).
  2. A: split the pseudo-labels into train/test, fit, and score on the held-out
     part -> internal overall accuracy and kappa.
  3. B: train on the pseudo-labels, predict the class of every water pixel, and
     compare the prediction to URM.

Class alignment for B (as described in the manuscript): the comparison is
restricted to pixels where URM is benthic (Sand/Seagrass/Coral) AND the
prediction is benthic. Deep-water predictions are excluded from the 3x3
comparison and accounted for separately (how many fall in URM-unmapped water vs
on URM benthic). Deep water is not a URM class, so it never appears on the URM
side.

Outputs (outputs/):
  pseudo_internal_accuracy.csv  per scene x classifier: internal OA and kappa (Table 5)
  pseudo_vs_urm_perscene.csv    per scene x classifier: URM-agreement OA, kappa, per-class F1, counts
  pseudo_vs_urm_confusion.csv   pooled 3x3 confusion (URM row x predicted col) + agreement (Table 6)

Run:  python 02_pseudo_vs_urm.py
      Internal accuracy uses RF, SVM, XGBoost, and MLP (the Table 5 columns) on
      every scene; the three 2024 seasonal scenes reproduce Table 5. The URM
      comparison uses RF and XGBoost by default (both reach ~98-99 percent on
      their own labels yet ~25 percent against URM).

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fknms_core as fk       # noqa: E402
import urm_labels as ul       # noqa: E402

BENTHIC = [2, 3, 4]
NAME = {2: "Sand/Bare", 3: "Seagrass", 4: "Coral/Hard-bottom"}


def load_urm(urm_file, crs):
    import geopandas as gpd
    return gpd.read_file(urm_file).to_crs(crs)


def internal_predict(model, Xtr, ytr, Xte):
    """Fit on pseudo-labels and predict the held-out split; labels are 1..4."""
    if model == "MLP":
        return fk.TorchMLP(seed=42).fit(Xtr, ytr).predict(Xte)
    return fk.fit_predict(fk.make_model(model, 42), Xtr, ytr, Xte)


def main():
    ap = argparse.ArgumentParser(
        description="Pseudo-label internal accuracy (Table 5) and URM agreement (Table 6).")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=str(HERE / "data" / "urm_keys.geojson"))
    ap.add_argument("--field", default=None, help="URM class field (default: auto, Level 1)")
    ap.add_argument("--models", nargs="+", default=["RF", "XGBoost"],
                    help="models for the URM comparison (Table 6)")
    ap.add_argument("--internal_models", nargs="+", default=["RF", "SVM", "XGBoost", "MLP"],
                    help="models for the internal-accuracy table (Table 5 columns)")
    ap.add_argument("--nsamp", type=int, default=400, help="pseudo-label samples per class (protocol = 400)")
    ap.add_argument("--test_frac", type=float, default=0.3, help="held-out fraction for internal accuracy")
    ap.add_argument("--skip_internal", action="store_true", help="skip the Table 5 internal-accuracy pass")
    ap.add_argument("--label", default=fk.cpu_label())
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    gdf = None
    internal_rows, per_rows = [], []
    pooled = {m: np.zeros((3, 3), dtype=np.int64) for m in a.models}
    deepacc = {m: dict(pred_deep_total=0, pred_deep_in_unmapped=0, urm_benthic_pred_deep=0) for m in a.models}

    for sd in fk.find_scenes(a.data):
        meta = fk.read_metadata(sd); scene = meta["label"]
        print(f"\n=== {scene} ({meta['date']}) ===", flush=True)
        state = fk.prepare_scene(sd)
        Xtr, ytr, Rtr, Ctr = fk.sample_labeled(state["stack"], state["labels"], n=a.nsamp, seed=42)

        # --- A. internal accuracy on held-out pseudo-labels (Table 5) ---
        if not a.skip_internal:
            Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=a.test_frac,
                                              random_state=42, stratify=ytr)
            for m in a.internal_models:
                pred = internal_predict(m, Xa, ya, Xb)
                oa = float(accuracy_score(yb, pred))
                kappa = float(cohen_kappa_score(yb, pred))
                internal_rows.append(dict(scene=scene, model=m, n_train=int(len(ya)), n_test=int(len(yb)),
                                          OA=round(oa, 4), kappa=round(kappa, 4)))
                print(f"  [internal] {m:9s} OA={oa:.4f}  kappa={kappa:.4f}", flush=True)

        # --- B. agreement with URM over the full scene (Table 6) ---
        if gdf is None:
            gdf = load_urm(a.urm, state["geo"]["crs"])
            print(f"  URM polygons: {len(gdf):,}", flush=True)
        urm_lab, fld = ul.build_label_raster(state, gdf, field=a.field, include_deep=False)
        rr, cc, Xall = fk.water_pixels(state)
        urm_px = urm_lab[rr, cc]                       # 0 (unmapped) or 2/3/4
        print(f"  URM field={fld}  train={len(ytr):,}  water px={len(rr):,}  "
              f"URM-benthic px={int(np.isin(urm_px, BENTHIC).sum()):,}", flush=True)

        for m in a.models:
            pred = fk.fit_predict(fk.make_model(m, 42), Xtr, ytr, Xall)   # values in 1..4
            pred_deep = (pred == 1)
            deepacc[m]["pred_deep_total"] += int(pred_deep.sum())
            deepacc[m]["pred_deep_in_unmapped"] += int((pred_deep & (urm_px == 0)).sum())
            deepacc[m]["urm_benthic_pred_deep"] += int((pred_deep & np.isin(urm_px, BENTHIC)).sum())

            comp = np.isin(urm_px, BENTHIC) & np.isin(pred, BENTHIC)      # aligned, deep excluded both sides
            ut, pp = urm_px[comp], pred[comp]
            if len(ut) == 0:
                print(f"  {m:9s} (no aligned pixels)", flush=True)
                continue
            cmat = confusion_matrix(ut, pp, labels=BENTHIC)
            pooled[m] += cmat
            oa = accuracy_score(ut, pp)
            kappa = cohen_kappa_score(ut, pp, labels=BENTHIC)
            perf1 = {c: float(f1_score(ut, pp, labels=[c], average="macro", zero_division=0)) for c in BENTHIC}
            per_rows.append(dict(scene=scene, model=m, n_compared=int(comp.sum()),
                                 OA=round(float(oa), 4), kappa=round(float(kappa), 4),
                                 F1_Sand=round(perf1[2], 4), F1_Seagrass=round(perf1[3], 4),
                                 F1_Coral=round(perf1[4], 4),
                                 pred_deep_total=int(pred_deep.sum()),
                                 urm_benthic_pred_deep=int((pred_deep & np.isin(urm_px, BENTHIC)).sum())))
            print(f"  {m:9s} agreement OA={oa:.3f}  kappa={kappa:.3f}  n_compared={comp.sum():,}", flush=True)

    # --- Table 5: internal accuracy of the pseudo-label baseline ---
    if internal_rows:
        p0 = Path(a.out) / "pseudo_internal_accuracy.csv"
        with open(p0, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(internal_rows[0].keys()))
            w.writeheader(); w.writerows(internal_rows)
        print(f"\nSaved {p0}  (Table 5: internal accuracy of the pseudo-label baseline)", flush=True)

    # --- Table 6 and its per-scene detail: agreement with the field reference ---
    p1 = Path(a.out) / "pseudo_vs_urm_perscene.csv"
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys())); w.writeheader(); w.writerows(per_rows)
    p2 = Path(a.out) / "pseudo_vs_urm_confusion.csv"
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        for m in a.models:
            cm = pooled[m]; tot = int(cm.sum())
            oa = float(np.trace(cm) / tot) if tot else float("nan")
            w.writerow([f"# model={m}  pooled 3x3 (row=URM reference, col=pseudo-label prediction)  "
                        f"overall_agreement={oa:.4f}  N={tot}"])
            w.writerow(["URM \\ pred", "Sand/Bare", "Seagrass", "Coral/Hard-bottom", "row_total"])
            for i, c in enumerate(BENTHIC):
                w.writerow([NAME[c]] + [int(x) for x in cm[i]] + [int(cm[i].sum())])
            w.writerow(["col_total"] + [int(x) for x in cm.sum(0)] + [tot])
            d = deepacc[m]
            w.writerow([f"# predicted deep water: total={d['pred_deep_total']}  "
                        f"in_URM_unmapped={d['pred_deep_in_unmapped']}  on_URM_benthic={d['urm_benthic_pred_deep']}"])
            w.writerow([])
            print(f"[{m}] POOLED overall agreement with URM = {oa:.4f}  (N={tot:,})", flush=True)
    print(f"Saved {p1}\nSaved {p2}", flush=True)


if __name__ == "__main__":
    main()
