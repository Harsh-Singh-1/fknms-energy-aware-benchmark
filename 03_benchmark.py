"""
03_benchmark.py -- URM-supervised benchmark under spatial validation.

Trains nine classifiers on independent, field-validated Unified Reef Map labels
and evaluates them with two spatial protocols, producing the paper's primary
accuracy tables (overall accuracy, Cohen's kappa, per-class F1) plus a per-block
export for the spatial statistics.

Two design points make the evaluation honest:

  Train-test buffer.
    Both the spatial-block cross-validation and the west/east geographic holdout
    enforce a minimum train-test separation: test pixels within --buffer_px of
    the training region are dropped (spatial_buffer.py), so no test pixel is
    adjacent to a training pixel. --buffer_px 0 disables it.

  Equal training sample.
    Every model is trained on --nsamp samples per class (5000 by default) with no
    per-model cap, so accuracy and energy are directly comparable.

A per-block 3x3 confusion export (benchmark_block_metrics.csv) lets the block
bootstrap and scene-level tests run afterward without refitting any model.

Run:
  python 03_benchmark.py                       # full run (both protocols)
  python 03_benchmark.py --buffer_px 0         # disable the buffer
  for b in 0 30 60; do python 03_benchmark.py --buffer_px $b --out outputs_buf$b; done
Smoke test:
  python 03_benchmark.py --nsamp 300 --k 3 --n_side 3 --tab_models RF XGBoost \
         --deep_models MLP --no-figures --out outputs_smoke

Data and URM default to ./data and ./data/urm_keys.geojson (override --data / --urm).

Outputs (to --out, default ./outputs):
  benchmark_summary.csv        per scene/model/protocol: OA, kappa, per-class F1, fold CIs, train energy
  benchmark_confusion.csv      geo-holdout pooled confusion per scene/model
  benchmark_block_metrics.csv  per scene/protocol/block/model 3x3 confusion (for the bootstrap)
  figures/<scene> maps         best geo-holdout model map (unless --no-figures)

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # import the local core copies in this folder

import fknms_core as fk                     # noqa: E402
import urm_labels as ul                     # noqa: E402
from eval_engine import (             # noqa: E402
    eval_all, gather_test, pixel_blocks, _perf1_from_cm, _srow, _figure, BENTHIC,
)
import spatial_buffer as sb                 # noqa: E402


# --------------------------------------------------------------------------
# Local data / URM defaults
# --------------------------------------------------------------------------
def default_data():
    return str(HERE / "data")


def default_urm():
    p = HERE / "data" / "urm_keys.geojson"
    return str(p) if p.exists() else None


def load_urm_gdf(args):
    """Fetch URM once using the first scene's bbox/CRS (same path as E8)."""
    import geopandas as gpd
    scenes = fk.find_scenes(args.data)
    if not scenes:
        raise SystemExit(
            f"No scenes found under {args.data}. Copy the nine scene_* folders "
            f"from analysis/data into {HERE / 'data'} (see README), or pass --data.")
    if not args.download and (not args.urm or not Path(args.urm).exists()):
        raise SystemExit(
            "URM reference not found. It ships with this package as data/urm_keys.geojson. "
            f"{HERE / 'outputs'}, or pass --urm <file> / --download --service_url <url>.")
    st0 = fk.prepare_scene(scenes[0])
    H0, W0 = st0["geo"]["shape"]
    tf0 = st0["geo"]["transform"]
    crs0 = st0["geo"]["crs"]
    left, top = tf0 * (0, 0)
    right, bottom = tf0 * (W0, H0)
    g = gpd.GeoSeries.from_wkt(
        [f"POINT({left} {top})", f"POINT({right} {bottom})"], crs=crs0).to_crs(4326)
    bbox = (min(g.x), min(g.y), max(g.x), max(g.y))
    return ul.fetch_urm(bbox, crs0, urm=args.urm, download=args.download,
                        service_url=args.service_url, save_urm=args.save_urm)


def _write(path, rows):
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(args):
    if not fk.HAS_TORCH and args.deep_models:
        print("PyTorch not found; running tabular models only.")
        args.deep_models = []
    all_models = args.tab_models + args.deep_models
    codes = sorted(BENTHIC + ([1] if args.include_deep else []))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)

    gdf = load_urm_gdf(args)
    print(f"URM polygons: {len(gdf)}")
    print(f"buffer_px={args.buffer_px} ({args.buffer_px * 10} m)   "
          f"nsamp={args.nsamp}/class   "
          f"capped={args.cap_models or 'none'}"
          + (f"@{args.cap_nsamp}" if args.cap_models else "")
          + f"   models={all_models}")

    summ, conf, mcn, blockrows = [], [], [], []

    for sd in fk.find_scenes(args.data):
        meta = fk.read_metadata(sd)
        scene = meta["label"]
        print(f"\n=== {scene} ({meta['date']}) ===")
        st = fk.prepare_scene(sd)
        lab, fld = ul.build_label_raster(st, gdf, field=args.field,
                                         include_deep=args.include_deep)
        present = [c for c in codes if (lab == c).sum() >= 200]
        if len(present) < 2:
            print("  <2 usable classes; skipping.")
            continue
        print(f"  labels: { {fk.CLASSES[c]: int((lab == c).sum()) for c in present} }")

        lm_all = ul.label_masks(lab, st, present)
        pb = pixel_blocks(st["geo"]["shape"], args.n_side)
        labeled = st["water"] & (lab > 0)
        rng = np.random.default_rng(args.seed)
        blocks = np.unique(pb[labeled])
        rng.shuffle(blocks)
        fold_of = {b: i % args.k for i, b in enumerate(blocks)}

        # ---------- buffered spatial k-fold ----------
        per = {m: {"OA": [], "kappa": [], "macroF1": []} for m in all_models}
        pooled_cm = {m: np.zeros((len(present), len(present)), np.int64) for m in all_models}
        for f in range(args.k):
            trb = [b for b in blocks if fold_of[b] != f]
            teb = [b for b in blocks if fold_of[b] == f]
            tr_pool = labeled & np.isin(pb, trb)
            te_all = labeled & np.isin(pb, teb)
            te_buf = sb.buffered_test_mask(te_all, tr_pool, args.buffer_px)
            if tr_pool.sum() < 500 or te_buf.sum() < 200:
                continue
            Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=args.nsamp,
                                                   seed=args.seed, restrict_mask=tr_pool)
            Xte, yte, Rte, Cte = gather_test(st["stack"], lab, te_buf,
                                             args.test_cap, args.seed + f)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            res = eval_all(args.tab_models, args.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, args.patch, args.epochs, args.seed,
                           cap_models=args.cap_models, cap_nsamp=args.cap_nsamp)
            blk = pb[Rte, Cte]
            for m, (met, _en, yp) in res.items():
                per[m]["OA"].append(met["OA"])
                per[m]["kappa"].append(met["kappa"])
                per[m]["macroF1"].append(met["macroF1"])
                pooled_cm[m] += met["cm"]
                for b, cmb in sb.block_confusions(yte, yp, blk).items():
                    row = dict(scene=scene, protocol="spatial_cv", fold=f, block=b,
                               model=m, n=int(cmb.sum()))
                    row.update(sb.flatten_cm(cmb))
                    blockrows.append(row)
        for m in all_models:
            if not per[m]["OA"]:
                continue
            oa = fk.mean_ci(per[m]["OA"])
            ka = fk.mean_ci(per[m]["kappa"])
            mf = fk.mean_ci(per[m]["macroF1"])
            f1 = _perf1_from_cm(pooled_cm[m], present)
            summ.append(_srow(scene, "spatial_cv", m, oa, ka, mf, f1, present, float("nan")))
            print(f"    [cv]  {m:11s} OA={oa[0]:.4f} [{oa[1]:.4f},{oa[2]:.4f}] "
                  f"kappa={ka[0]:.4f} macroF1={mf[0]:.4f}")

        # ---------- buffered geo holdout (west -> east) ----------
        west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
        tr_pool = labeled & west
        te_all = labeled & ~west
        te_buf = sb.buffered_test_mask(te_all, tr_pool, args.buffer_px)
        Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=args.nsamp,
                                               seed=args.seed, restrict_mask=tr_pool)
        Xte, yte, Rte, Cte = gather_test(st["stack"], lab, te_buf, args.test_cap, args.seed)
        best_model, best_oa = None, -1.0
        if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
            res = eval_all(args.tab_models, args.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, args.patch, args.epochs, args.seed,
                           measure_energy=True,
                           cap_models=args.cap_models, cap_nsamp=args.cap_nsamp)
            blk = pb[Rte, Cte]
            geo_preds = {}
            for m in all_models:
                met, en, yp = res[m]
                geo_preds[m] = yp
                f1 = {fk.CLASSES[c]: met["perF1"][fk.CLASSES[c]] for c in present}
                summ.append(_srow(scene, "geo_holdout", m, (met["OA"],) * 3,
                                  (met["kappa"],) * 3, (met["macroF1"],) * 3, f1, present, en))
                for i, ct in enumerate(present):
                    for j, cp in enumerate(present):
                        conf.append(dict(scene=scene, model=m, protocol="geo_holdout",
                                         true=fk.CLASSES[ct], pred=fk.CLASSES[cp],
                                         count=int(met["cm"][i, j])))
                for b, cmb in sb.block_confusions(yte, yp, blk).items():
                    row = dict(scene=scene, protocol="geo_holdout", fold=-1, block=b,
                               model=m, n=int(cmb.sum()))
                    row.update(sb.flatten_cm(cmb))
                    blockrows.append(row)
                print(f"    [geo] {m:11s} OA={met['OA']:.4f} kappa={met['kappa']:.4f} "
                      f"macroF1={met['macroF1']:.4f} E={en:10.1f}uWh")
                if met["OA"] > best_oa:
                    best_oa, best_model = met["OA"], m
            for a, b in itertools.combinations(list(geo_preds), 2):
                t = fk.mcnemar_test(yte, geo_preds[a], geo_preds[b])
                mcn.append(dict(scene=scene, model_a=a, model_b=b, n01=t["n01"],
                                n10=t["n10"], chi2=round(t["chi2"], 3),
                                p=round(t["p"], 4), significant=t["p"] < 0.05))

        if args.figures and best_model:
            _figure(st, lab, present, west, args.map_model or best_model, args, scene)

    out = Path(args.out)
    _write(out / "benchmark_summary.csv", summ)
    _write(out / "benchmark_confusion.csv", conf)
    _write(out / "benchmark_mcnemar.csv", mcn)
    _write(out / "benchmark_block_metrics.csv", blockrows)
    print(f"\nDone. buffer_px={args.buffer_px}, nsamp={args.nsamp}/class, "
          f"capped={args.cap_models or 'none'} -> {out}")


def build_parser():
    ap = argparse.ArgumentParser(
        description="T1 (spatial buffer) + T2 (equal per-class sample) URM benchmark.")
    ap.add_argument("--data", default=default_data())
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--figdir", default=str(HERE / "figures"))
    ap.add_argument("--map_model", default="LogReg",
                    help="classifier used for the map figures (default LogReg, matching the paper); pass an empty string to use each scene's best model")
    ap.add_argument("--urm", default=default_urm(), help="cached URM GeoJSON/shapefile")
    ap.add_argument("--download", action="store_true", help="query the FWC service instead of a cache")
    ap.add_argument("--service_url", default=None)
    ap.add_argument("--save_urm", default=None)
    ap.add_argument("--field", default=None, help="URM class field (auto-detected if omitted)")
    ap.add_argument("--include_deep", action="store_true")
    # T1
    ap.add_argument("--buffer_px", type=int, default=30,
                    help="minimum train-test separation in pixels (x10 m); 0 = original no-buffer split")
    # protocol (defaults match the reported 5000/class run)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--nsamp", type=int, default=5000)
    ap.add_argument("--test_cap", type=int, default=100000)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    # T2: nothing capped by default -> SVM and kNN also train on --nsamp per class
    ap.add_argument("--cap_models", nargs="*", default=[],
                    help="models to train on a capped sample; empty = none (T2). "
                         "Use '--cap_models SVM kNN --cap_nsamp 2000' to reproduce the original caps.")
    ap.add_argument("--cap_nsamp", type=int, default=5000)
    ap.add_argument("--tab_models", nargs="+", default=fk.TABULAR_MODELS)
    ap.add_argument("--deep_models", nargs="+", default=["MLP", "2D_CNN"])
    ap.add_argument("--figures", dest="figures", action="store_true", default=True)
    ap.add_argument("--no-figures", dest="figures", action="store_false")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
