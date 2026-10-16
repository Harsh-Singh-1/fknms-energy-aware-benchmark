"""
e_ablations_buffered.py  

Re-runs the three ablations under the SAME protocol as the headline results, so
their conclusions rest on the same evaluation as the main benchmark.

All three ablations run through the same buffered evaluation as 03_benchmark.py:
    - 300 m (30 px) train-test buffer on both spatial-CV folds and the geo holdout
    - every model trained on 5000/class (no per-model cap)
    - patch=9, epochs=60 (matching the reported CNN)

Ablations (choose with --ablation, default all three):
    dii             stack with vs without the 3 Lyzenga DII features
    bathymetry      stack with vs without an independent CUDEM depth channel
    learning_curve  spatial-CV accuracy at 500/1000/2000/5000 samples per class

Outputs (outputs/):
    ablation_dii.csv          feature_set = spectral | no_dii
    ablation_bathymetry.csv   feature_set = spectral | spectral_plus_depth
    ablation_learning_curve.csv   one row per (nsamp, scene, model, protocol)

Speed: dii and bathymetry skip the CNN by default (feature effects show clearly
on the tabular models + MLP, and the CNN roughly triples runtime); the learning
curve includes the CNN because Figure 9 is about it. Add/remove with --models /
--lc_models. test_cap keeps each fold's test set at <=100k pixels, so these are
far cheaper than full-scene inference.

Bathymetry needs the grid-aligned CUDEM raster the original e13 cached at
analysis/data/cudem/depth_aligned_third_<H>x<W>.tif. Point --depth_raster at it
(default tries that path). If it is missing, run the CUDEM downloader (see 01_download_data.py) once
to download/build it, or pass any grid-aligned depth GeoTIFF.

Run:
    python e_ablations_buffered.py --urm outputs/urm_keys.geojson --label i7_16GB
    python e_ablations_buffered.py --ablation dii bathymetry        # the fast two
    python e_ablations_buffered.py --ablation learning_curve        # the slow one

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fknms_core as fk       # noqa: E402
import urm_labels as ul       # noqa: E402
import spatial_buffer as sb   # noqa: E402
from eval_engine import (eval_all, pixel_blocks, gather_test,   # noqa: E402
                               _perf1_from_cm, BENTHIC)

NODII = ["B02", "B03", "B04", "B08", "NDVI", "NDWI", "SAVI"]


def load_urm(urm_file, crs):
    import geopandas as gpd
    return gpd.read_file(urm_file).to_crs(crs)


def _row(scene, protocol, model, oa, ka, mf, f1, extra):
    row = dict(scene=scene, protocol=protocol, model=model,
               OA_mean=round(oa[0], 4), OA_lo=round(oa[1], 4), OA_hi=round(oa[2], 4),
               kappa_mean=round(ka[0], 4), macroF1_mean=round(mf[0], 4))
    for cname in ["Sand/Bare", "Seagrass", "Coral/Hard-bottom"]:
        key = "F1_" + cname.split("/")[0].replace(" ", "")
        row[key] = round(f1.get(cname, float("nan")), 4) if cname in f1 else ""
    row.update(extra)
    return row


def buffered_benchmark(a, gdf, extra, tab_models, deep_models, nsamp,
                       drop_dii=False, add_channel=None):
    """the shared benchmark with the 300 m buffer added and no caps. Returns rows."""
    all_models = tab_models + deep_models
    codes = sorted(BENTHIC)
    rows = []
    tag = ", ".join(f"{k}={v}" for k, v in extra.items())
    for sd in fk.find_scenes(a.data):
        meta = fk.read_metadata(sd); scene = meta["label"]
        print(f"\n=== {scene} ({meta['date']}) [{tag}] ===", flush=True)
        st = fk.prepare_scene(sd)
        if drop_dii:
            st["stack"] = fk.build_stack(st["feats"], fnames=NODII)
        if add_channel is not None:
            if add_channel.shape != tuple(st["geo"]["shape"]):
                raise SystemExit(f"depth shape {add_channel.shape} != grid {tuple(st['geo']['shape'])}")
            st["stack"] = np.concatenate(
                [st["stack"], add_channel[..., None].astype(np.float32)], axis=-1)
        lab, fld = ul.build_label_raster(st, gdf, field=a.field, include_deep=False)
        present = [c for c in codes if (lab == c).sum() >= 200]
        if len(present) < 2:
            print("  <2 usable classes; skipping."); continue
        lm_all = ul.label_masks(lab, st, present)
        pb = pixel_blocks(st["geo"]["shape"], a.n_side)
        labeled = st["water"] & (lab > 0)
        rng = np.random.default_rng(a.seed)
        blocks = np.unique(pb[labeled]); rng.shuffle(blocks)
        fold_of = {b: i % a.k for i, b in enumerate(blocks)}

        # buffered spatial k-fold
        per = {m: {"OA": [], "kappa": [], "macroF1": []} for m in all_models}
        pooled = {m: np.zeros((len(present), len(present)), np.int64) for m in all_models}
        for f in range(a.k):
            trb = [b for b in blocks if fold_of[b] != f]
            teb = [b for b in blocks if fold_of[b] == f]
            trm = labeled & np.isin(pb, trb)
            tem = sb.buffered_test_mask(labeled & np.isin(pb, teb), trm, a.buffer_px)
            if trm.sum() < 500 or tem.sum() < 200:
                continue
            Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=nsamp,
                                                   seed=a.seed, restrict_mask=trm)
            Xte, yte, Rte, Cte = gather_test(st["stack"], lab, tem, a.test_cap, a.seed + f)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            res = eval_all(tab_models, deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, a.patch, a.epochs, a.seed,
                           cap_models=[], cap_nsamp=nsamp)
            for m, (met, _, _) in res.items():
                per[m]["OA"].append(met["OA"]); per[m]["kappa"].append(met["kappa"])
                per[m]["macroF1"].append(met["macroF1"]); pooled[m] += met["cm"]
        for m in all_models:
            if not per[m]["OA"]:
                continue
            oa = fk.mean_ci(per[m]["OA"]); ka = fk.mean_ci(per[m]["kappa"]); mf = fk.mean_ci(per[m]["macroF1"])
            rows.append(_row(scene, "spatial_cv", m, oa, ka, mf,
                             _perf1_from_cm(pooled[m], present), extra))
            print(f"    [cv]  {m:11s} OA={oa[0]:.4f}  kappa={ka[0]:.4f}  macroF1={mf[0]:.4f}", flush=True)

        # buffered geo holdout (west -> east)
        west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
        trm = labeled & west
        tem = sb.buffered_test_mask(labeled & ~west, trm, a.buffer_px)
        Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=nsamp,
                                               seed=a.seed, restrict_mask=trm)
        Xte, yte, Rte, Cte = gather_test(st["stack"], lab, tem, a.test_cap, a.seed)
        if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
            res = eval_all(tab_models, deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, a.patch, a.epochs, a.seed,
                           cap_models=[], cap_nsamp=nsamp)
            for m in all_models:
                met, _en, _yp = res[m]
                f1 = {fk.CLASSES[c]: met["perF1"][fk.CLASSES[c]] for c in present}
                rows.append(_row(scene, "geo_holdout", m, (met["OA"],) * 3,
                                 (met["kappa"],) * 3, (met["macroF1"],) * 3, f1, extra))
                print(f"    [geo] {m:11s} OA={met['OA']:.4f}  kappa={met['kappa']:.4f}  "
                      f"macroF1={met['macroF1']:.4f}", flush=True)
    return rows


def write_csv(rows, path):
    if not rows:
        print(f"  (no rows for {path.name})"); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Saved {path}", flush=True)


def load_depth(path, shape):
    import rasterio
    if not path or not Path(path).exists():
        return None
    with rasterio.open(path) as ds:
        d = ds.read(1).astype(np.float32)
    if d.shape != shape:
        print(f"  depth raster shape {d.shape} != grid {shape}; skipping bathymetry."); return None
    return d


def main():
    ap = argparse.ArgumentParser(description="Buffered re-run of the DII, bathymetry, and learning-curve ablations.")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=str(HERE / "data" / "urm_keys.geojson"))
    ap.add_argument("--field", default=None)
    ap.add_argument("--label", default=fk.cpu_label())
    ap.add_argument("--ablation", nargs="+", default=["dii", "bathymetry", "learning_curve"],
                    choices=["dii", "bathymetry", "learning_curve"])
    ap.add_argument("--models", nargs="+", default=fk.TABULAR_MODELS + ["MLP"],
                    help="models for dii / bathymetry (CNN off by default)")
    ap.add_argument("--lc_models", nargs="+", default=["LogReg", "RF", "MLP", "2D_CNN"],
                    help="models for the learning curve")
    ap.add_argument("--lc_sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    ap.add_argument("--depth_raster", default=str(HERE / "data" / "cudem" / "depth.tif"))
    ap.add_argument("--buffer_px", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--nsamp", type=int, default=5000)
    ap.add_argument("--test_cap", type=int, default=100000)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    crs0 = fk.prepare_scene(fk.find_scenes(a.data)[0])["geo"]["crs"]
    gdf = load_urm(a.urm, crs0)
    print(f"URM polygons: {len(gdf):,}   buffer={a.buffer_px * 10} m   nsamp={a.nsamp}/class   "
          f"caps=none   models(dii/bath)={a.models}", flush=True)

    def split_models(models):
        deep = [m for m in models if m in ("MLP", "2D_CNN")]
        tab = [m for m in models if m not in ("MLP", "2D_CNN")]
        return tab, deep

    if "dii" in a.ablation:
        tab, deep = split_models(a.models)
        rows = []
        rows += buffered_benchmark(a, gdf, {"feature_set": "spectral"}, tab, deep, a.nsamp, drop_dii=False)
        rows += buffered_benchmark(a, gdf, {"feature_set": "no_dii"}, tab, deep, a.nsamp, drop_dii=True)
        write_csv(rows, Path(a.out) / "ablation_dii.csv")

    if "bathymetry" in a.ablation:
        shape = tuple(fk.prepare_scene(fk.find_scenes(a.data)[0])["geo"]["shape"])
        depth = load_depth(a.depth_raster, shape)
        if depth is None:
            print("Bathymetry skipped: no grid-aligned CUDEM depth raster found "
                  f"at {a.depth_raster}. Run the CUDEM downloader (see 01_download_data.py) once to build it, "
                  "or pass --depth_raster.", flush=True)
        else:
            over = depth[depth > 0]
            print(f"depth raster ok: median={np.median(over):.1f} m range=[{over.min():.1f},{over.max():.1f}] m", flush=True)
            tab, deep = split_models(a.models)
            rows = []
            rows += buffered_benchmark(a, gdf, {"feature_set": "spectral"}, tab, deep, a.nsamp, add_channel=None)
            rows += buffered_benchmark(a, gdf, {"feature_set": "spectral_plus_depth"}, tab, deep, a.nsamp, add_channel=depth)
            write_csv(rows, Path(a.out) / "ablation_bathymetry.csv")

    if "learning_curve" in a.ablation:
        tab, deep = split_models(a.lc_models)
        rows = []
        for ns in a.lc_sizes:
            print(f"\n########## learning curve: nsamp={ns}/class ##########", flush=True)
            rows += buffered_benchmark(a, gdf, {"nsamp": ns}, tab, deep, ns)
        write_csv(rows, Path(a.out) / "ablation_learning_curve.csv")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
