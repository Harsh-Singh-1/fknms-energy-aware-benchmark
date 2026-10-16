"""
E8 - URM-supervised classification (non-circular).

Trains the same model set on real habitat labels rasterized from Florida's
Unified Reef Map instead of spectral pseudo-labels, and validates on spatially
held-out URM blocks. This breaks the circularity the reviewers flagged: labels
come from an independent field-validated source, features come from the imagery,
and train/test pixels never share a spatial block.

Default is the 3-class benthic problem (Sand/Bare, Seagrass, Coral/Hard-bottom)
with labels entirely from URM. Deep Water can be added with --include_deep
(taken from the water mask where URM is unmapped).

For each scene: train on balanced samples from training blocks, then predict
every URM-labeled pixel in the held-out blocks (prevalence-weighted, the real
operational metric). Reports OA/kappa/per-class F1 with CIs over spatial folds,
a west/east geo holdout, training energy, and confusion matrices.

Getting URM (query once and cache):
  python eval_engine.py --download --service_url "<layer>/query" --save_urm outputs/urm_keys.geojson
Then reuse the cached file:
  python eval_engine.py --urm outputs/urm_keys.geojson

Outputs: outputs/e8_urm_summary.csv, outputs/e8_urm_confusion.csv,
figures/e8_urm_<scene>.png
"""
import argparse, csv, itertools
from pathlib import Path
import numpy as np
import fknms_core as fk
import urm_labels as ul

BASE = Path(__file__).parent
BENTHIC = [2, 3, 4]


def _cap_samples(X, y, cap, seed=0):
    """Subsample each class to at most `cap` rows (for models that scale poorly)."""
    if cap is None or cap <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        if len(ci) > cap:
            ci = rng.choice(ci, cap, replace=False)
        idx.append(ci)
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def pixel_blocks(shape, n_side):
    H, W = shape
    gr = (np.arange(H)[:, None] * n_side // H).clip(0, n_side - 1)
    gc = (np.arange(W)[None, :] * n_side // W).clip(0, n_side - 1)
    return (gr * n_side + gc).astype(int)


def gather_test(stack, lab, test_mask, cap, seed):
    """All labeled test pixels (prevalence-preserving), capped to `cap`."""
    tr, tc = np.where(test_mask)
    X = stack[tr, tc]
    ok = np.all(np.isfinite(X), axis=1)
    tr, tc, X = tr[ok], tc[ok], X[ok]
    y = lab[tr, tc]
    if len(y) > cap:
        idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
        tr, tc, X, y = tr[idx], tc[idx], X[idx], y[idx]
    return X, y, tr, tc


def eval_all(models, deep_models, stack, Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
             codes, patch, epochs, seed, measure_energy=False,
             cap_models=(), cap_nsamp=None):
    """Return {model: (metrics, energy_uWh, predictions)} for one split."""
    out = {}
    for m in models:
        Xt, yt = (_cap_samples(Xtr, ytr, cap_nsamp) if m in cap_models else (Xtr, ytr))
        en = float("nan")
        if measure_energy:
            with fk.Energy(m) as e:
                yp = fk.fit_predict(fk.make_model(m, seed), Xt, yt, Xte)
            en = e.energy_uWh
        else:
            yp = fk.fit_predict(fk.make_model(m, seed), Xt, yt, Xte)
        out[m] = (fk.classification_metrics(yte, yp, labels=codes), en, yp)
    if "MLP" in deep_models:
        en = float("nan")
        if measure_energy:
            with fk.Energy("MLP") as e:
                mlp = fk.TorchMLP(seed=seed, epochs=100).fit(Xtr, ytr)
            en = e.energy_uWh
        else:
            mlp = fk.TorchMLP(seed=seed, epochs=100).fit(Xtr, ytr)
        yp = mlp.predict(Xte)
        out["MLP"] = (fk.classification_metrics(yte, yp, labels=codes), en, yp)
    if "2D_CNN" in deep_models:
        en = float("nan")
        if measure_energy:
            with fk.Energy("2D_CNN") as e:
                cnn = fk.Torch2DCNN(stack, seed=seed, patch=patch, epochs=epochs).fit(Rtr, Ctr, ytr)
            en = e.energy_uWh
        else:
            cnn = fk.Torch2DCNN(stack, seed=seed, patch=patch, epochs=epochs).fit(Rtr, Ctr, ytr)
        yp = cnn.predict(Rte, Cte)
        out["2D_CNN"] = (fk.classification_metrics(yte, yp, labels=codes), en, yp)
    return out


def run(args):
    if not fk.HAS_TORCH and args.deep_models:
        print("PyTorch not found; running tabular models only.")
        args.deep_models = []
    all_models = args.tab_models + args.deep_models

    scenes = fk.find_scenes(args.data)
    # fetch URM once, using the first scene's bbox/crs
    st0 = fk.prepare_scene(scenes[0])
    H0, W0 = st0["geo"]["shape"]; tf0 = st0["geo"]["transform"]; crs0 = st0["geo"]["crs"]
    import geopandas as gpd
    left, top = tf0 * (0, 0); right, bottom = tf0 * (W0, H0)
    g = gpd.GeoSeries.from_wkt([f"POINT({left} {top})", f"POINT({right} {bottom})"], crs=crs0).to_crs(4326)
    bbox = (min(g.x), min(g.y), max(g.x), max(g.y))
    gdf = ul.fetch_urm(bbox, crs0, urm=args.urm, download=args.download,
                       service_url=args.service_url, save_urm=args.save_urm)
    print(f"URM polygons: {len(gdf)}")

    summ, conf, mcn = [], [], []
    codes = BENTHIC + ([1] if args.include_deep else [])
    codes = sorted(codes)

    for sd in scenes:
        meta = fk.read_metadata(sd); scene = meta["label"]
        print(f"\n=== {scene} ({meta['date']}) ===")
        st = fk.prepare_scene(sd)
        lab, fld = ul.build_label_raster(st, gdf, field=args.field, include_deep=args.include_deep)
        present = [c for c in codes if (lab == c).sum() >= 200]
        comp = {fk.CLASSES[c]: int((lab == c).sum()) for c in present}
        print(f"  label field: {fld}  composition: {comp}")
        if len(present) < 2:
            print("  <2 usable classes; skipping scene."); continue

        lm_all = ul.label_masks(lab, st, present)
        pb = pixel_blocks(st["geo"]["shape"], args.n_side)
        rng = np.random.default_rng(args.seed)
        labeled = st["water"] & (lab > 0)
        blocks = np.unique(pb[labeled]); rng.shuffle(blocks)
        fold_of = {b: i % args.k for i, b in enumerate(blocks)}

        # ---- spatial k-fold ----
        per = {m: {"OA": [], "kappa": [], "macroF1": []} for m in all_models}
        pooled_cm = {m: np.zeros((len(present), len(present)), dtype=np.int64) for m in all_models}
        for f in range(args.k):
            trb = [b for b in blocks if fold_of[b] != f]
            teb = [b for b in blocks if fold_of[b] == f]
            trm = labeled & np.isin(pb, trb)
            tem = labeled & np.isin(pb, teb)
            if trm.sum() < 500 or tem.sum() < 200:
                continue
            Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=args.nsamp,
                                                   seed=args.seed, restrict_mask=trm)
            Xte, yte, Rte, Cte = gather_test(st["stack"], lab, tem, args.test_cap, args.seed + f)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            res = eval_all(args.tab_models, args.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, args.patch, args.epochs, args.seed,
                           cap_models=args.cap_models, cap_nsamp=args.cap_nsamp)
            for m, (met, _, _) in res.items():
                per[m]["OA"].append(met["OA"]); per[m]["kappa"].append(met["kappa"])
                per[m]["macroF1"].append(met["macroF1"]); pooled_cm[m] += met["cm"]

        print(f"  [spatial_cv k={args.k}]  {'model':11s} OA(95% CI)          kappa    macroF1")
        for m in all_models:
            if not per[m]["OA"]:
                continue
            oa = fk.mean_ci(per[m]["OA"]); ka = fk.mean_ci(per[m]["kappa"]); mf = fk.mean_ci(per[m]["macroF1"])
            f1 = _perf1_from_cm(pooled_cm[m], present)
            summ.append(_srow(scene, "spatial_cv", m, oa, ka, mf, f1, present, float("nan")))
            print(f"    {m:11s} {oa[0]:.4f} [{oa[1]:.4f},{oa[2]:.4f}] {ka[0]:.4f}  {mf[0]:.4f}")

        # ---- geo holdout (west -> east), with energy + confusion + figure ----
        west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
        trm = labeled & west; tem = labeled & ~west
        Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=args.nsamp,
                                               seed=args.seed, restrict_mask=trm)
        Xte, yte, Rte, Cte = gather_test(st["stack"], lab, tem, args.test_cap, args.seed)
        print("  [geo_holdout west->east]")
        best_model, best_oa = None, -1
        if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
            res = eval_all(args.tab_models, args.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xte, yte, Rte, Cte,
                           present, args.patch, args.epochs, args.seed, measure_energy=True,
                           cap_models=args.cap_models, cap_nsamp=args.cap_nsamp)
            geo_preds = {}
            for m in all_models:
                met, en, yp = res[m]
                geo_preds[m] = yp
                f1 = {fk.CLASSES[c]: met["perF1"][fk.CLASSES[c]] for c in present}
                summ.append(_srow(scene, "geo_holdout", m,
                                  (met["OA"],) * 3, (met["kappa"],) * 3, (met["macroF1"],) * 3,
                                  f1, present, en))
                for i, ct in enumerate(present):
                    for j, cp in enumerate(present):
                        conf.append(dict(scene=scene, model=m, protocol="geo_holdout",
                                         true=fk.CLASSES[ct], pred=fk.CLASSES[cp],
                                         count=int(met["cm"][i, j])))
                print(f"    {m:11s} OA={met['OA']:.4f} kappa={met['kappa']:.4f} "
                      f"macroF1={met['macroF1']:.4f} E={en:9.1f}uWh")
                if met["OA"] > best_oa:
                    best_oa, best_model = met["OA"], m
            for a, b in itertools.combinations(list(geo_preds), 2):
                t = fk.mcnemar_test(yte, geo_preds[a], geo_preds[b])
                mcn.append(dict(scene=scene, model_a=a, model_b=b,
                                n01=t["n01"], n10=t["n10"], chi2=round(t["chi2"], 3),
                                p=round(t["p"], 4), significant=t["p"] < 0.05))
            if "SVM" in geo_preds and "LogReg" in geo_preds:
                t = fk.mcnemar_test(yte, geo_preds["SVM"], geo_preds["LogReg"])
                print(f"    McNemar SVM vs LogReg: chi2={t['chi2']:.2f} p={t['p']:.4g} "
                      f"({'significant' if t['p'] < 0.05 else 'not significant'})")

        # figure for the best geo-holdout model: predict full east water region
        if best_model:
            _figure(st, lab, present, west, best_model, args, scene)

    # write outputs
    s = Path(args.out) / "e8_urm_summary.csv"
    with open(s, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    c = Path(args.out) / "e8_urm_confusion.csv"
    with open(c, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conf[0].keys())); w.writeheader(); w.writerows(conf)
    print(f"\nSaved {s}\nSaved {c}")
    if mcn:
        mc = Path(args.out) / "e8_urm_mcnemar.csv"
        with open(mc, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mcn[0].keys())); w.writeheader(); w.writerows(mcn)
        print(f"Saved {mc}")


def _perf1_from_cm(cm, present):
    f1 = {}
    for i, c in enumerate(present):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[fk.CLASSES[c]] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return f1


def _srow(scene, protocol, model, oa, ka, mf, f1, present, energy):
    row = dict(scene=scene, protocol=protocol, model=model,
               OA_mean=round(oa[0], 4), OA_lo=round(oa[1], 4), OA_hi=round(oa[2], 4),
               kappa_mean=round(ka[0], 4), macroF1_mean=round(mf[0], 4))
    for cname in ["Sand/Bare", "Seagrass", "Coral/Hard-bottom", "Deep Water"]:
        row["F1_" + cname.split("/")[0].replace(" ", "")] = round(f1.get(cname, float("nan")), 4) if cname in f1 else ""
    row["train_energy_uWh"] = round(energy, 2) if energy == energy else ""
    return row


def _figure(st, lab, present, west, model, args, scene):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        lm_all = ul.label_masks(lab, st, present)
        trm = (st["water"] & (lab > 0)) & west
        Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=args.nsamp, seed=args.seed, restrict_mask=trm)
        east_water = st["water"] & ~west
        rr, cc = np.where(east_water)
        X = st["stack"][rr, cc]; ok = np.all(np.isfinite(X), axis=1)
        rr, cc, X = rr[ok], cc[ok], X[ok]
        H, W = st["geo"]["shape"]
        pmap = np.zeros((H, W), np.uint8); umap = np.zeros((H, W), np.uint8)
        umap[east_water & (lab > 0)] = lab[east_water & (lab > 0)]
        if model == "MLP":
            mdl = fk.TorchMLP(seed=args.seed, epochs=100).fit(Xtr, ytr); pr = mdl.predict(X)
        elif model == "2D_CNN":
            mdl = fk.Torch2DCNN(st["stack"], seed=args.seed, patch=args.patch, epochs=args.epochs).fit(Rtr, Ctr, ytr)
            pr = mdl.predict(rr, cc)
        else:
            base = fk.make_model(model, args.seed)
            Xf, yf = (_cap_samples(Xtr, ytr, args.cap_nsamp) if model in args.cap_models else (Xtr, ytr))
            labs = sorted(np.unique(yf)); lmm = {l: i for i, l in enumerate(labs)}; imm = {i: l for l, i in lmm.items()}
            base.fit(Xf, np.array([lmm[v] for v in yf]))
            pr = np.empty(len(X), dtype=int)
            for s in range(0, len(X), 500000):
                e = min(s + 500000, len(X)); pr[s:e] = [imm[int(p)] for p in base.predict(X[s:e])]
        pmap[rr, cc] = pr
        cols = ListedColormap(["#00000000", "#1f3b73", "#e8d29a", "#2e8b57", "#c0553b"])
        agree = np.zeros((H, W)); mask = east_water & (lab > 0)
        agree[mask & (umap == pmap)] = 1; agree[mask & (umap != pmap)] = 2
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(pmap, cmap=cols, vmin=0, vmax=4); ax[0].set_title(f"Predicted ({model}), east holdout")
        ax[1].imshow(umap, cmap=cols, vmin=0, vmax=4); ax[1].set_title("URM labels (east)")
        ax[2].imshow(agree, cmap=ListedColormap(["#00000000", "#2e8b57", "#c0553b"]), vmin=0, vmax=2)
        ax[2].set_title("Agreement / disagreement")
        for a in ax:
            a.axis("off")
        p = Path(args.figdir) / f"e8_urm_{scene}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        print(f"    Saved {p}")
    except Exception as ex:
        print(f"    (figure skipped: {ex})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "data"))
    ap.add_argument("--out", default=str(BASE / "outputs"))
    ap.add_argument("--figdir", default=str(BASE / "figures"))
    ap.add_argument("--urm", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--service_url", default=None)
    ap.add_argument("--save_urm", default=None)
    ap.add_argument("--field", default=None)
    ap.add_argument("--include_deep", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--nsamp", type=int, default=500)
    ap.add_argument("--test_cap", type=int, default=100000)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap_models", nargs="*", default=["SVM", "kNN"],
                    help="tabular models trained on a capped sample (they scale poorly)")
    ap.add_argument("--cap_nsamp", type=int, default=2000,
                    help="per-class training cap for --cap_models")
    ap.add_argument("--tab_models", nargs="+", default=fk.TABULAR_MODELS)
    ap.add_argument("--deep_models", nargs="+", default=["MLP", "2D_CNN"])
    args = ap.parse_args()
    run(args)
