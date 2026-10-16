"""
04_energy.py -- per-stage energy on a single machine, for every model, with the
models fitted on the same task as the accuracy benchmark.

Training uses the URM-supervised geographic-holdout draw rebuilt by urm_draw.py
(western half of each scene, --nsamp samples per class, seed 42, the same
sampling functions as 03_benchmark.py), so the energy and accuracy results come
from identically fitted models. Each scene's draw is fingerprinted with a
SHA-256 digest of the sorted (row, column, label) triplets and written to
outputs/train_draw_digest.csv for verification against the benchmark.

Measures CodeCarbon energy (an estimate; on Windows there is no RAPL, so it is
derived from the processor thermal-design-power model and load, not a
wall-socket reading) for each stage of the pipeline:

  model-independent, once per scene:
      preprocess       band loading, indices, water mask
      dii_watercol     Lyzenga depth-invariant-index water-column correction
      feature_extract  feature stack + the URM-supervised labeled draw

  per model:
      train            model fit on the URM draw
      fullmap_infer    classify every water pixel of the scene (~11 million)

All nine classifiers are covered on both axes. Each stage is measured --repeats
times (default 5). Full-scene inference for the support vector machine and
k-nearest-neighbours is measured with --sv_repeats passes (default 1); a single
pass aggregates roughly eleven million predictions per scene. The SVM
support-vector count is recorded to outputs/svm_support.csv, since
support-vector inference cost scales with it.

Design for an unattended run:
  - RESUMABLE: any (scene, stage, model) already in the output CSV is skipped,
    so a stop or crash never loses completed work; run the same command again.
  - INCREMENTAL: each result is written the moment it is measured.
  - Non-physical (<=0 uWh) CodeCarbon readings are dropped before averaging.
  - Memory is released between scenes.

The 2D CNN uses patch=9 and 60 epochs, matching the architecture reported in
the manuscript; pass --patch / --cnn_epochs if your accuracy runs differed.

Run on the SAME machine as the main benchmark, naming it explicitly:
    python 04_energy.py --label i7_16GB
Probe on one scene first:
    python 04_energy.py --label i7_16GB --limit_scenes 1 --repeats 1

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fknms_core as fk                 # noqa: E402
import urm_draw as ud                   # noqa: E402

FIELDS = ["hardware", "scene", "stage", "model", "energy_uWh_mean", "energy_uWh_sd",
          "emissions_kg_mean", "seconds_mean", "repeats", "n_valid"]
ALL_MODELS = ["RF", "ExtraTrees", "XGBoost", "LightGBM", "SVM", "LogReg", "kNN", "MLP", "2D_CNN"]
SV_MODELS = {"SVM", "kNN"}


def repeat_energy(fn, repeats):
    e_uWh, emis, secs = [], [], []
    for _ in range(repeats):
        with fk.Energy("stage") as e:
            fn()
        e_uWh.append(e.energy_uWh); emis.append(e.emissions_kg); secs.append(e.seconds)
    e = np.array(e_uWh, dtype=float)
    valid = e[np.isfinite(e) & (e > 0)]
    mean = float(valid.mean()) if valid.size else float("nan")
    sd = float(valid.std(ddof=1)) if valid.size > 1 else 0.0
    return mean, sd, float(np.nanmean(emis)), float(np.mean(secs)), int(valid.size)


def load_done(path):
    done = set()
    if path.exists():
        for r in csv.DictReader(open(path)):
            done.add((r["scene"], r["stage"], r["model"]))
    return done


def load_scenes_col(path):
    seen = set()
    if path.exists():
        for r in csv.DictReader(open(path)):
            seen.add(r["scene"])
    return seen


def append_dict(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Per-stage energy with URM-supervised model fits.")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=ud.default_urm())
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--service_url", default=None)
    ap.add_argument("--save_urm", default=None)
    ap.add_argument("--field", default=None, help="URM class field (auto-detected if omitted)")
    ap.add_argument("--label", default=fk.cpu_label(), help="hardware tag, e.g. i7_16GB")
    ap.add_argument("--nsamp", type=int, default=5000, help="training samples per class (match the benchmark)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (match the benchmark)")
    ap.add_argument("--repeats", type=int, default=5, help="measurements per stage (manuscript: 5)")
    ap.add_argument("--sv_repeats", type=int, default=1,
                    help="full-scene inference passes for SVM and kNN (manuscript: 1)")
    ap.add_argument("--chunk", type=int, default=500000, help="pixels per tabular prediction chunk")
    ap.add_argument("--cnn_chunk", type=int, default=20000, help="pixels per CNN inference chunk")
    ap.add_argument("--patch", type=int, default=9, help="CNN patch window (manuscript: 9)")
    ap.add_argument("--cnn_epochs", type=int, default=60, help="CNN training epochs (manuscript: 60)")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--limit_scenes", type=int, default=0, help="0 = all scenes; N = first N only")
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    outp = Path(a.out) / "energy_per_stage.csv"
    digp = Path(a.out) / "train_draw_digest.csv"
    supp = Path(a.out) / "svm_support.csv"
    done = load_done(outp)
    digested = load_scenes_col(digp)
    supported = load_scenes_col(supp)
    gdf = ud.load_urm_gdf(a.data, a.urm, a.download, a.service_url, a.save_urm)
    scenes = fk.find_scenes(a.data)
    if a.limit_scenes:
        scenes = scenes[:a.limit_scenes]
    print(f"URM polygons: {len(gdf)}   nsamp={a.nsamp}/class  seed={a.seed}  "
          f"repeats={a.repeats}  sv_repeats={a.sv_repeats}  hardware={a.label}", flush=True)

    def measure(scene, stage, model, fn, repeats):
        if (scene, stage, model) in done:
            return
        res = repeat_energy(fn, repeats)
        append_dict(outp, dict(hardware=a.label, scene=scene, stage=stage, model=model,
                               energy_uWh_mean=round(res[0], 2), energy_uWh_sd=round(res[1], 2),
                               emissions_kg_mean=f"{res[2]:.3e}", seconds_mean=round(res[3], 3),
                               repeats=repeats, n_valid=res[4]))
        print(f"    {stage:15s} {model:9s} {res[0]:11.0f} uWh  {res[3]:8.2f}s  n={res[4]}  -> saved", flush=True)

    for sd in scenes:
        meta = fk.read_metadata(sd)
        scene = meta["label"]
        print(f"\n=== {scene} ({meta['date']}) on {a.label} ===", flush=True)

        measure(scene, "preprocess", "-", lambda: (
            lambda b, s, g: fk.water_mask(b, s, fk.spectral_indices(b)[0]))(*fk.load_bands(sd)),
            a.repeats)

        st = fk.prepare_scene(sd)
        tr = ud.build_urm_training(st, gdf, nsamp=a.nsamp, seed=a.seed, field=a.field)
        if tr is None:
            print("  fewer than two usable URM classes; scene skipped.", flush=True)
            del st
            gc.collect()
            continue
        print(f"  URM draw: {tr['counts']}   sha={tr['draw_sha']}", flush=True)
        if scene not in digested:
            append_dict(digp, dict(scene=scene, n_train=len(tr["ytr"]),
                                   **{f"n_{k}": v for k, v in tr["counts"].items()},
                                   draw_sha=tr["draw_sha"]))

        measure(scene, "dii_watercol", "-", lambda: fk.apply_features(
            st["bands"], st["water"], fk.fit_water_column(st["bands"], st["water"])), a.repeats)
        measure(scene, "feature_extract", "-", lambda: fk.sample_labeled(
            fk.build_stack(st["feats"]), tr["lm_all"], n=a.nsamp, seed=a.seed,
            restrict_mask=tr["tr_pool"]), a.repeats)

        X, y, R, C = tr["Xtr"], tr["ytr"], tr["Rtr"], tr["Ctr"]
        yc = tr["yc"]
        need_infer = any((scene, "fullmap_infer", m) not in done for m in a.models)
        rr = cc = Xall = None
        if need_infer:
            rr, cc, Xall = fk.water_pixels(st)
            print(f"  train sample {len(y):,}   full-scene pixels {len(rr):,}", flush=True)

        for m in a.models:
            if m == "MLP":
                measure(scene, "train", m, lambda: fk.TorchMLP(seed=a.seed).fit(X, y), a.repeats)
            elif m == "2D_CNN":
                measure(scene, "train", m, lambda: fk.Torch2DCNN(
                    st["stack"], seed=a.seed, patch=a.patch, epochs=a.cnn_epochs).fit(R, C, y),
                    a.repeats)
            else:
                measure(scene, "train", m, lambda m=m: fk.make_model(m, a.seed).fit(X, yc), a.repeats)

            if (scene, "fullmap_infer", m) in done:
                continue
            reps = a.sv_repeats if m in SV_MODELS else a.repeats
            if m == "MLP":
                mdl = fk.TorchMLP(seed=a.seed).fit(X, y)
                measure(scene, "fullmap_infer", m, lambda: mdl.predict(Xall), reps)
                del mdl
            elif m == "2D_CNN":
                cnn = fk.Torch2DCNN(st["stack"], seed=a.seed, patch=a.patch,
                                    epochs=a.cnn_epochs).fit(R, C, y)
                measure(scene, "fullmap_infer", m, lambda: cnn.predict(rr, cc, chunk=a.cnn_chunk), reps)
                del cnn
            else:
                base = fk.make_model(m, a.seed)
                base.fit(X, yc)
                if m == "SVM" and scene not in supported:
                    svc = base
                    if hasattr(base, "named_steps"):
                        svc = list(base.named_steps.values())[-1]
                    if hasattr(svc, "n_support_"):
                        append_dict(supp, dict(scene=scene, n_train=len(yc),
                                               n_support=int(sum(svc.n_support_))))
                        print(f"    [SVM] support vectors: {int(sum(svc.n_support_)):,}", flush=True)
                measure(scene, "fullmap_infer", m,
                        lambda base=base: [base.predict(Xall[s:s + a.chunk])
                                           for s in range(0, len(Xall), a.chunk)], reps)
                del base
            gc.collect()

        del st, tr, X
        if Xall is not None:
            del Xall, rr, cc
        gc.collect()

    print(f"\nDone. Output: {outp}", flush=True)


if __name__ == "__main__":
    main()
