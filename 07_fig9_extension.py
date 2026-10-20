"""
07_fig9_extension.py -- Fig. 9 learning-curve extension for the two deep models.

Adds larger training-set sizes (default 20,000 and 50,000 per class) for the
two deep models (MLP, 2D_CNN) under the IDENTICAL protocol as the published
learning curve: it calls buffered_benchmark() from 05_ablations.py unchanged,
so the 300 m buffer, 5-fold spatial CV + geo holdout, seed 42, patch=9,
epochs=60 (CNN) / 100 (MLP), and the 100k test cap are all inherited rather
than re-implemented. No hyperparameters are re-tuned at the new sizes.

Honesty guard: fknms_core.sample_labeled silently draws min(n, available)
without replacement. At 20k/50k a class pool could in principle fall short,
which would make the x-axis lie. This script wraps the sampler to LOG every
draw's requested size and actual per-class counts (learning_curve_draws.csv),
and prints a warning on any shortfall.

Outputs (in --out, default ./outputs):
    ablation_learning_curve_ext.csv   new rows only (nsamp in {sizes})
    learning_curve_draws.csv          per-draw requested vs actual class counts
    ablation_learning_curve.csv       baseline rows + new rows, merged, so that
                                      06_figures.py regenerates Fig. 9 with all
                                      six sizes with no changes to 06_figures.
                                      (Any pre-existing file at that path is
                                      first backed up with suffix _base_backup.)

Run (long job; 20k first gives usable partial results sooner):
    python 07_fig9_extension.py --data <scene_dir> --sizes 20000
    python 07_fig9_extension.py --data <scene_dir> --sizes 50000
or both in one go:
    python 07_fig9_extension.py --data <scene_dir>

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fknms_core as fk  # noqa: E402


def _load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


abl = _load_mod("ablations05", "05_ablations.py")

DRAW_LOG = []
_orig_sample = fk.sample_labeled


def logging_sample_labeled(stack, label_masks, n=400, seed=42, restrict_mask=None):
    X, y, R, C = _orig_sample(stack, label_masks, n=n, seed=seed,
                              restrict_mask=restrict_mask)
    y_arr = np.asarray(y)
    counts = {fk.CLASSES.get(c, str(c)): int((y_arr == c).sum())
              for c in np.unique(y_arr)}
    short = {k: v for k, v in counts.items() if v < n}
    DRAW_LOG.append(dict(call=len(DRAW_LOG) + 1, requested_per_class=n,
                         **{f"drawn_{k.split('/')[0].replace(' ', '')}": v
                            for k, v in counts.items()}))
    if short:
        print(f"    !! draw shortfall at n={n}: {short} "
              f"(sampled without replacement; pool exhausted)", flush=True)
    return X, y, R, C


def main():
    ap = argparse.ArgumentParser(description="Fig. 9 extension: larger sizes for the deep models.")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=str(HERE / "data" / "urm_keys.geojson"))
    ap.add_argument("--field", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[20000, 50000])
    ap.add_argument("--models", nargs="+", default=["MLP", "2D_CNN"])
    ap.add_argument("--baseline_csv", default=None,
                    help="published learning-curve CSV to merge with; default: "
                         "<out>/ablation_learning_curve.csv if present, else "
                         "reference_outputs/ablation_learning_curve.csv")
    ap.add_argument("--buffer_px", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_side", type=int, default=5)
    ap.add_argument("--test_cap", type=int, default=100000)
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # protocol parity statement, printed into the log for the response letter
    print(f"Fig.9 extension | sizes={a.sizes} models={a.models} | buffer={a.buffer_px*10} m "
          f"k={a.k} n_side={a.n_side} seed={a.seed} patch={a.patch} epochs={a.epochs} "
          f"test_cap={a.test_cap} | protocol inherited from 05_ablations.buffered_benchmark",
          flush=True)

    gdf = abl.load_urm(a.urm, fk.prepare_scene(fk.find_scenes(a.data)[0])["geo"]["crs"])
    print(f"URM polygons: {len(gdf):,}", flush=True)

    fk.sample_labeled = logging_sample_labeled
    try:
        tab = [m for m in a.models if m in fk.TABULAR_MODELS]
        deep = [m for m in a.models if m not in fk.TABULAR_MODELS]
        rows = []
        for ns in a.sizes:
            print(f"\n########## learning-curve extension: nsamp={ns}/class ##########",
                  flush=True)
            rows += abl.buffered_benchmark(a, gdf, {"nsamp": ns}, tab, deep, ns)
    finally:
        fk.sample_labeled = _orig_sample

    ext_path = out / "ablation_learning_curve_ext.csv"
    abl.write_csv(rows, ext_path)
    if DRAW_LOG:
        keys = sorted({k for r in DRAW_LOG for k in r}, key=lambda s: (s != "call", s))
        with open(out / "learning_curve_draws.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(DRAW_LOG)
        print(f"Saved {out / 'learning_curve_draws.csv'}")

    # merge with the published baseline so 06_figures.py replots Fig. 9 as-is
    base = Path(a.baseline_csv) if a.baseline_csv else (
        out / "ablation_learning_curve.csv"
        if (out / "ablation_learning_curve.csv").exists()
        else HERE / "reference_outputs" / "ablation_learning_curve.csv")
    if not base.exists():
        print(f"Baseline CSV not found at {base}; merged file not written.")
        return
    merged_rows, fieldnames = [], []
    ext_rows = []
    with open(ext_path, newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames)
        ext_rows = list(rd)
    ext_keys = {(r.get("scene"), r.get("protocol"), r.get("model"), r.get("nsamp"))
                for r in ext_rows}
    with open(base, newline="") as f:
        rd = csv.DictReader(f)
        for fn in rd.fieldnames:
            if fn not in fieldnames:
                fieldnames.append(fn)
        for r in rd:
            # idempotence: a re-run of a size replaces, never duplicates
            if (r.get("scene"), r.get("protocol"), r.get("model"), r.get("nsamp")) in ext_keys:
                continue
            merged_rows.append(r)
    merged_rows += ext_rows
    target = out / "ablation_learning_curve.csv"
    if target.exists() and target.resolve() != base.resolve():
        shutil.copy2(target, out / "ablation_learning_curve_base_backup.csv")
    elif target.exists():
        shutil.copy2(target, out / "ablation_learning_curve_base_backup.csv")
    with open(target, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(merged_rows)
    print(f"Saved {ext_path}\nSaved merged {target} "
          f"({len(merged_rows)} rows; run 06_figures.py to regenerate Fig. 9)")


if __name__ == "__main__":
    main()
