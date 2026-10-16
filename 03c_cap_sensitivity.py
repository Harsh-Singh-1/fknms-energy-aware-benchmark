"""
03c_cap_sensitivity.py -- sensitivity of the reported accuracies to the
100,000-pixel test-set cap.

Under the geographic holdout, models are evaluated on the URM-labeled pixels in
the buffered held-out region; when a test set exceeds 100,000 pixels it is
evaluated on a random sample of 100,000 (gather_test in eval_engine.py). This
script measures, per scene and model, overall accuracy on ALL held-out
URM-labeled pixels versus the capped sample used in the main run, and reports
the difference. The training draw, buffer, and models are identical to
03_benchmark.py, so the only thing that varies is the test-set size.

Output:
    outputs/test_cap_sensitivity.csv   per scene/model: n_all, n_cap, OA_all,
                                       OA_cap, dOA (= OA_cap - OA_all)
    the maximum |dOA| across all scenes and models is printed at the end.

Run after 01_download_data.py:
    python 03c_cap_sensitivity.py

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fknms_core as fk                 # noqa: E402
import spatial_buffer as sb             # noqa: E402
import urm_draw as ud                   # noqa: E402
from eval_engine import eval_all, gather_test   # noqa: E402

BIG = 10 ** 9   # effectively uncapped


def main():
    ap = argparse.ArgumentParser(description="Test-set cap sensitivity (geographic holdout).")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=ud.default_urm())
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--service_url", default=None)
    ap.add_argument("--save_urm", default=None)
    ap.add_argument("--field", default=None)
    ap.add_argument("--nsamp", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer_px", type=int, default=30, help="train-test buffer in pixels (300 m)")
    ap.add_argument("--cap", type=int, default=100000, help="the cap under test (main run: 100000)")
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tab_models", nargs="+", default=fk.TABULAR_MODELS)
    ap.add_argument("--deep_models", nargs="+", default=["MLP", "2D_CNN"])
    a = ap.parse_args()

    if not fk.HAS_TORCH:
        a.deep_models = []
        print("PyTorch not found; tabular models only.")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    gdf = ud.load_urm_gdf(a.data, a.urm, a.download, a.service_url, a.save_urm)
    rows = []

    for sd in fk.find_scenes(a.data):
        meta = fk.read_metadata(sd)
        scene = meta["label"]
        st = fk.prepare_scene(sd)
        tr = ud.build_urm_training(st, gdf, nsamp=a.nsamp, seed=a.seed, field=a.field)
        if tr is None:
            print(f"[skip] {scene}: fewer than two usable URM classes")
            continue
        lab, tr_pool = tr["lab"], tr["tr_pool"]
        Xtr, ytr, Rtr, Ctr = tr["Xtr"], tr["ytr"], tr["Rtr"], tr["Ctr"]

        labeled = st["water"] & (lab > 0)
        west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
        te_all = labeled & ~west
        te_buf = sb.buffered_test_mask(te_all, tr_pool, a.buffer_px)

        Xa, ya, Ra, Ca = gather_test(st["stack"], lab, te_buf, BIG, a.seed)
        Xc, yc_t, Rc, Cc = gather_test(st["stack"], lab, te_buf, a.cap, a.seed)
        if len(np.unique(ytr)) < 2 or len(np.unique(ya)) < 2:
            print(f"[skip] {scene}: fewer than two classes in train or test")
            continue
        print(f"\n=== {scene} ===  held-out URM pixels: all={len(ya):,}  capped={len(yc_t):,}")

        res_all = eval_all(a.tab_models, a.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xa, ya, Ra, Ca,
                           tr["present"], a.patch, a.epochs, a.seed)
        res_cap = eval_all(a.tab_models, a.deep_models, st["stack"],
                           Xtr, ytr, Rtr, Ctr, Xc, yc_t, Rc, Cc,
                           tr["present"], a.patch, a.epochs, a.seed)
        for m in res_all:
            oa_all = res_all[m][0]["OA"]
            oa_cap = res_cap[m][0]["OA"]
            rows.append(dict(scene=scene, model=m, n_all=len(ya), n_cap=len(yc_t),
                             OA_all=round(oa_all, 4), OA_cap=round(oa_cap, 4),
                             dOA=round(oa_cap - oa_all, 4)))
            print(f"  {m:11s} OA_all={oa_all:.4f}  OA_cap={oa_cap:.4f}  dOA={oa_cap - oa_all:+.4f}")
        del st, tr

    outp = Path(a.out) / "test_cap_sensitivity.csv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mx = max(abs(r["dOA"]) for r in rows)
    print(f"\nSaved {outp}")
    print(f"Maximum |dOA| across all scenes and models: {mx:.4f}")


if __name__ == "__main__":
    main()
