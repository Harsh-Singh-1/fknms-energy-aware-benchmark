"""
08_holdout_majority_rate.py -- majority-class rate on the actual test pixels
of each protocol.

Rebuilds, per scene, the EXACT geographic-holdout test set used by
03_benchmark.py -- URM-labeled water pixels in the eastern half, buffered 300 m
from the western training pool, capped at --test_cap with seed --seed via
eval_engine.gather_test -- and reports the class shares of those pixels. The
pooled seagrass share across scenes is the no-information baseline for
Section V-C and the dashed line in Figure 3a.

Outputs (in --out):
    holdout_test_class_shares.csv   per scene: n_test, share per class,
                                    majority class and its rate
    plus a printed pooled summary (pixel-weighted and scene-averaged).

Run (fast; no model training, ~a minute per scene for I/O and masking):
    python 08_holdout_majority_rate.py --data <scene_dir>

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fknms_core as fk                       # noqa: E402
import urm_labels as ul                       # noqa: E402
import spatial_buffer as sb                   # noqa: E402
from eval_engine import gather_test, pixel_blocks, BENTHIC  # noqa: E402
from urm_draw import load_urm_gdf             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Class shares of the buffered holdout test pixels.")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--urm", default=str(HERE / "data" / "urm_keys.geojson"))
    ap.add_argument("--field", default=None)
    ap.add_argument("--buffer_px", type=int, default=30)
    ap.add_argument("--test_cap", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", choices=["holdout", "cv", "both"], default="both",
                    help="which protocol's test pixels to profile")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_side", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    gdf = load_urm_gdf(a.data, a.urm)
    codes = sorted(BENTHIC)
    rows, pooled = [], {c: 0 for c in codes}
    per_scene_major = []
    protocols = ["holdout", "cv"] if a.protocol == "both" else [a.protocol]

    for sd in fk.find_scenes(a.data):
        meta = fk.read_metadata(sd); scene = meta["label"]
        st = fk.prepare_scene(sd)
        lab, _ = ul.build_label_raster(st, gdf, field=a.field, include_deep=False)
        present = [c for c in codes if (lab == c).sum() >= 200]
        if len(present) < 2:
            print(f"=== {scene}: <2 usable classes; skipped ==="); continue
        labeled = st["water"] & (lab > 0)
        if "cv" in protocols:
            # exact replication of the benchmark's fold construction (seed 42)
            pb = pixel_blocks(st["geo"]["shape"], a.n_side)
            rng = np.random.default_rng(a.seed)
            blocks = np.unique(pb[labeled]); rng.shuffle(blocks)
            fold_of = {b: i % a.k for i, b in enumerate(blocks)}
            ycv = []
            for f in range(a.k):
                trb = [b for b in blocks if fold_of[b] != f]
                teb = [b for b in blocks if fold_of[b] == f]
                trm = labeled & np.isin(pb, trb)
                tem = sb.buffered_test_mask(labeled & np.isin(pb, teb), trm, a.buffer_px)
                if trm.sum() < 500 or tem.sum() < 200:
                    continue
                _Xf, yf, _rf, _cf = gather_test(st["stack"], lab, tem, a.test_cap,
                                                a.seed + f)
                ycv.append(yf)
            ycv = np.concatenate(ycv)
            sh = {c: float((ycv == c).sum()) / len(ycv) for c in codes}
            mj = max(sh, key=sh.get)
            rowc = dict(protocol="spatial_cv", scene=scene, n_test=len(ycv),
                        majority_class=fk.CLASSES[mj], majority_rate=round(sh[mj], 4))
            for c in codes:
                rowc[f"share_{fk.CLASSES[c].split('/')[0].replace(' ', '')}"] = round(sh[c], 4)
            rows.append(rowc)
            print(f"=== {scene} [cv]: n={len(ycv):,}  majority={fk.CLASSES[mj]} {sh[mj]:.4f}  "
                  + "  ".join(f"{fk.CLASSES[c].split('/')[0]}={sh[c]:.3f}" for c in codes),
                  flush=True)
        if "holdout" not in protocols:
            continue
        west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
        tr_pool = labeled & west
        te_buf = sb.buffered_test_mask(labeled & ~west, tr_pool, a.buffer_px)
        _X, yte, _r, _c = gather_test(st["stack"], lab, te_buf, a.test_cap, a.seed)
        n = len(yte)
        shares = {c: float((yte == c).sum()) / n for c in codes}
        maj = max(shares, key=shares.get)
        for c in codes:
            pooled[c] += int((yte == c).sum())
        per_scene_major.append(shares[maj])
        row = dict(protocol="geo_holdout", scene=scene, n_test=n,
                   majority_class=fk.CLASSES[maj],
                   majority_rate=round(shares[maj], 4))
        for c in codes:
            row[f"share_{fk.CLASSES[c].split('/')[0].replace(' ', '')}"] = round(shares[c], 4)
        rows.append(row)
        print(f"=== {scene}: n={n:,}  majority={fk.CLASSES[maj]} {shares[maj]:.4f}  "
              + "  ".join(f"{fk.CLASSES[c].split('/')[0]}={shares[c]:.3f}" for c in codes),
              flush=True)

    if pooled and sum(pooled.values()):
        tot = sum(pooled.values())
        pooled_shares = {c: pooled[c] / tot for c in codes}
        pmaj = max(pooled_shares, key=pooled_shares.get)
        print(f"\nPOOLED over {tot:,} holdout test pixels: "
              f"majority = {fk.CLASSES[pmaj]} at {pooled_shares[pmaj]:.4f}  "
              f"(scene-mean majority rate {np.mean(per_scene_major):.4f})")
    cvrows = [r for r in rows if r["protocol"] == "spatial_cv"]
    if cvrows:
        ncv = sum(r["n_test"] for r in cvrows)
        wmean = sum(r["majority_rate"] * r["n_test"] for r in cvrows) / ncv
        print(f"POOLED over {ncv:,} spatial-CV test pixels: "
              f"pixel-weighted majority rate {wmean:.4f} "
              f"(scene range {min(r['majority_rate'] for r in cvrows):.4f}"
              f"-{max(r['majority_rate'] for r in cvrows):.4f})")
    print("The pooled (or scene-mean) rate is the no-information baseline used in "
          "Section V-C and Figure 3a of the article.")

    p = out / "test_class_shares.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
