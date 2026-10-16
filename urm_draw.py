"""
urm_draw.py -- the URM-supervised training draw shared by 03c and 04.

Rebuilds, with the same functions and seed as 03_benchmark.py, the training
sample used under the geographic holdout: the Unified Reef Map label raster,
the classes with at least --min_px labeled pixels, the labeled pool restricted
to the western half of the scene, and a balanced draw of --nsamp pixels per
class (seed 42 by default). Because the construction is identical, the pixels
are identical to the ones behind the reported accuracies; each draw is
fingerprinted with a SHA-256 digest of the sorted (row, column, label) triplets
so this can be verified against reference_outputs/train_draw_digest.csv.

Author: Harsh Deep Singh Narula
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fknms_core as fk                 # noqa: E402
import urm_labels as ul                 # noqa: E402
from eval_engine import BENTHIC         # noqa: E402


def default_urm():
    p = HERE / "data" / "urm_keys.geojson"
    return str(p) if p.exists() else None


def load_urm_gdf(data, urm, download=False, service_url=None, save_urm=None):
    """Fetch the Unified Reef Map once, using the first scene's bounds and CRS."""
    import geopandas as gpd
    scenes = fk.find_scenes(data)
    if not scenes:
        raise SystemExit(f"No scenes found under {data}; run 01_download_data.py first.")
    if not download and (not urm or not Path(urm).exists()):
        raise SystemExit(
            "URM reference not found. It ships with this package as data/urm_keys.geojson; "
            "pass --urm <file>, or --download with --service_url to query the live layer.")
    st0 = fk.prepare_scene(scenes[0])
    H0, W0 = st0["geo"]["shape"]
    tf0 = st0["geo"]["transform"]
    crs0 = st0["geo"]["crs"]
    left, top = tf0 * (0, 0)
    right, bottom = tf0 * (W0, H0)
    g = gpd.GeoSeries.from_wkt(
        [f"POINT({left} {top})", f"POINT({right} {bottom})"], crs=crs0).to_crs(4326)
    bbox = (min(g.x), min(g.y), max(g.x), max(g.y))
    return ul.fetch_urm(bbox, crs0, urm=urm, download=download,
                        service_url=service_url, save_urm=save_urm)


def build_urm_training(st, gdf, nsamp=5000, seed=42, field=None, min_px=200):
    """The geographic-holdout training draw for one prepared scene, or None when
    fewer than two benthic classes have min_px labeled pixels (the same skip
    rule as the benchmark)."""
    lab, _fld = ul.build_label_raster(st, gdf, field=field, include_deep=False)
    codes = sorted(BENTHIC)
    present = [c for c in codes if (lab == c).sum() >= min_px]
    if len(present) < 2:
        return None
    lm_all = ul.label_masks(lab, st, present)
    labeled = st["water"] & (lab > 0)
    west = fk.region_split(st["geo"]["shape"], axis=1, frac=0.5)
    tr_pool = labeled & west
    Xtr, ytr, Rtr, Ctr = fk.sample_labeled(st["stack"], lm_all, n=nsamp,
                                           seed=seed, restrict_mask=tr_pool)
    labels = sorted(np.unique(ytr))
    lm = {l: i for i, l in enumerate(labels)}
    yc = np.array([lm[v] for v in ytr])
    order = np.lexsort((Ctr, Rtr))
    trip = np.stack([np.asarray(Rtr)[order], np.asarray(Ctr)[order],
                     np.asarray(ytr)[order]]).astype(np.int64)
    sha = hashlib.sha256(trip.tobytes()).hexdigest()[:12]
    return dict(lab=lab, present=present, lm_all=lm_all, tr_pool=tr_pool,
                Xtr=Xtr, ytr=ytr, yc=yc, Rtr=Rtr, Ctr=Ctr, draw_sha=sha,
                counts={fk.CLASSES[c]: int((ytr == c).sum()) for c in present})
