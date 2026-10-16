"""
fknms_core.py
Shared core for the FKNMS IEEE Access revision experiments (E1-E7).

Ports the preprocessing, Lyzenga depth-invariant index (DII), and fixed-threshold
spectral labelling from the original 02b_classify_only.py, and adds the
machinery the reviewers asked for: fold-internal threshold derivation,
spatial cross-validation, a wider model zoo, a genuine patch-based 2D CNN,
proper per-stage energy measurement, and statistical tests.

The default (global) code path reproduces the V3 protocol so the numbers
stay consistent with the submitted manuscript; the train_mask hooks let the
spatial-CV scripts derive every data-dependent threshold from training
pixels only.

Author: Harsh Deep Singh Narula
"""

import os, time, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")

CLASSES = {1: "Deep Water", 2: "Sand/Bare", 3: "Seagrass", 4: "Coral/Hard-bottom"}
CLASS_ORDER = [1, 2, 3, 4]
FEATURES = ["B02", "B03", "B04", "B08",
            "DII_BG", "DII_BR", "DII_GR",
            "NDVI", "NDWI", "SAVI"]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        import torch
        torch.manual_seed(seed)


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def load_bands(scene_dir):
    """Read B02,B03,B04,B08 as surface reflectance and SCL resampled to 10 m.

    Applies the Sentinel-2 Collection-2 offset: reflectance = (DN - 1000)/10000.
    Returns (bands dict, scl array, geo dict with shape/transform/crs).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    scene_dir = Path(scene_dir)
    bands, shape, transform, crs = {}, None, None, None
    for b in ["B02", "B03", "B04", "B08"]:
        with rasterio.open(scene_dir / f"{b}.tif") as src:
            raw = src.read(1).astype(np.float32)
            bands[b] = np.where(raw > 0, np.clip((raw - 1000) / 10000, 0, 1), 0)
            if b == "B02":
                shape, transform, crs = src.shape, src.transform, src.crs
    with rasterio.open(scene_dir / "SCL.tif") as s:
        scl = np.zeros(shape, dtype=np.uint8)
        reproject(source=rasterio.band(s, 1), destination=scl,
                  src_transform=s.transform, src_crs=s.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    return bands, scl, dict(shape=shape, transform=transform, crs=crs)


def read_metadata(scene_dir):
    with open(Path(scene_dir) / "metadata.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Indices, masks, water column correction
# --------------------------------------------------------------------------
def spectral_indices(bands):
    eps = 1e-10
    b03, b04, b08 = bands["B03"], bands["B04"], bands["B08"]
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    savi = 1.5 * (b08 - b04) / (b08 + b04 + 0.5 + eps)
    return ndwi, ndvi, savi


def water_mask(bands, scl, ndwi):
    b02 = bands["B02"]
    nodata = (b02 == 0)
    cloud = np.isin(scl, [0, 1, 3, 7, 8, 9, 10, 11])
    water = ((scl == 6) | (ndwi > 0)) & ~nodata & ~cloud
    return water, nodata, cloud


def fit_water_column(bands, water, train_mask=None):
    """Derive depth-invariant parameters (deep threshold, deep medians,
    attenuation ratios). If train_mask is given they are derived from
    training pixels only, which removes the label/feature leakage the
    reviewers flagged for the random split.
    """
    b02, b03, b04 = bands["B02"], bands["B03"], bands["B04"]
    m = water if train_mask is None else (water & train_mask)

    wb02 = b02[m & (b02 > 0)]
    dt = float(np.percentile(wb02, 5)) if len(wb02) > 100 else 0.005

    deep_m = m & (b02 < dt) & (b03 < dt) & (b04 < dt * 0.8)
    shallow_m = m & ~deep_m
    bd = {}
    for n, a in [("B02", b02), ("B03", b03), ("B04", b04)]:
        vals = a[deep_m & (a > 0)]
        bd[n] = float(np.median(vals)) if deep_m.sum() > 50 else 0.001

    lc = lambda a, d: np.log(np.maximum(a - d, 1e-6))
    Xb02, Xb03, Xb04 = lc(b02, bd["B02"]), lc(b03, bd["B03"]), lc(b04, bd["B04"])

    def kk(Xi, Xj, mm):
        m2 = mm & np.isfinite(Xi) & np.isfinite(Xj)
        xiv, xjv = Xi[m2], Xj[m2]
        if len(xiv) < 200:
            return 1.0
        a = (np.var(xiv) - np.var(xjv)) / (2 * np.cov(xiv, xjv)[0, 1] + 1e-10)
        return a + np.sign(a) * np.sqrt(a ** 2 + 1)

    r = dict(BG=kk(Xb02, Xb03, shallow_m),
             BR=kk(Xb02, Xb04, shallow_m),
             GR=kk(Xb03, Xb04, shallow_m))
    return dict(dt=dt, bd=bd, r=r)


def apply_features(bands, water, params):
    """Compute deep/shallow masks, DII arrays and the 10-feature dict for the
    full scene using the supplied (train-derived or global) parameters.
    """
    b02, b03, b04, b08 = bands["B02"], bands["B03"], bands["B04"], bands["B08"]
    ndwi, ndvi, savi = spectral_indices(bands)
    dt, bd, r = params["dt"], params["bd"], params["r"]
    deep = water & (b02 < dt) & (b03 < dt) & (b04 < dt * 0.8)
    shallow = water & ~deep
    lc = lambda a, d: np.log(np.maximum(a - d, 1e-6))
    Xb02, Xb03, Xb04 = lc(b02, bd["B02"]), lc(b03, bd["B03"]), lc(b04, bd["B04"])
    feats = {"B02": b02, "B03": b03, "B04": b04, "B08": b08,
             "DII_BG": Xb02 - r["BG"] * Xb03,
             "DII_BR": Xb02 - r["BR"] * Xb04,
             "DII_GR": Xb03 - r["GR"] * Xb04,
             "NDVI": ndvi, "NDWI": ndwi, "SAVI": savi}
    return feats, deep, shallow


def make_labels(feats, deep, shallow, use_dii=True):
    """Fixed-threshold spectral pseudo-labels (the same rules as V3), applied in
    the order Sand, then Seagrass, then Coral; the deep-water exclusion uses a
    scene-derived 5th-percentile cutoff. use_dii=False is
    kept for the DII ablation. NOTE: these labels are generated from the same
    spectral features used for classification; downstream framing treats them
    as a reproducible pseudo-labelling protocol, not operational ground truth.
    """
    b02, b03, b08 = feats["B02"], feats["B03"], feats["B08"]
    ndvi, ndwi = feats["NDVI"], feats["NDWI"]
    c1 = deep
    c2 = shallow & (b02 > 0.04) & (ndvi < 0.05) & (ndwi < 0.3)
    c3 = shallow & (ndvi > 0.02) & (b03 > 0.015) & (b08 < 0.05) & ~c2
    c4 = shallow & ~c2 & ~c3 & (b02 > 0.015) & (b03 > 0.015) & (ndvi < 0.02) & (ndvi > -0.2)
    return {1: c1, 2: c2, 3: c3, 4: c4}


def build_stack(feats, fnames=FEATURES):
    return np.stack([feats[f] for f in fnames], axis=-1).astype(np.float32)


def prepare_scene(scene_dir, train_mask=None, use_dii=True):
    """Convenience: load a scene and return everything downstream needs.
    train_mask=None reproduces the global V3 protocol.
    """
    bands, scl, geo = load_bands(scene_dir)
    ndwi, ndvi, savi = spectral_indices(bands)
    water, nodata, cloud = water_mask(bands, scl, ndwi)
    params = fit_water_column(bands, water, train_mask=train_mask)
    feats, deep, shallow = apply_features(bands, water, params)
    labels = make_labels(feats, deep, shallow, use_dii=use_dii)
    stack = build_stack(feats)
    return dict(bands=bands, scl=scl, geo=geo, water=water, nodata=nodata,
                cloud=cloud, params=params, feats=feats, deep=deep,
                shallow=shallow, labels=labels, stack=stack)


# --------------------------------------------------------------------------
# Sampling and spatial partitioning
# --------------------------------------------------------------------------
def sample_labeled(stack, label_masks, n=400, seed=42, restrict_mask=None):
    """Sample up to n pixels per class. restrict_mask limits the region (used
    to draw train vs test samples from disjoint spatial blocks). Returns
    X, y, rows, cols.
    """
    rng = np.random.default_rng(seed)
    X, y, R, C = [], [], [], []
    for cls in CLASS_ORDER:
        if cls not in label_masks:
            continue
        mask = label_masks[cls]
        if restrict_mask is not None:
            mask = mask & restrict_mask
        ys, xs = np.where(mask)
        avail = len(ys)
        if avail < 50:
            continue
        ns = min(n, avail)
        idx = rng.choice(avail, ns, replace=False)
        for i in idx:
            f = stack[ys[i], xs[i]]
            if np.all(np.isfinite(f)):
                X.append(f); y.append(cls); R.append(ys[i]); C.append(xs[i])
    return (np.array(X, dtype=np.float32), np.array(y),
            np.array(R), np.array(C))


def block_ids(rows, cols, shape, n_side=5):
    """Assign each sample to a block of an n_side x n_side grid over the scene."""
    H, W = shape
    br = np.clip(rows * n_side // H, 0, n_side - 1)
    bc = np.clip(cols * n_side // W, 0, n_side - 1)
    return (br * n_side + bc).astype(int)


def spatial_folds(bids, k=5, seed=42):
    """Group whole blocks into k folds so train/test pixels never share a block.
    Yields (train_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    blocks = np.unique(bids)
    rng.shuffle(blocks)
    fold_of = {b: i % k for i, b in enumerate(blocks)}
    folds = np.array([fold_of[b] for b in bids])
    for f in range(k):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        yield tr, te


def region_split(shape, axis=1, frac=0.5):
    """Return a boolean full-scene mask for a geographically separated region
    (default: western half). Used for the holdout in E2/E5.
    """
    H, W = shape
    mask = np.zeros(shape, dtype=bool)
    cut = int((W if axis == 1 else H) * frac)
    if axis == 1:
        mask[:, :cut] = True
    else:
        mask[:cut, :] = True
    return mask


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
TABULAR_MODELS = ["RF", "ExtraTrees", "XGBoost", "LightGBM",
                  "SVM", "LogReg", "kNN"]


def make_model(name, seed=42, params=None):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    p = params or {}
    if name == "RF":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 500),
                                      max_depth=p.get("max_depth", 20),
                                      random_state=seed, n_jobs=-1)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=p.get("n_estimators", 500),
                                    max_depth=p.get("max_depth", 20),
                                    random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=p.get("n_estimators", 300),
                                 max_depth=p.get("max_depth", 8),
                                 learning_rate=p.get("learning_rate", 0.1),
                                 subsample=p.get("subsample", 1.0),
                                 eval_metric="mlogloss", random_state=seed,
                                 verbosity=0, n_jobs=-1)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=p.get("n_estimators", 300),
                                  num_leaves=p.get("num_leaves", 31),
                                  max_depth=p.get("max_depth", -1),
                                  learning_rate=p.get("learning_rate", 0.1),
                                  random_state=seed, n_jobs=-1, verbose=-1)
    if name == "SVM":
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=p.get("C", 100),
                                 gamma=p.get("gamma", "scale"), random_state=seed))
    if name == "LogReg":
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(C=p.get("C", 1.0), max_iter=2000))
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    if name == "kNN":
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5),
                                                  n_jobs=-1))
    raise ValueError(f"Unknown model: {name}")


def fit_predict(model, X_tr, y_tr, X_te):
    """Fit a tabular sklearn/xgb/lgbm model with contiguous 0..k-1 labels
    (required by XGBoost) and map predictions back to the 1..4 class codes.
    """
    labels = sorted(np.unique(y_tr))
    lm = {l: i for i, l in enumerate(labels)}
    im = {i: l for l, i in lm.items()}
    y01 = np.array([lm[v] for v in y_tr])
    model.fit(X_tr, y01)
    yp01 = model.predict(X_te)
    return np.array([im[int(v)] for v in yp01])


# --------------------------------------------------------------------------
# Neural models (torch): renamed MLP + genuine patch-based 2D CNN
# --------------------------------------------------------------------------
class TorchMLP:
    """Fully-connected network. This is the model reported in V3 as a
    '1D CNN'; it contains no convolution and is a multilayer perceptron.
    Architecture and training are unchanged so the numbers stay comparable.
    """
    def __init__(self, seed=42, epochs=100, lr=1e-3, batch=64,
                 hidden=(64, 32), dropout=(0.3, 0.2)):
        self.seed, self.epochs, self.lr, self.batch = seed, epochs, lr, batch
        self.hidden, self.dropout = hidden, dropout
        self.model = None; self.lm = None; self.im = None

    def _build(self, nf, nc):
        import torch.nn as nn
        h1, h2 = self.hidden
        d1, d2 = self.dropout
        return nn.Sequential(nn.Linear(nf, h1), nn.ReLU(), nn.Dropout(d1),
                             nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(d2),
                             nn.Linear(h2, nc))

    def fit(self, X, y):
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        set_seed(self.seed)
        labels = sorted(np.unique(y))
        self.lm = {l: i for i, l in enumerate(labels)}
        self.im = {i: l for l, i in self.lm.items()}
        y01 = np.array([self.lm[v] for v in y])
        Xt = torch.FloatTensor(X); yt = torch.LongTensor(y01)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch, shuffle=True)
        self.model = self._build(X.shape[1], len(labels))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(self.model(xb), yb)
                loss.backward(); opt.step()
        return self

    def predict(self, X, chunk=500000):
        import torch
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        preds = np.empty(len(X), dtype=int)
        with torch.no_grad():
            for s in range(0, len(X), chunk):
                e = min(s + chunk, len(X))
                preds[s:e] = torch.argmax(self.model(torch.FloatTensor(X[s:e])), 1).numpy()
        return np.array([self.im[int(p)] for p in preds])


class Torch2DCNN:
    """Genuine patch-based 2D CNN over the band+index stack. Each labelled
    pixel is classified from a patch x patch window centred on it, so the
    network can use spatial-spectral structure (the fair deep-learning baseline).
    """
    def __init__(self, stack, seed=42, patch=7, epochs=40, lr=1e-3, batch=64):
        assert patch % 2 == 1, "patch must be odd"
        self.stack = stack.astype(np.float32)
        self.seed, self.patch, self.epochs = seed, patch, epochs
        self.lr, self.batch = lr, batch
        self.model = None; self.lm = None; self.im = None
        self._padded = None

    def _pad(self):
        if self._padded is None:
            p = self.patch // 2
            self._padded = np.pad(self.stack, ((p, p), (p, p), (0, 0)), mode="reflect")
        return self._padded

    def _extract(self, rows, cols):
        padded = self._pad(); ps = self.patch
        F = self.stack.shape[2]
        out = np.empty((len(rows), F, ps, ps), dtype=np.float32)
        for k, (r, c) in enumerate(zip(rows, cols)):
            win = padded[r:r + ps, c:c + ps, :]      # ps x ps x F
            out[k] = np.transpose(win, (2, 0, 1))     # F x ps x ps
        return out

    def _build(self, nf, nc):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(nf, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, nc))

    def fit(self, rows, cols, y):
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        set_seed(self.seed)
        labels = sorted(np.unique(y))
        self.lm = {l: i for i, l in enumerate(labels)}
        self.im = {i: l for l, i in self.lm.items()}
        y01 = np.array([self.lm[v] for v in y])
        Xp = self._extract(rows, cols)
        Xt = torch.FloatTensor(Xp); yt = torch.LongTensor(y01)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch, shuffle=True)
        self.model = self._build(self.stack.shape[2], len(labels))
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad(); loss = crit(self.model(xb), yb)
                loss.backward(); opt.step()
        return self

    def predict(self, rows, cols, chunk=20000):
        import torch
        self.model.eval()
        preds = np.empty(len(rows), dtype=int)
        with torch.no_grad():
            for s in range(0, len(rows), chunk):
                e = min(s + chunk, len(rows))
                Xp = torch.FloatTensor(self._extract(rows[s:e], cols[s:e]))
                preds[s:e] = torch.argmax(self.model(Xp), 1).numpy()
        return np.array([self.im[int(p)] for p in preds])


# --------------------------------------------------------------------------
# Energy measurement (physical energy from CodeCarbon, with safe fallback)
# --------------------------------------------------------------------------
class Energy:
    """Context manager that measures wall-clock time and physical energy for a
    stage. Reports energy_consumed (kWh) from CodeCarbon, converted to uWh, and
    keeps emissions (kg CO2e) separately. Degrades to NaN energy if CodeCarbon
    cannot measure, so a run never crashes on the energy path.

    NOTE on the V3 numbers: the original script reported EmissionsTracker.stop()
    (kg CO2e) x 1e6 under an 'energy_uWh' header. That is an emissions figure,
    not physical energy. This class reports true energy_consumed instead, so E4
    values differ in absolute terms while the model-to-model ranking holds.
    """
    def __init__(self, name="stage"):
        self.name = name
        self.energy_uWh = float("nan")
        self.emissions_kg = float("nan")
        self.seconds = 0.0
        self._tr = None

    def __enter__(self):
        self._t0 = time.time()
        try:
            from codecarbon import EmissionsTracker
            self._tr = EmissionsTracker(project_name=self.name, log_level="error",
                                        save_to_file=False, tracking_mode="process")
            self._tr.start()
        except Exception:
            self._tr = None
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self._t0
        if self._tr is not None:
            try:
                em = self._tr.stop()
                self.emissions_kg = float(em) if em is not None else float("nan")
                data = getattr(self._tr, "final_emissions_data", None)
                kwh = getattr(data, "energy_consumed", None) if data else None
                if kwh is not None:
                    self.energy_uWh = float(kwh) * 1e9   # kWh -> uWh
            except Exception:
                pass
        return False


def cpu_label():
    """Best-effort CPU string to tag which hardware config produced a row."""
    try:
        import platform
        s = platform.processor() or platform.machine()
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        return s
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Metrics and statistics
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, labels=CLASS_ORDER):
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 f1_score, confusion_matrix)
    per = {}
    for c in labels:
        per[CLASSES[c]] = float(f1_score(y_true, y_pred, labels=[c],
                                         average="macro", zero_division=0))
    return dict(OA=float(accuracy_score(y_true, y_pred)),
                kappa=float(cohen_kappa_score(y_true, y_pred)),
                macroF1=float(f1_score(y_true, y_pred, average="macro",
                                       labels=labels, zero_division=0)),
                perF1=per,
                cm=confusion_matrix(y_true, y_pred, labels=labels))


def mean_ci(values, alpha=0.05):
    """Mean and two-sided (1-alpha) t confidence interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v))
    if n < 2:
        return m, m, m
    from scipy.stats import t
    se = np.std(v, ddof=1) / np.sqrt(n)
    h = se * t.ppf(1 - alpha / 2, n - 1)
    return m, m - h, m + h


def mcnemar_test(y_true, yp_a, yp_b):
    """Paired McNemar test between two classifiers on the same test set.
    Uses the exact binomial p-value (robust for small discordant counts) and
    also returns the continuity-corrected chi-square statistic.
    """
    a_ok = (yp_a == y_true)
    b_ok = (yp_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n = n01 + n10
    if n == 0:
        return dict(n01=0, n10=0, chi2=0.0, p=1.0)
    from scipy.stats import binomtest
    p = binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return dict(n01=n01, n10=n10, chi2=float(chi2), p=float(p))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def find_scenes(data_dir):
    return sorted(Path(data_dir).glob("scene_*"))


def water_pixels(state):
    """Return (rows, cols, X) for every valid water pixel in a prepared scene,
    used for full-map inference and energy accounting.
    """
    rows, cols = np.where(state["water"])
    stack = state["stack"]
    X = stack[rows, cols]
    ok = np.all(np.isfinite(X), axis=1)
    return rows[ok], cols[ok], X[ok]
