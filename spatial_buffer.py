import numpy as np
from scipy.ndimage import distance_transform_edt
BENTHIC3 = [2, 3, 4]
def buffered_test_mask(test_mask, train_pool, buffer_px):
    if buffer_px is None or buffer_px <= 0:
        return test_mask
    if not train_pool.any():
        return test_mask
    dist_to_train = distance_transform_edt(~train_pool)
    return test_mask & (dist_to_train >= buffer_px)
def block_confusions(y_true, y_pred, block_ids, classes=BENTHIC3):
    classes = list(classes); pos = {c: i for i, c in enumerate(classes)}
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); block_ids=np.asarray(block_ids)
    out = {}
    for b in np.unique(block_ids):
        m = block_ids == b; cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for t, p in zip(y_true[m], y_pred[m]):
            it, ip = pos.get(int(t)), pos.get(int(p))
            if it is not None and ip is not None: cm[it, ip] += 1
        out[int(b)] = cm
    return out
def flatten_cm(cm, classes=BENTHIC3):
    classes=list(classes); d={}
    for i,t in enumerate(classes):
        for j,p in enumerate(classes): d[f"c{t}_{p}"]=int(cm[i,j])
    return d
