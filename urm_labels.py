"""
urm_labels.py
Turn Florida's Unified Reef Map (URM) into supervised habitat labels on a
scene grid.

This replaces the fixed-threshold spectral pseudo-labels with independent,
field-validated labels, which removes the circularity of same-source labeling.
Benthic labels (Sand/Bare=2, Seagrass=3, Coral/Hard-bottom=4) come entirely
from URM. Open Deep Water=1 is optional and, when requested, is taken from the
scene water mask where URM is unmapped, because URM does not map open water.

The download path can cache to a local file with
save_urm so the FWC service is queried only once.
"""
import numpy as np
from pathlib import Path

# URM class-field value -> study class, by case-insensitive substring.
CROSSWALK = [
    (["seagrass"], 3),
    (["coral", "reef", "hardbottom", "hard bottom", "pavement", "ridge",
      "rubble", "rock", "scattered", "aggregate", "patch", "colonized"], 4),
    (["sediment", "sand", "mud", "unconsolidated", "bare"], 2),
]


def crosswalk_value(v):
    s = str(v).lower()
    for keys, code in CROSSWALK:
        if any(k in s for k in keys):
            return code
    return 0   # unmapped / not classified -> excluded


def pick_field(gdf, field):
    import pandas as pd
    if field and field in gdf.columns:
        return field
    for cand in ["ClassLv1", "ClassLv2", "ClassLv3", "ClassLv4", "ClassLv0",
                 "FIRST_ClassLv1", "FIRST_ClassLv2", "FIRST_ClassLv3",
                 "FIRST_ClassLv4", "FIRST_ClassLv0",
                 "CLASS_NAME", "Class_Name", "DESCRIPT", "Habitat"]:
        if cand in gdf.columns:
            return cand
    geom = gdf.geometry.name
    cols = [c for c in gdf.columns if c != geom]
    for c in cols:                                   # prefer class/habitat-like names
        if "class" in c.lower() or "hab" in c.lower():
            return c
    for c in cols:                                   # else first text column
        if pd.api.types.is_string_dtype(gdf[c]) or gdf[c].dtype == object:
            return c
    if cols:
        return cols[0]
    raise SystemExit(f"No class field found; columns={list(gdf.columns)}. Pass --field.")


def fetch_urm(bbox_lonlat, crs, urm=None, download=False, service_url=None, save_urm=None):
    """Return URM polygons as a GeoDataFrame reprojected to `crs`.
    Reads a local file if `urm` is given, else queries the ArcGIS service.
    """
    import geopandas as gpd
    if urm and Path(urm).exists():
        gdf = gpd.read_file(urm)
    elif download:
        if not service_url:
            raise SystemExit("download=True requires service_url")
        import requests
        w, s, e, n = bbox_lonlat
        feats, offset = [], 0
        while True:
            params = dict(where="1=1", outFields="*", f="geojson",
                          geometry=f"{w},{s},{e},{n}", geometryType="esriGeometryEnvelope",
                          inSR=4326, spatialRel="esriSpatialRelIntersects",
                          resultOffset=offset, resultRecordCount=2000)
            r = requests.get(service_url, params=params, timeout=180)
            r.raise_for_status()
            got = r.json().get("features", [])
            feats += got
            if len(got) < 2000:
                break
            offset += 2000
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        if save_urm:
            gdf.to_file(save_urm, driver="GeoJSON")
    else:
        raise SystemExit("Provide urm=<file> or download=True with service_url.")
    return gdf.to_crs(crs)


def rasterize_benthic(gdf, field, shape, transform):
    """Rasterize crosswalked URM polygons to the grid (0 where unmapped)."""
    from rasterio.features import rasterize
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["_code"] = gdf[field].map(crosswalk_value)
    gdf = gdf[gdf["_code"] > 0]
    if len(gdf) == 0:
        return np.zeros(shape, dtype="uint8")
    shapes = ((geom, int(c)) for geom, c in zip(gdf.geometry, gdf["_code"]))
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                     dtype="uint8", all_touched=False)


def build_label_raster(state, gdf, field=None, include_deep=False):
    """Return (label_raster HxW uint8 with 0=unlabeled, field_used).
    Benthic labels from URM, restricted to water pixels; Deep Water optional.
    """
    fld = pick_field(gdf, field)
    H, W = state["geo"]["shape"]
    lab = rasterize_benthic(gdf, fld, (H, W), state["geo"]["transform"])
    lab = lab.copy()
    lab[~state["water"]] = 0            # benthic labels only over water
    if include_deep:
        deep = state["deep"] & (lab == 0)
        lab[deep] = 1
    return lab, fld


def label_masks(lab, state, classes):
    """Dict {class: boolean mask} over valid water pixels for the given classes."""
    return {c: (lab == c) & state["water"] for c in classes}
