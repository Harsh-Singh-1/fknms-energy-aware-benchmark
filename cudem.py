"""
cudem.py -- download and grid-align a NOAA NCEI CUDEM topobathymetric DEM.

Provides the independent depth channel used by the bathymetry ablation
(05_ablations.py, the spectral_plus_depth arm). The download is cached at three
levels (URL list, individual tiles, the final grid-aligned raster), so anything
already present under the cache directory is reused and nothing is re-downloaded.

Source: NOAA NCEI Topobathymetric DEMs, https://coast.noaa.gov/htdata/raster2/elevation/
Default resolution is 1/3 arc-second (~10 m), matching the imagery grid.

build_depth(geo, cudem_dir) returns an (H, W) float32 array of water depth
(positive down) aligned to the scene grid, and writes a canonical depth.tif.
"""
import re
from pathlib import Path

import numpy as np

CUDEM = {
    "third": dict(base="https://coast.noaa.gov/htdata/raster2/elevation/NCEI_third_Topobathy_2014_8580",
                  urllist="urllist8580.txt", prefix="ncei13"),
    "ninth": dict(base="https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483",
                  urllist="urllist8483.txt", prefix="ncei19"),
}
TILE_RE = re.compile(r"(ncei(?:13|19))_n(\d{2})x(\d{2})_w(\d{3})x(\d{2})_\d{4}v\d+\.tif$", re.I)
TILE_DEG = 0.25


def bbox_lonlat(geo):
    """(w, s, e, n) of the scene grid in lon/lat."""
    import geopandas as gpd
    H, W = geo["shape"]; tf = geo["transform"]
    left, top = tf * (0, 0); right, bottom = tf * (W, H)
    g = gpd.GeoSeries.from_wkt(
        [f"POINT({left} {top})", f"POINT({right} {bottom})"], crs=geo["crs"]).to_crs(4326)
    return (min(g.x), min(g.y), max(g.x), max(g.y))


def _tile_bounds(name):
    m = TILE_RE.search(name)
    if not m:
        return None
    nwlat = int(m.group(2)) + int(m.group(3)) / 100.0
    nwlon = int(m.group(4)) + int(m.group(5)) / 100.0
    west = -nwlon
    return (west, nwlat - TILE_DEG, west + TILE_DEG, nwlat)


def _select_tiles(urls, bbox, buffer=0.005):
    w, s, e, n = bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer
    return [u for u in urls
            if (tb := _tile_bounds(u)) and tb[2] >= w and tb[0] <= e and tb[3] >= s and tb[1] <= n]


def _download(url, dest):
    import requests
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  reuse {dest.name}"); return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    return dest


def _get_urls(cfg, cache):
    txt = _download(f"{cfg['base']}/{cfg['urllist']}", Path(cache) / cfg["urllist"])
    urls = []
    for line in Path(txt).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.lower().endswith(".tif") and cfg["prefix"] in line.lower():
            urls.append(line if line.startswith("http") else f"{cfg['base']}/{line}")
    return urls


def build_depth(geo, cudem_dir, cudem_res="third", depth_raster=None):
    """Depth (H x W float32, positive down) aligned to `geo`, cached. Writes depth.tif."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling
    cache = Path(cudem_dir); cache.mkdir(parents=True, exist_ok=True)
    H, W = geo["shape"]
    canonical = cache / "depth.tif"
    if canonical.exists():
        print(f"reuse aligned depth raster {canonical}")
        with rasterio.open(canonical) as ds:
            return ds.read(1).astype(np.float32)

    if depth_raster:
        srcs = [Path(depth_raster)]
        print(f"using supplied DEM {depth_raster}")
    else:
        cfg = CUDEM[cudem_res]
        sel = _select_tiles(_get_urls(cfg, cache), bbox_lonlat(geo))
        if not sel:
            raise SystemExit("No CUDEM tiles intersect the study area; try cudem_res='ninth'.")
        print(f"{len(sel)} CUDEM tiles cover the study area")
        srcs = [_download(u, cache / "tiles" / Path(u).name) for u in sel]

    dss = [rasterio.open(s) for s in srcs]
    mosaic, mtf = merge(dss)
    src_crs, src_nodata = dss[0].crs, dss[0].nodata
    for d in dss:
        d.close()
    dst = np.full((H, W), np.nan, np.float32)
    reproject(source=mosaic[0], destination=dst, src_transform=mtf, src_crs=src_crs,
              dst_transform=geo["transform"], dst_crs=geo["crs"],
              src_nodata=src_nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    depth = np.where(np.isfinite(-dst), -dst, 0.0).astype(np.float32)   # elevation -> depth
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                crs=geo["crs"], transform=geo["transform"], compress="deflate")
    with rasterio.open(canonical, "w", **prof) as out:
        out.write(depth, 1)
    print(f"saved aligned depth raster {canonical}")
    return depth
