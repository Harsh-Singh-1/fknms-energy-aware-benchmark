"""
01_download_data.py -- acquire all inputs for the analysis.

Establishes one common 10 m grid for the study area, then:
  1. Sentinel-2 Level-2A scenes from Microsoft Planetary Computer, mosaicked and
     clipped onto that grid (data/scene_YYYYMMDD/).
  2. A NOAA CUDEM depth raster aligned to the same grid (data/cudem/depth.tif),
     used only by the bathymetry ablation.
  3. The Unified Reef Map field reference (data/urm_keys.geojson) is bundled with
     this package as the exact snapshot used in the paper, so no query is needed;
     the current layer is at https://myfwc.com/research/gis/fisheries/unified-reef-map/.

The grid is derived deterministically from the study bounding box and CRS, so it
does not depend on any pre-existing file and is reproducible on a clean machine.

The paper uses nine scenes: three 2024 seasonal windows (dry, spring, wet) plus
six additional dates from Oct 2022 to Feb 2024, each screened to under 5 percent
in-area cloud (the three seasonal scenes were kept regardless). Selecting the
clearest dates in a range reproduces an equivalent set; the exact dates used are
listed in the README for byte-for-byte reproduction.

Install once:  pip install pystac-client planetary-computer
Run:
  python 01_download_data.py --start 2022-10-01 --end 2024-12-31 --max_scenes 9
  python 01_download_data.py --skip_cudem            # imagery only
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds

import cudem

HERE = Path(__file__).resolve().parent
BBOX_LONLAT = (-81.85, 24.38, -81.35, 24.72)   # w, s, e, n  -- Lower Keys subregion
CRS = "EPSG:32617"                              # UTM Zone 17N
RES = 10.0
BANDS10 = ["B02", "B03", "B04", "B08"]
CLOUD_SCL = {3, 8, 9, 10}                       # cloud shadow, cloud med/high, thin cirrus


def build_grid():
    """Deterministic common grid (transform, shape) from the study bbox."""
    w, s, e, n = transform_bounds("EPSG:4326", CRS, *BBOX_LONLAT)
    left, right = np.floor(w / RES) * RES, np.ceil(e / RES) * RES
    bottom, top = np.floor(s / RES) * RES, np.ceil(n / RES) * RES
    width, height = int(round((right - left) / RES)), int(round((top - bottom) / RES))
    return dict(crs=CRS, transform=from_origin(left, top, RES, RES),
                shape=(height, width), width=width, height=height,
                bounds=(left, bottom, right, top))


def _to_grid(href, grid, dtype, resampling):
    out = np.zeros((grid["height"], grid["width"]), dtype=dtype)
    with rasterio.open(href) as src:
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=src.nodata, dst_nodata=0, resampling=resampling)
    return out


def _mosaic(hrefs, grid, dtype, resampling):
    acc = np.zeros((grid["height"], grid["width"]), dtype=dtype)
    for h in hrefs:
        acc = np.where(acc == 0, _to_grid(h, grid, dtype, resampling), acc)
    return acc


def _write(path, arr, grid, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"], width=grid["width"],
                       count=1, dtype=dtype, crs=grid["crs"], transform=grid["transform"],
                       nodata=0, compress="deflate") as dst:
        dst.write(arr, 1)


def download_scenes(grid, a):
    import pystac_client
    import planetary_computer
    w, s, e, n = transform_bounds(grid["crs"], "EPSG:4326", *grid["bounds"])
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    search = cat.search(collections=["sentinel-2-l2a"], bbox=[w, s, e, n],
                        datetime=f"{a.start}/{a.end}", query={"eo:cloud_cover": {"lt": a.cloud}})
    try:
        items = list(search.items())
    except AttributeError:
        items = list(search.get_items())
    print(f"STAC items returned: {len(items)}")

    by_day = {}
    for it in items:
        if a.months and it.datetime.month not in a.months:
            continue
        by_day.setdefault(it.datetime.strftime("%Y%m%d"), []).append(it)
    days = sorted(by_day, key=lambda d: np.mean(
        [i.properties.get("eo:cloud_cover", 100) for i in by_day[d]]))

    outdir = Path(a.data); outdir.mkdir(parents=True, exist_ok=True)
    written, last = 0, None
    for d in days:
        if written >= a.max_scenes:
            break
        folder = outdir / f"scene_{d}"
        if folder.exists():
            print(f"  {d}: already present, skipping"); written += 1; continue
        dt = np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        if last is not None and abs((dt - last) / np.timedelta64(1, "D")) < a.min_gap_days:
            continue
        its = by_day[d]
        scl = _mosaic([i.assets["SCL"].href for i in its], grid, "uint8", Resampling.nearest)
        valid = scl > 0
        cf = float(np.isin(scl, list(CLOUD_SCL))[valid].sum()) / max(valid.sum(), 1)
        cov = float(valid.sum()) / valid.size
        if cov < a.min_cover or cf > a.area_cloud:
            print(f"  {d}: skip (coverage {cov:.2f}, area cloud {cf:.2f})"); continue
        folder.mkdir(parents=True)
        for b in BANDS10:
            _write(folder / f"{b}.tif",
                   _mosaic([i.assets[b].href for i in its], grid, "uint16", Resampling.bilinear),
                   grid, "uint16")
        _write(folder / "SCL.tif", scl, grid, "uint8")
        (folder / "metadata.json").write_text(json.dumps(dict(
            date=f"{d[:4]}-{d[4:6]}-{d[6:]}", label=f"s2_{d}",
            avg_cloud=round(float(np.mean([i.properties.get("eo:cloud_cover", 0) for i in its])), 2),
            tiles=sorted({i.properties.get("s2:mgrs_tile", "") for i in its})), indent=2))
        print(f"  {d}: wrote {folder.name}  area_cloud={cf:.2f}")
        written += 1; last = dt
    print(f"Scenes present/written: {written}")


def main():
    ap = argparse.ArgumentParser(description="Download Sentinel-2 + CUDEM onto a common grid.")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--start", default="2022-10-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--cloud", type=float, default=10, help="max scene-level cloud percent")
    ap.add_argument("--area_cloud", type=float, default=0.05, help="max cloud fraction in study area")
    ap.add_argument("--min_cover", type=float, default=0.5, help="min valid coverage of the bbox")
    ap.add_argument("--months", type=int, nargs="*", default=None)
    ap.add_argument("--max_scenes", type=int, default=9)
    ap.add_argument("--min_gap_days", type=int, default=20)
    ap.add_argument("--cudem_res", choices=["third", "ninth"], default="third")
    ap.add_argument("--skip_cudem", action="store_true", help="skip the CUDEM depth download")
    a = ap.parse_args()

    grid = build_grid()
    print(f"Common grid: {grid['shape'][0]} x {grid['shape'][1]} at {RES:.0f} m, {CRS}")
    print(f"Study bbox (lon/lat): {BBOX_LONLAT}")

    print("\n[1/3] Sentinel-2 scenes")
    download_scenes(grid, a)

    print("\n[2/3] CUDEM depth raster")
    if a.skip_cudem:
        print("  skipped (--skip_cudem)")
    else:
        cudem.build_depth(grid, Path(a.data) / "cudem", cudem_res=a.cudem_res)

    print("\n[3/3] Unified Reef Map reference")
    urm = Path(a.data) / "urm_keys.geojson"
    print(f"  {'found' if urm.exists() else 'MISSING'}: {urm}")
    if not urm.exists():
        print("  The URM snapshot ships with this package; restore data/urm_keys.geojson.")
    print("\nDone.")


if __name__ == "__main__":
    main()
