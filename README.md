# Energy-aware benthic habitat classification, Lower Florida Keys

Code and reference outputs for the study benchmarking nine classifiers for
Sentinel-2 benthic habitat mapping in the Lower Florida Keys National Marine
Sanctuary, using independent field-validated labels from Florida's Unified Reef
Map, spatially separated evaluation with a train-test buffer, and per-stage
energy measurement on models fitted to the same task as the accuracy benchmark.

The pipeline reproduces the paper's primary results: the accuracy tables under
the two spatial protocols, the spatially aware significance tables, the
per-stage energy tables, the test-cap sensitivity check, the pseudo-label
versus field-reference comparison, the feature and bathymetry ablations, and
the figures.

Single author: Harsh Deep Singh Narula.

## Data

Scene imagery (Sentinel-2 Level-2A) and the NOAA CUDEM depth raster are not in
this repository; `01_download_data.py` downloads both from public sources
(Microsoft Planetary Computer and NOAA) into `data/`. The Unified Reef Map field
reference used in the paper is included as `data/urm_keys.geojson`; step 01
does not regenerate it. See Data sources below for details.
`README_R5_additions.md` describes the revision-round scripts (07-10) and the
run order.

## Project structure

```
fknms_ieee_access/
├── README.md
├── README_R5_additions.md   # Round 5 (revision) scripts and run order
├── LICENSE                  # MIT
├── requirements.txt
│
├── fknms_core.py            # scene IO, water masking, Lyzenga DII, feature stack,
│                            #   the nine-model zoo, patch-based 2D CNN, CodeCarbon energy
├── urm_labels.py            # Unified Reef Map crosswalk to 3 classes + rasterization
├── spatial_buffer.py        # train-test buffer masks + per-block confusion export
├── eval_engine.py           # shared evaluation engine (spatial folds, metrics, maps)
├── urm_draw.py              # the URM-supervised training draw + SHA-256 digest
├── cudem.py                 # NOAA CUDEM depth downloader and grid alignment
│
├── 01_download_data.py      # Sentinel-2 (Planetary Computer) + CUDEM onto a common grid
├── 02_pseudo_vs_urm.py      # pseudo-label pipeline vs the field reference (circularity)
├── 03_benchmark.py          # URM-supervised benchmark, buffered spatial validation
├── 03b_statistics.py        # scene-stratified block bootstrap + scene-level + pairwise tests
├── 03c_cap_sensitivity.py   # accuracy with vs without the 100,000-pixel test cap
├── 04_energy.py             # per-stage energy, models fitted on the URM-supervised draw
├── 05_ablations.py          # DII, independent bathymetry, and learning-curve ablations
├── 06_figures.py            # manuscript figures from the output CSVs
│
├── 07_fig9_extension.py     # Round 5: MLP + 2D CNN learning curve at 20k and 50k/class
├── 08_holdout_majority_rate.py # Round 5: holdout test-pixel class shares (baseline)
├── 09_block_audit.py        # Round 5: per-scene test-block counts
├── 10_stats_additions.py    # Round 5: Friedman omnibus test + per-scene kappa
├── run_all_overnight.py     # Round 5 runner (09, 08, 10, 07, 06); logs to run_all.log
├── make_packet.py           # author's packaging script; not needed to reproduce
│
├── data/                    # populated by 01; ships with urm_keys.geojson (field reference)
├── outputs/                 # CSVs written by 02-05, 03b, 03c, 07-10
├── figures/                 # PNGs written by 03 and 06
└── reference_outputs/       # the CSVs and figures from the run reported in the paper
```

## Requirements

Python 3.10 or newer. Install everything with:

```bash
pip install -r requirements.txt
```

A CPU-only PyTorch build is sufficient; the energy figures reported in the paper
are CPU figures (Intel Core i7, 16 GB RAM). GPU is not used.

## Data sources

- **Sentinel-2 Level-2A** surface reflectance from Microsoft Planetary Computer
  (collection `sentinel-2-l2a`, STAC at
  https://planetarycomputer.microsoft.com/api/stac/v1). Bands B02, B03, B04, B08
  at 10 m plus the Scene Classification Layer. Downloaded and mosaicked by step 01.
- **NOAA NCEI CUDEM** topobathymetric DEM (1/3 arc-second, ~10 m) from
  https://coast.noaa.gov/htdata/raster2/elevation/. Used only by the bathymetry
  ablation. Downloaded and grid-aligned by step 01 (or on demand by step 05).
- **Unified Reef Map** (Florida Fish and Wildlife Conservation Commission, Fish
  and Wildlife Research Institute). The exact snapshot used in the paper ships in
  this package as `data/urm_keys.geojson`, so no query is required and the field
  reference is fixed. The current layer is at
  https://myfwc.com/research/gis/fisheries/unified-reef-map/.

Study area: the Lower Keys subregion, longitude 81.85W to 81.35W, latitude 24.38N
to 24.72N, projected to EPSG:32617 (UTM Zone 17N) on a common 10 m grid. The grid
is derived deterministically from this bounding box in step 01, so it does not
depend on any pre-existing file.

## How to reproduce

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. imagery + depth onto the common grid (needs internet)
python 01_download_data.py --start 2022-10-01 --end 2024-12-31 --max_scenes 9
#    imagery only, skipping the CUDEM download:
#    python 01_download_data.py --skip_cudem

# 3. pseudo-label pipeline vs the field reference (the circularity demonstration)
python 02_pseudo_vs_urm.py

# 4. the main benchmark (both spatial protocols; also writes the map figures)
python 03_benchmark.py

# 5. spatially aware significance, from the per-block export in step 4
python 03b_statistics.py

# 6. sensitivity of the accuracies to the 100,000-pixel test cap
python 03c_cap_sensitivity.py

# 7. per-stage energy on this machine (name it so the tag is recorded)
python 04_energy.py --label i7_16GB

# 8. the three ablations (DII, bathymetry, learning curve)
python 05_ablations.py

# 9. figures from the output CSVs
python 06_figures.py

# 10. revision-round analyses (07-10; see README_R5_additions.md):
#     09, 08, 10, then 07 at 20,000 and 50,000 per class, then 06 again
python run_all_overnight.py
```

The nine scenes used in the paper are three 2024 seasonal windows (dry, spring,
wet) plus six additional dates from October 2022 to February 2024. Selecting the
clearest low-cloud dates in the range above reproduces an equivalent set; the
three seasonal scenes were retained regardless of the cloud screen, and the
wet-season scene at 7.52 percent in-area cloud is the one above the 5 percent
level (its cloud and cloud-shadow pixels are masked as for every scene).

### Runtime and resumption

Steps 03_benchmark, 03c, and 04_energy dominate the total runtime. 04_energy is
resumable: rerun the same command after any interruption and it continues from
where it stopped. To verify the numbers without rerunning, use
`reference_outputs/`, which contains every CSV and figure from the run reported
in the paper.

## Measurement protocol notes

- Every model is fitted on the same URM-supervised training draw as the
  accuracy benchmark (western half of each scene, 5,000 samples per class, seed
  42). `04_energy.py` writes a SHA-256 digest of each scene's sorted (row,
  column, label) training triplets to `outputs/train_draw_digest.csv`; the
  digests from the reported run are in `reference_outputs/train_draw_digest.csv`.
- Each stage is measured five times per scene, except SVM and
  k-nearest-neighbour full-scene inference, which are measured with a single
  pass per scene; one pass aggregates roughly eleven million predictions. On
  this task the fitted SVM retains 9,565 to 11,358 of the 15,000 training
  samples as support vectors (`reference_outputs/svm_support.csv`), and
  support-vector inference cost scales with that count.
- The block bootstrap resamples blocks with replacement within each scene
  (stratified by scene), 5,000 iterations, seed 20240517, with the resampled
  block set shared across models so pairwise differences are paired.
- Map figures use a single classifier for every scene, logistic regression
  (`03_benchmark.py --map_model LogReg`, the default), the model with the
  highest mean geographic-holdout accuracy.

## Results summary

### URM-supervised accuracy (independent labels, spatial validation)

Mean over the nine scenes. Geo-holdout trains on the western half and tests on
the eastern half with a 300 m gap; spatial CV uses 5x5 blocks in five folds with
the same buffer.

| Model       | Geo-holdout OA | Geo-holdout kappa | Spatial-CV OA |
|-------------|:--------------:|:-----------------:|:-------------:|
| LogReg      | 0.549          | 0.278             | 0.557         |
| SVM         | 0.540          | 0.289             | 0.573         |
| MLP         | 0.533          | 0.288             | 0.547         |
| LightGBM    | 0.524          | 0.249             | 0.566         |
| XGBoost     | 0.519          | 0.241             | 0.562         |
| Extra-Trees | 0.515          | 0.239             | 0.557         |
| RF          | 0.514          | 0.232             | 0.559         |
| kNN         | 0.505          | 0.222             | 0.543         |
| 2D CNN      | 0.506          | 0.230             | 0.531         |

A majority-class baseline (always predict seagrass, the 53 percent class) scores
0.53. The classifiers exceed it by about 0.02 (holdout) to 0.04 (cross
validation); the gain comes from the minority Sand and Coral classes. No pairwise
difference is significant under the conservative scene-level test. Evaluating on
all held-out URM-labeled pixels instead of the 100,000-pixel capped sample
changes overall accuracy by at most 0.0047 across every scene and model
(`reference_outputs/test_cap_sensitivity.csv`).

### Operational energy (one Intel Core i7, 16 GB RAM; URM-fitted models)

Per-model energy in microwatt-hours, mean over scenes. Operational total is
training at 5,000 samples per class plus full-scene inference over roughly eleven
million water pixels.

| Model       | Train   | Full-scene inference | Operational | x LogReg |
|-------------|--------:|---------------------:|------------:|:--------:|
| LogReg      |   2,284 |               52,455 |      54,739 | 1.0x     |
| MLP         | 402,947 |               24,305 |     427,252 | 7.8x     |
| XGBoost     |  23,831 |              555,023 |     578,854 | 10.6x    |
| kNN         |       9 |              896,821 |     896,830 | 16.4x    |
| RF          |  32,178 |              940,184 |     972,362 | 17.8x    |
| Extra-Trees |  13,520 |              997,369 |   1,010,889 | 18.5x    |
| LightGBM    |  12,514 |            1,602,761 |   1,615,275 | 29.5x    |
| 2D CNN      | 826,055 |            3,739,114 |   4,565,169 | 83.4x    |
| SVM         |  57,340 |           29,128,441 |  29,185,781 | 533.2x   |

The ranking inverts between the two stages: k-nearest-neighbours is nearly free
to fit but costs 0.90 Wh to apply, and the support vector machine, far cheaper
to fit than the neural networks, is by far the most expensive model to deploy
(19.7 to 39.0 Wh per scene, scaling with its retained support vectors). Only
logistic regression is cheap on both axes, and it also has the highest mean
holdout accuracy. Model-independent stages (preprocessing 11,100, water-column
correction 12,000, feature extraction 10,400 microwatt-hours, about 33,500
combined per scene) are small beside a single full-scene inference for most
models.

### Pseudo-label pipeline versus the field reference

Trained and tested on adaptive spectral pseudo-labels, the classifiers reach 98.5
to 99.4 percent internal accuracy. Compared against the Unified Reef Map over the
same area, with deep water excluded from both sides, the same pipeline agrees only
about 25 percent of the time (Random Forest 24.5 percent, XGBoost 25.3 percent,
over roughly 66 million pixels), with Cohen's kappa near zero. The confusion is a
collapse toward Coral/Hard-bottom: about 88 percent recall on true coral but only
0.9 percent on seagrass and 3 percent on sand, so the agreement is essentially the
coral overlap alone.

### Ablations

- **DII (Lyzenga water-column correction):** removing the three depth-invariant
  indices changes overall accuracy by at most 0.005 for the tabular models with no
  consistent direction, an honest null under independent labels.
- **Independent bathymetry (CUDEM):** adding a depth channel raises overall
  accuracy by 0.002 to 0.010 (spatial CV) and by 0.005 to 0.009 for seven of eight
  models (geo-holdout); Coral/Hard-bottom F1 rises by about 0.009 on average, up to
  0.014 for logistic regression.
- **Learning curve:** accuracy plateaus by 5,000 samples per class; the neural
  networks gain the most from added data but do not overtake the low-energy models
  at any size.

## Important caveats

- The pseudo-label protocol uses adaptive spectral thresholds, so its reference
  samples are generated from the same spectral features used for classification.
  This circularity inflates its absolute accuracy and makes that number a measure
  of self-consistency, not of agreement with the seabed. The URM-supervised
  results avoid this by using labels independent of the imagery.
- Measured accuracy is agreement with the best available regional reference and is
  bounded by the Unified Reef Map's own label noise and by the crosswalk that
  groups finer URM classes into three.
- Energy is estimated with CodeCarbon, which derives energy from processor power
  models and load rather than a wall-socket meter (there is no RAPL counter on the
  Windows host). Relative comparisons on identical hardware are more reliable than
  absolute values, and runtimes vary by machine, so they are not tabulated per
  model.

## Benthic classes

| Code | Class             | Unified Reef Map source classes (grouped)                    |
|:----:|-------------------|--------------------------------------------------------------|
| 2    | Sand/Bare         | unconsolidated sediment, sand, mud, bare substrate           |
| 3    | Seagrass          | continuous seagrass, patchy (discontinuous) seagrass         |
| 4    | Coral/Hard-bottom | coral reef, aggregate reef, patch reef, pavement, ridge, rubble, rock, scattered coral and rock, colonized hardbottom |

Deep Water is not a URM benthic class and is excluded from the benthic task; where
requested it is taken from the scene water mask in areas URM leaves unmapped.

## Reference outputs

`reference_outputs/` holds the CSVs and figures from the run reported in the
paper, under the same filenames the scripts write, so results can be verified
without rerunning the full pipeline. Running `03b_statistics.py` against
`reference_outputs/benchmark_block_metrics.csv` reproduces the block bootstrap
confidence intervals in `stats_block_ci.csv` exactly.

## Citation

Pending; manuscript under review at IEEE Access.
