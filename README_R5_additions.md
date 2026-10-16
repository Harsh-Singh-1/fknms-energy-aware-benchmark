# Round 5 additions (V4 supplementary package)

This package extends the V3 codebase with four scripts added for the final
revision round. All existing scripts, reference outputs, and the URM cache are
unchanged from V3; nothing published was modified.

## New scripts

- `07_fig9_extension.py` - learning-curve extension for the deep models (R2
  comment 9). Adds 20,000 and 50,000 samples per class for MLP and 2D_CNN by
  calling `buffered_benchmark()` from `05_ablations.py` unchanged, so the
  protocol (300 m buffer, 5-fold spatial CV + geographic holdout, seed 42,
  patch 9, CNN 60 epochs, 100k test cap) is inherited, with no re-tuning. Logs
  every training draw's actual per-class counts (`learning_curve_draws.csv`)
  because `sample_labeled` draws min(n, available) without replacement, and
  writes a merged `ablation_learning_curve.csv` so `06_figures.py` regenerates
  Figure 9 with all six sizes unmodified.
- `08_holdout_majority_rate.py` - class shares of the exact buffered
  geographic-holdout test pixels, per scene and pooled (R2 comment 2). The
  pooled seagrass share is the correct no-information baseline for Section V-C
  and Figure 3a.
- `09_block_audit.py` - per-scene test-block counts per protocol from
  `benchmark_block_metrics.csv` (R2 comment 11). Documents that block counts
  are determined per scene after cloud masking and the boundary buffer:
  holdout 11 + 11 + 11 + 11 + 11 + 11 + 11 + 11 + 10 (wet_2024) = 98;
  cross-validation counts vary 14 to 20 per scene, totaling 144.
- `10_stats_additions.py` - Friedman omnibus test across the nine classifiers
  over the nine paired scenes, per protocol, with Kendall's W and mean ranks
  (R2 comment 1), plus per-scene Cohen's kappa under the geographic holdout,
  giving the kappa range quoted in the abstract and the Table 14 kappa columns
  (R2 comment 5). Reads `reference_outputs/benchmark_summary.csv` by default
  and writes `friedman_omnibus.csv`, `model_mean_ranks.csv`,
  `table14_kappa.csv`, and `kappa_range.txt`.

## Run order for the revision

```
python 09_block_audit.py
python 08_holdout_majority_rate.py --data <scene_dir>
python 10_stats_additions.py
python 07_fig9_extension.py --data <scene_dir> --sizes 20000
python 07_fig9_extension.py --data <scene_dir> --sizes 50000
python 06_figures.py            # regenerates fig9_learning_curve.png
```

`<scene_dir>` is the directory holding the nine downloaded scenes (from
`01_download_data.py`). 09 and 10 are instant; 08 takes minutes (no
training); 07 is the long job - run 20k first so partial results are available
early, and budget hours for the 50k CNN fits on CPU.

`run_all_overnight.py` runs the same sequence unattended, logging to
`run_all.log`; completed learning-curve sizes are skipped on a re-run.
