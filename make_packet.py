"""
make_packet.py -- assemble and verify the supplementary package in this folder.

Copies the unchanged pipeline files byte-identical from the previous package,
applies the single map-model change to 03_benchmark.py, populates data/ and
reference_outputs/ (regenerating the statistics CSVs and the energy figures
from the shipped inputs), verifies the manifest and key values, and zips the
result. Run once from this folder:

    python make_packet.py

Expects the sibling folders:
    ../fknms_supplementary_materials_IEEE_access   (previous package)
    ../Revised_analysis_v3                          (measurement outputs)
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2P = HERE.parent / "fknms_supplementary_materials_IEEE_access"
V3A = HERE.parent / "Revised_analysis_v3"

UNCHANGED = ["fknms_core.py", "urm_labels.py", "spatial_buffer.py", "eval_engine.py",
             "cudem.py", "01_download_data.py", "02_pseudo_vs_urm.py",
             "05_ablations.py", "06_figures.py", "requirements.txt"]
REF_KEEP = ["benchmark_block_metrics.csv", "benchmark_confusion.csv", "benchmark_summary.csv",
            "pseudo_vs_urm_confusion.csv", "pseudo_vs_urm_perscene.csv",
            "ablation_bathymetry.csv", "ablation_dii.csv", "ablation_learning_curve.csv"]
REF_FIGS_KEEP = ["fig4_perclass_f1.png", "fig5_confusion.png",
                 "fig9_learning_curve.png", "fig10_bathymetry.png"]
NEW_SCRIPTS = ["03b_statistics.py", "03c_cap_sensitivity.py", "04_energy.py",
               "urm_draw.py", "README.md"]


def fail(msg):
    print(f"FAILED: {msg}")
    sys.exit(1)


def preflight():
    """Confirm the interpreter running this script has the packages the pipeline
    scripts import, since they are launched with the same interpreter."""
    missing = []
    for mod in ["numpy", "pandas", "scipy", "matplotlib"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"the interpreter running this script is missing {', '.join(missing)}.\n"
             f"  interpreter: {sys.executable}\n"
             f"  Install them for this interpreter:\n"
             f'    "{sys.executable}" -m pip install -r requirements.txt\n'
             f"  or rerun this script with the interpreter that has them, for example:\n"
             f'    "<path to python.exe>" make_packet.py')
    print(f"  interpreter: {sys.executable}")


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  copied {dst.relative_to(HERE)}")


def main():
    print("== preflight ==")
    preflight()
    for p, name in [(R2P, "previous package"), (V3A, "measurement outputs")]:
        if not p.exists():
            fail(f"{name} not found at {p}")
    for f in NEW_SCRIPTS:
        if not (HERE / f).exists():
            fail(f"missing new file {f}")

    print("== copying unchanged pipeline files ==")
    for f in UNCHANGED:
        copy(R2P / f, HERE / f)
    copy(R2P / "data" / "urm_keys.geojson", HERE / "data" / "urm_keys.geojson")

    print("== 03_benchmark.py: map-model setting ==")
    src = (R2P / "03_benchmark.py").read_text(encoding="utf-8")
    old_call = "_figure(st, lab, present, west, best_model, args, scene)"
    new_call = "_figure(st, lab, present, west, args.map_model or best_model, args, scene)"
    old_arg = 'ap.add_argument("--figdir", default=str(HERE / "figures"))'
    new_arg = (old_arg + '\n    ap.add_argument("--map_model", default="LogReg",\n'
               '                    help="classifier used for the map figures (default LogReg, '
               'matching the paper); pass an empty string to use each scene\'s best model")')
    if src.count(old_call) != 1:
        fail(f"03_benchmark map call found {src.count(old_call)} times (expected 1)")
    if src.count(old_arg) != 1:
        fail(f"03_benchmark figdir argument found {src.count(old_arg)} times (expected 1)")
    src = src.replace(old_call, new_call).replace(old_arg, new_arg)
    (HERE / "03_benchmark.py").write_text(src, encoding="utf-8")
    print("  patched 03_benchmark.py (--map_model, default LogReg)")

    print("== reference_outputs: unchanged benchmark artifacts ==")
    for f in REF_KEEP:
        copy(R2P / "reference_outputs" / f, HERE / "reference_outputs" / f)
    for f in REF_FIGS_KEEP:
        copy(R2P / "reference_outputs" / "figures" / f,
             HERE / "reference_outputs" / "figures" / f)

    print("== reference_outputs: artifacts from the reported measurement run ==")
    copy(V3A / "outputs" / "e4e_energy_urm.csv", HERE / "reference_outputs" / "energy_per_stage.csv")
    copy(V3A / "outputs" / "e4_train_draw_urm.csv", HERE / "reference_outputs" / "train_draw_digest.csv")
    copy(V3A / "outputs" / "e4f_svm_support_urm.csv", HERE / "reference_outputs" / "svm_support.csv")
    copy(V3A / "tables" / "test_cap_sensitivity.csv", HERE / "reference_outputs" / "test_cap_sensitivity.csv")
    for src_name, dst_name in [("fig11_spring.png", "map_spring.png"),
                               ("fig12_dry.png", "map_dry.png"),
                               ("fig13_wet.png", "map_wet.png")]:
        copy(V3A / "figures" / src_name, HERE / "reference_outputs" / "figures" / dst_name)

    print("== regenerating statistics reference CSVs (03b, stratified) ==")
    rc = subprocess.run([sys.executable, str(HERE / "03b_statistics.py"),
                         "--metrics", str(HERE / "reference_outputs" / "benchmark_block_metrics.csv"),
                         "--out", str(HERE / "reference_outputs")]).returncode
    if rc != 0:
        fail("03b_statistics.py returned nonzero")
    import platform
    import numpy, pandas, scipy, matplotlib
    (HERE / "reference_outputs" / "environment.txt").write_text(
        f"python {platform.python_version()}\nnumpy {numpy.__version__}\n"
        f"scipy {scipy.__version__}\npandas {pandas.__version__}\n"
        f"matplotlib {matplotlib.__version__}\n")
    print("  wrote reference_outputs/environment.txt")

    print("== regenerating energy figures (fig3, fig6) from reference CSVs ==")
    import importlib.util
    import pandas as pd
    spec = importlib.util.spec_from_file_location("figs", HERE / "06_figures.py")
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    summ = pd.read_csv(HERE / "reference_outputs" / "benchmark_summary.csv")
    fe = pd.read_csv(HERE / "reference_outputs" / "energy_per_stage.csv")
    figdir = HERE / "reference_outputs" / "figures"
    figs.fig_accuracy_energy(summ, fe, figdir)
    figs.fig_pareto(summ, fe, figdir)

    print("== working directories ==")
    for d in ["data", "outputs", "figures"]:
        (HERE / d).mkdir(exist_ok=True)
        (HERE / d / ".gitkeep").write_text("")

    print("== verification ==")
    # manifest
    expected = (UNCHANGED + NEW_SCRIPTS + ["03_benchmark.py"]
                + ["data/urm_keys.geojson"]
                + [f"reference_outputs/{f}" for f in REF_KEEP]
                + ["reference_outputs/energy_per_stage.csv",
                   "reference_outputs/train_draw_digest.csv",
                   "reference_outputs/svm_support.csv",
                   "reference_outputs/test_cap_sensitivity.csv",
                   "reference_outputs/stats_block_ci.csv",
                   "reference_outputs/stats_pairwise.csv",
                   "reference_outputs/stats_scene_level.csv",
                   "reference_outputs/environment.txt"]
                + [f"reference_outputs/figures/{f}" for f in REF_FIGS_KEEP]
                + ["reference_outputs/figures/fig3_accuracy_energy.png",
                   "reference_outputs/figures/fig6_accuracy_vs_energy.png",
                   "reference_outputs/figures/map_spring.png",
                   "reference_outputs/figures/map_dry.png",
                   "reference_outputs/figures/map_wet.png"])
    missing = [f for f in expected if not (HERE / f).exists()]
    if missing:
        fail(f"missing files: {missing}")
    print(f"  manifest complete ({len(expected)} files)")

    # energy CSV: 189 rows; single-pass SVM/kNN inference; 5 repeats elsewhere
    rows = list(csv.DictReader(open(HERE / "reference_outputs" / "energy_per_stage.csv")))
    if len(rows) != 189:
        fail(f"energy_per_stage.csv has {len(rows)} rows (expected 189)")
    sv = [r for r in rows if r["stage"] == "fullmap_infer" and r["model"] in ("SVM", "kNN")]
    if len(sv) != 18 or any(r["repeats"] != "1" for r in sv):
        fail("SVM/kNN inference rows are not 18 single-pass measurements")
    if any(r["repeats"] != "5" for r in rows if r not in sv):
        fail("a non-SVM/kNN-inference row does not have 5 repeats")
    print("  energy_per_stage.csv: 189 rows; repeats pattern correct")

    # statistics: Table 10 digits and significance counts
    ci = {(r["protocol"], r["model"]): r for r in
          csv.DictReader(open(HERE / "reference_outputs" / "stats_block_ci.csv"))}
    checks = [("geo_holdout", "RF", "0.5139", "0.4906", "0.5384"),
              ("geo_holdout", "LogReg", "0.5485", "0.5211", "0.5748"),
              ("spatial_cv", "SVM", "0.5726", "0.5515", "0.5935")]
    for p, m, oa, lo, hi in checks:
        r = ci[(p, m)]
        if (r["OA"], r["CI_lo"], r["CI_hi"]) != (oa, lo, hi):
            fail(f"stats_block_ci {p}/{m} = ({r['OA']}, {r['CI_lo']}, {r['CI_hi']}), "
                 f"expected ({oa}, {lo}, {hi})")
    pw = list(csv.DictReader(open(HERE / "reference_outputs" / "stats_pairwise.csv")))
    geo = sum(r["significant"] == "True" for r in pw if r["protocol"] == "geo_holdout")
    cv = sum(r["significant"] == "True" for r in pw if r["protocol"] == "spatial_cv")
    if (geo, cv) != (17, 18):
        fail(f"pairwise significant counts geo={geo}, cv={cv} (expected 17, 18)")
    sl = list(csv.DictReader(open(HERE / "reference_outputs" / "stats_scene_level.csv")))
    if sum(r["significant"] == "True" for r in sl) != 0:
        fail("scene-level test shows significant pairs (expected none)")
    print("  statistics: Table 10 spot checks pass; 17/18 bootstrap pairs; 0 scene-level")

    # sensitivity and digests
    sens = list(csv.DictReader(open(HERE / "reference_outputs" / "test_cap_sensitivity.csv")))
    mx = max(abs(float(r["dOA"])) for r in sens)
    if len(sens) != 81 or round(mx, 4) != 0.0047:
        fail(f"test_cap_sensitivity: {len(sens)} rows, max |dOA| {mx} (expected 81, 0.0047)")
    dig = list(csv.DictReader(open(HERE / "reference_outputs" / "train_draw_digest.csv")))
    sup = list(csv.DictReader(open(HERE / "reference_outputs" / "svm_support.csv")))
    if len(dig) != 9 or len(sup) != 9:
        fail(f"digests {len(dig)}, support rows {len(sup)} (expected 9, 9)")
    ns = [int(r["n_support"]) for r in sup]
    if min(ns) != 9565 or max(ns) != 11358:
        fail(f"support-vector range {min(ns)}-{max(ns)} (expected 9565-11358)")
    print("  sensitivity (81 rows, max 0.0047), digests (9), support counts (9,565-11,358)")

    print("== zip ==")
    import zipfile
    skip = {"make_packet.py"}
    zpath = HERE.parent / (HERE.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.rglob("*")):
            rel = p.relative_to(HERE)
            if p.is_dir() or p.name in skip or "__pycache__" in rel.parts or p.suffix == ".pyc":
                continue
            z.write(p, arcname=str(Path(HERE.name) / rel))
    print(f"  wrote {zpath} (assembly script excluded)")
    print("\nPacket assembled and verified.")


if __name__ == "__main__":
    main()
