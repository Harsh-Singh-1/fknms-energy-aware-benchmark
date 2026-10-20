"""
run_all_overnight.py -- extended-analyses runner.

Runs the extended analyses in order, logging to run_all.log next to this
script. Safe to re-run: completed learning-curve sizes are detected from the
merged CSV (36 rows per size = 9 scenes x 2 models x 2 protocols) and skipped,
and 07's merge logic replaces rather than duplicates on a re-run.

Order (cheap and independent first, so early failures surface fast):
    1. 09_block_audit.py            instant       (block-count audit)
    2. 08_holdout_majority_rate.py  minutes       (test-pixel majority baselines)
    3. 10_stats_additions.py        instant       (Friedman omnibus and kappa)
    4. 07_fig9_extension.py 20000   hours         (learning-curve extension)
    5. 07_fig9_extension.py 50000   hours         (learning-curve extension)
    6. 06_figures.py                minutes       (regenerates Figures 3 and 9)

Run from this folder:
    python run_all_overnight.py
Options:
    --sizes 20000 50000     learning-curve sizes to ensure (default both)
    --skip 07               step numbers to skip, e.g. --skip 08 09
    --stop_on_error         halt the sequence on the first failure
                            (default: log the failure, continue with later
                            steps that do not depend on it; 06 always runs
                            last with whatever outputs exist)

Suggested overnight invocation (keeps the console log too):
    python run_all_overnight.py > run_all_console.txt 2>&1

Author: Harsh Deep Singh Narula
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "run_all.log"
EXPECTED_ROWS_PER_SIZE = 36   # 9 scenes x {MLP, 2D_CNN} x {spatial_cv, geo_holdout}


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, cmd, stop_on_error):
    log(f"START {name}: {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    dt = time.time() - t0
    if r.returncode == 0:
        log(f"OK    {name} ({dt/60:.1f} min)")
        return True
    log(f"FAIL  {name} (exit {r.returncode}, {dt/60:.1f} min)")
    if stop_on_error:
        log("Stopping on first error (--stop_on_error).")
        sys.exit(r.returncode)
    return False


def size_done(size):
    """A learning-curve size counts as done when the merged CSV already holds
    the full 36 rows for it."""
    merged = HERE / "outputs" / "ablation_learning_curve.csv"
    if not merged.exists():
        return False
    with open(merged, newline="") as f:
        n = sum(1 for r in csv.DictReader(f) if r.get("nsamp") == str(size))
    if 0 < n < EXPECTED_ROWS_PER_SIZE:
        log(f"NOTE  size {size}: {n}/{EXPECTED_ROWS_PER_SIZE} rows present "
            f"(incomplete earlier run?); re-running it. 07's merge replaces "
            f"these rows, so no duplicates result.")
    return n >= EXPECTED_ROWS_PER_SIZE


def main():
    ap = argparse.ArgumentParser(description="Run the extended analyses in order.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[20000, 50000])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step numbers to skip, e.g. --skip 08 09")
    ap.add_argument("--stop_on_error", action="store_true")
    ap.add_argument("--data", default=None,
                    help="scene directory; omit when scenes live in ./data")
    a = ap.parse_args()
    py = sys.executable
    data_args = ["--data", a.data] if a.data else []

    log("=" * 60)
    log(f"Extended-analyses runner starting | sizes={a.sizes} | skip={a.skip}")

    if "09" not in a.skip:
        run_step("09 block audit", [py, "09_block_audit.py"], a.stop_on_error)
    if "08" not in a.skip:
        run_step("08 majority baselines",
                 [py, "08_holdout_majority_rate.py", *data_args], a.stop_on_error)
    if "10" not in a.skip:
        run_step("10 Friedman + kappa", [py, "10_stats_additions.py"], a.stop_on_error)

    if "07" not in a.skip:
        for size in a.sizes:
            if size_done(size):
                log(f"SKIP  07 learning-curve size {size}: already complete "
                    f"({EXPECTED_ROWS_PER_SIZE} rows in merged CSV)")
                continue
            run_step(f"07 learning-curve extension n={size}",
                     [py, "07_fig9_extension.py", *data_args, "--sizes", str(size)],
                     a.stop_on_error)

    if "06" not in a.skip:
        run_step("06 regenerate figures", [py, "06_figures.py"], a.stop_on_error)

    log("Runner finished. Review run_all.log, outputs/, and figures/.")
    log("Check learning_curve_draws.csv for any '!! draw shortfall' events "
        "before quoting the new Fig. 9 sizes.")


if __name__ == "__main__":
    main()
