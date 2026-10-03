#!/usr/bin/env bash
# One command: cache any new Batch_* folder, extract KPIs + per-site GPs, compare batches, write report.
# Usage: ./analytical_benchmarks/run_all.sh   (drop a new batch into data/Batch_N first)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PMDB=${PMDB:-$(dirname "$HERE")}
(cd "$PMDB" && python3 scripts/build_cache.py)          # skips sites already cached
cd "$HERE"
export PYTHONPATH="$HERE:$PMDB" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 kpis.py 2>&1 | grep -v -i warn
python3 compare.py site_kpis.csv | tee verdicts.txt
python3 importance.py 2>&1 | grep -v -i warn
python3 overlays.py
python3 spots.py 2>&1 | grep -v -i warn
python3 figs.py 2>&1 | grep -v -i warn || true
python3 report.py && echo "report: $HERE/qc_report.html"
