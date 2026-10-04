#!/usr/bin/env bash
# Test day: score new organiser images and write the final submission. Compute runs on Modal; the last step is local. See docs/patch_mil.md (Test day runbook).
set -euo pipefail
cd "$(dirname "$0")/.."
ls data_test/Batch_test/img_*_BSE.tif >/dev/null 2>&1 || { echo "copy organiser TIFFs to data_test/Batch_test/img_<site>_<BSE|Inlens|ETD|SE>.tif"; exit 1; }
test -f outputs/menu/selection.json || { echo "outputs/menu/selection.json missing: the menu must be frozen (modal run modal_patch_mil.py --mode menu) before test images"; exit 1; }
modal run modal_test_prep.py::main
modal run modal_patch_mil.py --mode test
# fem_a1 (the selected model) needs free-lateral FEM curves on the test sites
sites=$(ls data_test/Batch_test/img_*_BSE.tif | sed -E 's#.*/img_(.*)_BSE\.tif#Batch_test/\1#' | paste -sd, -)
for o in bottom top; do
  modal run modal_fem.py --mode full --orientation "$o" --sites "$sites" --tag full_free_test --solver-overrides '{"lateral":"left"}'
done
tmp=$(mktemp -d)
python scripts/fem_collect.py --edge outputs/modal/fem/results/full_free_test "$tmp"
mkdir -p outputs/fem/free_lateral_test
python - "$tmp" <<'PY'
import sys, pandas as pd
for n in ("site_curves.csv", "tile_curves.csv"):
    d = pd.read_csv(f"{sys.argv[1]}/{n}", dtype={"site": str})
    d[d["batch"] == "Batch_test"].to_csv(f"outputs/fem/free_lateral_test/{n}", index=False, float_format="%.6g")
PY
# any failure above aborts before this, so a stale submission is never left behind
python scripts/score_test_all.py
